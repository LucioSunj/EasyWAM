"""Exercise the checkpoint overwrite guard with two CPU/Gloo ranks."""

import json
import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp


def worker(rank, world_size, port, directory):
    os.environ.update(
        RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank),
        LOCAL_WORLD_SIZE=str(world_size), MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    from accelerate import Accelerator
    from trainer import EasyWAMTrainer
    import trainer as trainer_module

    project = Path(__file__).resolve().parents[1]
    assert Path(trainer_module.__file__).resolve() == project / "src/trainer.py"
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == world_size
    assert accelerator.device.type == "cpu"
    torch.set_num_threads(1)

    state_root = Path(directory) / f"rank-{rank}"
    state_root.mkdir()
    target = state_root / "step_000005"
    # Deliberately give only rank zero a view of a completed marker.
    if rank == 0:
        target.mkdir()
        (target / "trainer_state.json").write_text("existing completion\n")
    accelerator.wait_for_everyone()
    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.accelerator = accelerator
    trainer.global_step = 5
    trainer.state_dir = str(state_root)

    def unexpected_write(**kwargs):
        raise AssertionError("Guard allowed a checkpoint write")

    trainer._save_weights_checkpoint = unexpected_write
    try:
        trainer.save_checkpoint()
    except FileExistsError:
        pass
    else:
        raise AssertionError("Every rank must reject the overwrite")
    if rank == 0:
        assert (target / "trainer_state.json").read_text() == "existing completion\n"
    else:
        assert not target.exists()
    accelerator.wait_for_everyone()
    print(json.dumps({"rank": rank, "device": "cpu", "rejected_before_writes": True}), flush=True)
    torch.distributed.destroy_process_group()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="easywam-checkpoint-guard-") as directory:
        mp.spawn(worker, args=(2, port, directory), nprocs=2, join=True)
    print(json.dumps({"passed": True, "backend": "gloo", "ranks": 2}), flush=True)


if __name__ == "__main__":
    main()
