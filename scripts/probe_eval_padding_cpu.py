"""Check real two-rank evaluation reductions using a tiny CPU model."""

import json
import math
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


def worker(rank, port):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      LOCAL_WORLD_SIZE="2", MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port), CUDA_VISIBLE_DEVICES="")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project / "tests"))
    from accelerate import Accelerator
    from PIL import Image
    from fastwam_fixtures import make_model, make_sample
    from model.fastwam import FastWAMIDM
    from data.lerobot.transforms.action_state_merger import ConcatLeftAlign
    from trainer import EasyWAMTrainer
    import trainer as trainer_module

    assert Path(trainer_module.__file__).resolve() == project / "src/trainer.py"
    accelerator = Accelerator(cpu=True)
    assert accelerator.num_processes == 2 and accelerator.device.type == "cpu"
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = make_model(FastWAMIDM).eval()
    raw = {key: value[0] if isinstance(value, torch.Tensor) else value
           for key, value in make_sample().items()}
    raw["prompt"] = "short"
    raw["video"].fill_(float(rank))
    raw["action"].fill_(1.0 + 2.0 * rank)
    raw["image_is_pad"] = torch.tensor([False] * 5 if rank == 0 else [False] + [True] * 4)
    raw["action_is_pad"] = torch.tensor([False] * 4 if rank == 0 else [False] + [True] * 3)
    raw["action_dim_is_pad"] = torch.tensor([False, True, False] if rank == 0 else [False] * 3)
    raw["proprio_is_pad"] = torch.zeros(raw["proprio"].shape[0], dtype=torch.bool)
    frames = [Image.new("RGB", (32, 16), "black") for _ in range(5)]
    model.infer = lambda **kwargs: {"video": frames, "action": torch.zeros(4, 3)}
    shape_meta = {"action": [{"key": "default", "shape": 3}],
                  "state": [{"key": "default", "shape": 4}]}
    merger = ConcatLeftAlign()
    merger.set_shape_meta(shape_meta)
    processor = SimpleNamespace(shape_meta=shape_meta, action_state_merger=merger,
                                normalizer=SimpleNamespace(backward=lambda batch: batch))

    class Dataset:
        lerobot_dataset = SimpleNamespace(processor=processor)

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return raw

    trainer = EasyWAMTrainer.__new__(EasyWAMTrainer)
    trainer.model = model
    trainer.val_dataset = Dataset()
    trainer.global_step = 0
    trainer.eval_num_inference_steps = 1
    trainer.eval_save_video = False
    trainer.accelerator = accelerator
    result = trainer.evaluate()
    assert result["video_valid_frames"] == 6
    assert result["action_valid_elements"] == 11
    assert math.isclose(result["action_l1"], 17.0 / 11, rel_tol=1e-6)
    assert math.isclose(result["action_l2"], 35.0 / 11, rel_tol=1e-6)
    expected_psnr = (5 * 10 * math.log10(1 / (0.25 + 1e-8))
                     + 10 * math.log10(1 / (1 + 1e-8))) / 6
    assert math.isclose(result["psnr_rg"], expected_psnr, rel_tol=1e-6)
    raw["image_is_pad"].fill_(True)
    raw["action_is_pad"].fill_(True)
    empty = trainer.evaluate()
    assert empty["video_valid_frames"] == empty["action_valid_elements"] == 0
    assert "psnr_rg" not in empty and "ssim_rd" not in empty
    assert "action_l1" not in empty and "action_l2" not in empty
    assert math.isfinite(empty["val_loss"])
    print(json.dumps({"rank": rank, "weighted_reduction_passed": True,
                      "all_padding_omission_passed": True}), flush=True)
    accelerator.wait_for_everyone()
    torch.distributed.destroy_process_group()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port,), nprocs=2, join=True)
    print(json.dumps({"passed": True, "backend": "gloo", "ranks": 2}), flush=True)


if __name__ == "__main__":
    main()
