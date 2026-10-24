"""CPU checks for text-cache execution controls and model-loading decisions."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts import precompute_text_embeds as precompute


def make_config(tmp_path, **settings):
    config = {
        "model": {
            "backbone": {
                "name": "wan22",
                "model_id": str(tmp_path / "video-backbone"),
                "text_encoder_path": str(tmp_path / "separate-t5.pth"),
                "tokenizer_model_id": str(tmp_path / "separate-tokenizer"),
                "text_encoder_id": "wan22ti2v5b",
            }
        },
        "data": {
            "train": {
                "dataset_dirs": [str(tmp_path / "dataset")],
                "text_embedding_cache_dir": str(tmp_path / "cache"),
                "context_len": 4,
            }
        },
        "overwrite": False,
        "text_encoder_device": "cpu",
        "text_encoder_dtype": "float32",
    }
    config.update(settings)
    return OmegaConf.create(config)


def cache_path(cfg, prompt):
    return precompute.Path(cfg.data.train.text_embedding_cache_dir) / (
        precompute.text_embedding_cache_filename(prompt, 4, "wan22ti2v5b")
    )


def save_cache(cfg, prompt):
    payload = precompute.build_text_embedding_payload(
        context=torch.ones(4, 3),
        mask=torch.tensor([True, True, False, False]),
        context_len=4,
        encoder_id="wan22ti2v5b",
        prompt_digest=precompute.prompt_hash(prompt),
    )
    precompute._atomic_torch_save(payload, cache_path(cfg, prompt))


@pytest.fixture
def no_cuda(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Explicit CPU precomputation must not query or select CUDA.")

    for name in ("is_available", "device_count", "set_device"):
        monkeypatch.setattr(torch.cuda, name, fail)
    monkeypatch.setenv("WORLD_SIZE", "1")


def install_fake_encoder(monkeypatch, cfg):
    calls = []

    def resolve_configs(**kwargs):
        assert kwargs == {
            "model_id": cfg.model.backbone.model_id,
            "tokenizer_model_id": cfg.model.backbone.tokenizer_model_id,
            "text_encoder_path": cfg.model.backbone.text_encoder_path,
        }
        text_config = SimpleNamespace(
            path=kwargs["text_encoder_path"], resolve=lambda: None
        )
        tokenizer_config = SimpleNamespace(
            path=kwargs["tokenizer_model_id"], resolve=lambda: None
        )
        return None, text_config, None, tokenizer_config

    class Encoder:
        def eval(self):
            return self

        def __call__(self, ids, mask):
            assert ids.device.type == mask.device.type == "cpu"
            return torch.full((*ids.shape, 3), 2.25, dtype=torch.float32)

    def load_encoder(path, name, **kwargs):
        calls.append((path, name, kwargs))
        return Encoder()

    def tokenizer_factory(**kwargs):
        assert kwargs == {
            "name": cfg.model.backbone.tokenizer_model_id,
            "seq_len": 4,
            "clean": "whitespace",
        }

        def tokenize(prompts, **kwargs):
            return (
                torch.ones(len(prompts), 4, dtype=torch.long),
                torch.tensor([[1, 1, 0, 0]] * len(prompts)),
            )

        return tokenize

    monkeypatch.setattr(precompute, "_resolve_configs", resolve_configs)
    monkeypatch.setattr(precompute, "_load_registered_model", load_encoder)
    monkeypatch.setattr(precompute, "HuggingfaceTokenizer", tokenizer_factory)
    return calls


def test_explicit_cpu_paths_and_dtype_write_training_compatible_cache(
    tmp_path, monkeypatch, no_cuda
):
    cfg = make_config(tmp_path)
    monkeypatch.setattr(precompute, "_read_unique_prompts", lambda _: ["task"])
    calls = install_fake_encoder(monkeypatch, cfg)

    precompute.main.__wrapped__(cfg)

    assert calls == [
        (
            cfg.model.backbone.text_encoder_path,
            "wan_video_text_encoder",
            {"torch_dtype": torch.float32, "device": "cpu"},
        )
    ]
    payload = torch.load(cache_path(cfg, "task"), weights_only=True)
    assert payload["format_version"] == 3
    assert payload["encoder_id"] == "wan22ti2v5b"
    assert payload["context"].dtype == torch.bfloat16
    assert payload["mask"].dtype == torch.bool
    assert torch.equal(payload["context"][:2], torch.full((2, 3), 2.25))
    assert torch.count_nonzero(payload["context"][2:]) == 0


@pytest.mark.parametrize("rank", [None, 0, 1])
def test_complete_or_empty_rank_skips_encoder_but_keeps_collectives(
    tmp_path, monkeypatch, no_cuda, rank
):
    cfg = make_config(tmp_path)
    save_cache(cfg, "cached-task")
    original = cache_path(cfg, "cached-task").read_bytes()
    monkeypatch.setattr(
        precompute, "_read_unique_prompts", lambda _: ["cached-task"]
    )

    def fail(*args, **kwargs):
        pytest.fail("A rank without missing prompts must not resolve or load assets.")

    monkeypatch.setattr(precompute, "_resolve_configs", fail)
    monkeypatch.setattr(precompute, "_load_registered_model", fail)
    monkeypatch.setattr(precompute, "HuggingfaceTokenizer", fail)
    collectives = []
    if rank is not None:
        monkeypatch.setattr(
            precompute, "_init_distributed", lambda _: (True, rank, 2, rank)
        )
        monkeypatch.setattr(
            precompute.dist, "all_reduce",
            lambda tensor, **kwargs: collectives.append(tuple(tensor.shape)),
        )
        monkeypatch.setattr(precompute.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(precompute.dist, "get_rank", lambda: rank)
        monkeypatch.setattr(
            precompute.dist, "barrier", lambda: collectives.append("barrier")
        )
        monkeypatch.setattr(
            precompute.dist, "destroy_process_group",
            lambda: collectives.append("destroy"),
        )

    precompute.main.__wrapped__(cfg)

    assert cache_path(cfg, "cached-task").read_bytes() == original
    if rank is not None:
        assert collectives == [(2,), (1,), (1, 3), "barrier", "destroy"]


def test_partially_cached_prompts_only_encode_missing_items(
    tmp_path, monkeypatch, no_cuda
):
    cfg = make_config(tmp_path)
    save_cache(cfg, "old-task")
    original = cache_path(cfg, "old-task").read_bytes()
    monkeypatch.setattr(
        precompute, "_read_unique_prompts", lambda _: ["old-task", "new-task"]
    )
    calls = install_fake_encoder(monkeypatch, cfg)

    precompute.main.__wrapped__(cfg)

    assert len(calls) == 1
    assert cache_path(cfg, "old-task").read_bytes() == original
    assert cache_path(cfg, "new-task").is_file()


def test_cpu_distributed_initialization_uses_gloo(monkeypatch, no_cuda):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    calls = []
    monkeypatch.setattr(precompute.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        precompute.dist, "init_process_group", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(precompute.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(precompute.dist, "get_world_size", lambda: 2)

    assert precompute._init_distributed("cpu") == (True, 1, 2, 1)
    assert calls == [{"backend": "gloo", "init_method": "env://"}]


def test_default_gpu_dtype_and_local_rank_are_preserved(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert precompute._resolve_compute_settings(OmegaConf.create({})) == (
        "cuda:2", torch.bfloat16
    )


def test_unsupported_dtype_fails_before_loading(tmp_path, no_cuda):
    cfg = make_config(tmp_path, text_encoder_dtype="float64")
    with pytest.raises(ValueError, match="Unsupported text_encoder_dtype"):
        precompute.main.__wrapped__(cfg)
