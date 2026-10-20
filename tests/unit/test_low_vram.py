import gc
import inspect
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from kandinsky.core.components.text_embedder import Kandinsky6TextEmbedder
from kandinsky.pipeline import reconfigure
from kandinsky.pipeline.config import load_config
from kandinsky.pipeline.pipeline import Kandinsky6Pipeline, _release_modules_after_call
from kandinsky.pipeline.reconfigure import _text_needs_reload
from kandinsky.runtime.offload import reject_release_without_offload
from kandinsky.runtime.offload.block import BlockOffload
from kandinsky.runtime.offload.module import ModuleOffload
from kandinsky.runtime.profile import NoOpProfile, TimeMemModuleProfile

_DEVICES = Path(__file__).resolve().parents[2] / "kandinsky" / "configs" / "devices"


def test_offload_wrapper_preserves_compiled_bound_forward_and_hooks():
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = 3

        @torch.compile(backend=backend)
        def forward(self, value):
            return value + self.offset

    events = []

    @contextmanager
    def use_block(group, following):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    handle = BlockOffload.__new__(BlockOffload)
    handle._use_block = use_block
    block = Block()
    block.forward = handle._decorate(block.forward, object(), None)
    assert block(torch.ones(1)).item() == 4
    assert events == ["enter", "exit"]
    assert graphs, "Regional compile was disabled along with transfer orchestration"


def test_text_embedder_reloads_on_the_shared_load_path(monkeypatch):
    encoder = Kandinsky6TextEmbedder.__new__(Kandinsky6TextEmbedder)
    encoder.device = torch.device("cpu")
    encoder.quantized_qwen = True
    encoder.max_length = 8
    encoder.text_token_padding = False
    encoder._qwen_path = "qwen"
    encoder._clip_path = "clip"
    encoder.processor = object()
    encoder.tokenizer = object()
    encoder.qwen = SimpleNamespace(device=torch.device("cpu"))
    encoder.clip = SimpleNamespace(device=torch.device("cpu"))
    loaded = []

    def load(self):
        loaded.append(self.device)
        self.qwen = SimpleNamespace(device=self.device)
        self.clip = SimpleNamespace(device=self.device)

    monkeypatch.setattr(Kandinsky6TextEmbedder, "_load_models", load)
    encoder.release()
    assert encoder.qwen is encoder.clip is None
    encoder.ensure_loaded()
    encoder.ensure_loaded()
    assert loaded == [torch.device("cpu")]
    encoder.device = torch.device("meta")
    encoder.ensure_loaded()
    assert loaded == [torch.device("cpu"), torch.device("meta")]


def test_encode_returns_tensors_that_do_not_alias_model_storage():
    encoder = Kandinsky6TextEmbedder.__new__(Kandinsky6TextEmbedder)
    encoder.device = torch.device("cpu")
    encoder.max_length = 4
    encoder.qwen = SimpleNamespace(device=torch.device("cpu"))
    encoder.clip = SimpleNamespace(device=torch.device("cpu"))
    encoder.text_token_padding = True
    shared = torch.arange(133 * 3, dtype=torch.float32).reshape(1, 133, 3)
    pooled = torch.arange(4, dtype=torch.float32).reshape(1, 4)

    class Batch(dict):
        def to(self, _device):
            return self

    encoder.processor = lambda **_kwargs: Batch(
        {
            "input_ids": torch.zeros(1, 133, dtype=torch.long),
            "attention_mask": torch.ones(1, 133, dtype=torch.long),
        }
    )
    encoder.encode_qwen = lambda *_args, **_kwargs: {"hidden_states": [shared]}
    encoder.tokenizer = lambda *_args, **_kwargs: Batch({"input_ids": torch.zeros(1, 77, dtype=torch.long)})
    encoder.encode_clip = lambda _inputs: {"pooler_output": pooled}

    embeds, _cu_seqlens, _mask = encoder.encode(["hello"])
    assert embeds["text_embeds"].data_ptr() != shared[:, 129:].data_ptr()
    assert embeds["pooled_embed"].data_ptr() != pooled.data_ptr()
    assert torch.equal(embeds["text_embeds"], shared[0, 129:])
    assert torch.equal(embeds["pooled_embed"], pooled)


def test_profiler_measures_facade_methods_and_drops_weights():
    class Text:
        def __init__(self):
            self.qwen = torch.nn.Linear(1, 1)
            self.clip = torch.nn.Linear(1, 1)

        def encode_qwen(self, *args, **kwargs):
            return self.qwen(*args, **kwargs)

        def encode_clip(self, *args, **kwargs):
            return self.clip(*args, **kwargs)

        def release(self):
            self.qwen = self.clip = None

    text = Text()
    pipe = SimpleNamespace(text_embedder=text)
    timed = TimeMemModuleProfile()
    timed.apply(pipe)
    assert "encode_qwen" in text.__dict__
    assert "encode_clip" in text.__dict__
    assert "forward" not in text.qwen.__dict__
    old_qwen = weakref.ref(text.qwen)
    text.release()
    gc.collect()
    assert old_qwen() is None
    timed.clear(pipe)


def test_call_does_not_branch_on_the_release_flag():
    source = inspect.getsource(Kandinsky6Pipeline.__call__)
    source += inspect.getsource(_release_modules_after_call)
    assert "release_after_encode" not in source
    assert "release_text_encoder" not in source
    assert "ensure_loaded" not in source


def test_release_flag_requires_offload():
    with pytest.raises(ValueError, match="release_after_encode"):
        reject_release_without_offload("none", True)
    reject_release_without_offload("module", True)
    reject_release_without_offload("block", True)
    reject_release_without_offload("none", False)


def test_text_reload_ignores_release_flag():
    cfg = load_config(_DEVICES / "h100.yaml")
    flipped = cfg.model_copy(
        update={"text_embedder": cfg.text_embedder.model_copy(update={"release_after_encode": True})}
    )
    assert _text_needs_reload(cfg, flipped) is False
    quantized = cfg.model_copy(
        update={"text_embedder": cfg.text_embedder.model_copy(update={"quantized_qwen": True})}
    )
    assert _text_needs_reload(cfg, quantized) is True


def test_low_vram_preset_keeps_decode_fields_in_sr():
    cfg = load_config(_DEVICES / "rtx-5060-ti.yaml")
    assert cfg.offload.strategy == "block"
    assert cfg.text_embedder.quantized_qwen is True
    assert cfg.text_embedder.release_after_encode is True
    assert cfg.sr.decode_segment_frames == 4
    assert cfg.sr.decode_spatial_splits == 2
    assert cfg.sr.decode_spatial_overlap == 4


def test_module_offload_loads_on_entry_and_skips_prefetch():
    prefetched = []
    embedder = SimpleNamespace(device=torch.device("cpu"), loads=0, drops=0)

    def ensure_loaded():
        embedder.loads += 1

    def release():
        embedder.drops += 1

    embedder.ensure_loaded = ensure_loaded
    embedder.release = release
    handle = ModuleOffload.__new__(ModuleOffload)
    handle._modules = {"text_embedder": embedder, "dit": object()}
    handle._release = {"text_embedder"}
    handle._on_compute = set()
    handle._ready = {}
    handle.compute_device = torch.device("cpu")
    handle.prefetch = lambda *names: prefetched.extend(names)
    handle.empty_cache = lambda: None

    with handle.use("text_embedder", prefetch="dit"):
        assert embedder.loads == 1
        assert embedder.device == torch.device("cpu")
    assert embedder.drops == 1
    assert prefetched == []


def test_block_offload_release_stage_is_not_a_tensor_group():
    events = []
    embedder = SimpleNamespace(device=torch.device("meta"))

    def ensure_loaded():
        events.append(("load", embedder.device))

    embedder.ensure_loaded = ensure_loaded
    embedder.release = lambda: events.append("drop")
    handle = BlockOffload.__new__(BlockOffload)
    handle._active = set()
    handle._stages = {}
    handle._stage_roots = {}
    handle._release_modules = {}
    handle.compute_device = torch.device("cpu")
    handle.register("text_embedder", embedder, release=True)
    assert "text_embedder" not in handle._stages
    with torch.no_grad():
        with handle.use("text_embedder", prefetch="dit"):
            assert events == [("load", torch.device("cpu"))]
    assert events == [("load", torch.device("cpu")), "drop"]


def test_release_flag_retargets_text_without_rebuilding_offload(monkeypatch):
    def fail_build(*_args, **_kwargs):
        raise AssertionError("offload handle was rebuilt")

    monkeypatch.setattr(reconfigure, "build_offload", fail_build)
    registered = []
    handle = SimpleNamespace(
        strategy="block",
        pin_memory=True,
        releases=lambda _name: False,
        register=lambda name, module, *, release=False: registered.append((name, release)),
    )
    pipe = SimpleNamespace(
        offload=handle,
        device=torch.device("cpu"),
        text_embedder=SimpleNamespace(qwen=object(), clip=object()),
    )
    reconfigure._apply_offload(pipe, "block", True, force=False, release_text=True)
    assert registered == [("text_embedder", True)]
    registered.clear()
    handle.releases = lambda _name: True
    reconfigure._apply_offload(pipe, "block", True, force=False, release_text=True)
    assert registered == []


def test_reconfigure_registers_releasable_text(monkeypatch):
    registered = []

    def register(name, module, *, release=False):
        del module
        registered.append((name, release))

    handle = SimpleNamespace(register=register, strategy="block", pin_memory=True)
    monkeypatch.setattr(reconfigure, "build_offload", lambda *_args, **_kwargs: handle)
    pipe = SimpleNamespace(
        offload=SimpleNamespace(strategy="none", pin_memory=True),
        device=torch.device("cpu"),
        text_embedder=SimpleNamespace(qwen=object(), clip=object()),
        dit=object(),
        vae=object(),
        audio_vae=None,
        vocoder=None,
    )
    reconfigure._apply_offload(pipe, "block", True, force=True, release_text=True)
    assert registered == [("text_embedder", True), ("dit", False), ("vae", False)]


@pytest.mark.parametrize("warmup_enabled,expected_calls", [(False, 1), (True, 2)])
def test_cli_sr_no_grad_and_warmup_boolean(tmp_path, monkeypatch, warmup_enabled, expected_calls):
    from kandinsky import cli
    from kandinsky_sr.core.utils.offload import NoOpOffload
    from kandinsky_sr.pipeline import components, warmup

    calls = []

    class SR:
        offload = NoOpOffload()
        dit = object()
        resolution_scale = 2

        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            calls.append(kwargs["save_path"])
            return SimpleNamespace(frames=torch.zeros(1, 3, 1, 4, 4), path=kwargs["save_path"])

    monkeypatch.setattr(cli, "load_sr_pipeline", lambda *_args, **_kwargs: SR())
    monkeypatch.setattr(components, "scale_factor_for", lambda _pipe: (1, 1, 1))
    monkeypatch.setattr(warmup, "warmup", lambda *_args: None)
    monkeypatch.setattr(warmup, "compile_vae_decode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(warmup, "clip_base_resolution", lambda *_args: (2, 2))
    result = SimpleNamespace(frames=torch.zeros(1, 3, 1, 2, 2), audio=None)
    cli._super_resolve_if_enabled(
        "unused",
        "cpu",
        "block",
        result,
        tmp_path / "base.mp4",
        NoOpProfile(),
        fps=24,
        audio_fps=44100,
        warmup=warmup_enabled,
    )
    assert len(calls) == expected_calls
