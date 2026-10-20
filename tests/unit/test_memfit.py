"""Consumer presets run on an emulated 16, 24, or 32 GB budget."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from kandinsky.pipeline.config import load_config
from kandinsky.tools.memfit import (
    _GIB,
    JobResult,
    _worker_env,
    assign_gpus,
    consumer_presets,
    fill_down_to,
    next_reservation,
    preset_memory_gb,
    runnable_attention,
    write_job_config,
)


def test_sage_preset_uses_flash_attention_on_hopper() -> None:
    assert runnable_attention("sage", sage_installed=False, flash3_installed=True) == "flash_attention_3"
    assert runnable_attention("sage", sage_installed=True, flash3_installed=True) == "sage"
    assert runnable_attention("sage", sage_installed=False, flash3_installed=False) == "sdpa"
    assert runnable_attention("sdpa", sage_installed=False, flash3_installed=True) == "sdpa"


def test_worker_hides_recovered_allocator_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORCH_CPP_LOG_LEVEL", raising=False)
    env = _worker_env(3)
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["TORCH_CPP_LOG_LEVEL"] == "ERROR"
    monkeypatch.setenv("TORCH_CPP_LOG_LEVEL", "WARNING")
    assert _worker_env(0)["TORCH_CPP_LOG_LEVEL"] == "WARNING"


def test_worker_requires_the_gpu_to_be_selected_before_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from kandinsky.tools.memfit import _require_visible_gpu

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        _require_visible_gpu(1)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _require_visible_gpu(1)


_ROOT = Path(__file__).resolve().parents[2]
_DEVICES = _ROOT / "kandinsky" / "configs" / "devices"


class _Card:
    """Driver memory. ``overhead`` is extra bytes each allocation consumes."""

    def __init__(self, total: int, overhead: int = 0):
        self.total = total
        self.overhead = overhead
        self.held: list[int] = []

    def free_bytes(self) -> int:
        return self.total - sum(self.held)

    def allocate(self, nbytes: int) -> None:
        consumed = nbytes + self.overhead
        if consumed > self.free_bytes():
            raise MemoryError(consumed)
        self.held.append(consumed)

    def release_last(self) -> None:
        self.held.pop()


def test_consumer_presets_are_the_16_24_and_32_gb_cards() -> None:
    presets = consumer_presets(_DEVICES)
    assert [(item.name, item.memory_bytes // _GIB) for item in presets] == [
        ("rtx-5060-ti", 16),
        ("rtx-5080", 16),
        ("rtx-4090", 24),
        ("rtx-5090", 32),
    ]
    assert preset_memory_gb(_DEVICES / "h100.yaml") == 80
    assert preset_memory_gb(_DEVICES / "a100-80gb.yaml") == 80
    assert preset_memory_gb(_DEVICES / "rtx-pro-6000.yaml") == 96


def test_reservation_rounds_down_to_the_alignment() -> None:
    assert next_reservation(1000, 100, align=512) == 512
    assert next_reservation(1024, 0, align=512) == 1024
    assert next_reservation(600, 100, align=512) is None
    assert next_reservation(100, 100, align=512) is None
    assert next_reservation(50, 100, align=512) is None


def test_fill_stops_when_free_memory_is_already_in_band() -> None:
    card = _Card(_GIB + 1024)
    left = fill_down_to(card.free_bytes, card.allocate, card.release_last, _GIB, tolerance=64 * 1024**2)
    assert left == _GIB + 1024
    assert card.held == []


def test_fill_allocates_the_gap_when_malloc_is_exact() -> None:
    total = 80 * _GIB
    target = 32 * _GIB
    card = _Card(total)
    left = fill_down_to(card.free_bytes, card.allocate, card.release_last, target)
    assert left == target
    assert card.held == [48 * _GIB]


def test_fill_shrinks_when_malloc_consumes_more_than_requested() -> None:
    total = 80 * _GIB
    target = 16 * _GIB
    overhead = 128 * 1024**2
    card = _Card(total, overhead=overhead)
    left = fill_down_to(
        card.free_bytes,
        card.allocate,
        card.release_last,
        target,
        tolerance=64 * 1024**2,
    )
    assert abs(left - target) <= 64 * 1024**2
    assert card.free_bytes() == left


def test_fill_refuses_a_card_smaller_than_the_target() -> None:
    card = _Card(10 * _GIB)
    with pytest.raises(RuntimeError, match="cannot leave"):
        fill_down_to(card.free_bytes, card.allocate, card.release_last, 16 * _GIB)


def test_jobs_are_spread_across_gpus() -> None:
    plan = assign_gpus(["rtx-5060-ti", "rtx-5080", "rtx-4090", "rtx-5090"], [0, 3])
    assert plan == {
        0: ["rtx-5060-ti", "rtx-4090"],
        3: ["rtx-5080", "rtx-5090"],
    }


def test_job_config_keeps_the_preset_and_records_peaks(tmp_path: Path) -> None:
    source = _DEVICES / "rtx-5060-ti.yaml"
    dest = tmp_path / "rtx-5060-ti.yaml"
    write_job_config(source, dest)
    loaded = load_config(dest)
    assert loaded.checkpoint == "pro-distill"
    assert loaded.offload.strategy == "block"
    assert loaded.text_embedder.quantized_qwen is True
    assert loaded.text_embedder.release_after_encode is True
    assert loaded.attention.engine == "sage"
    assert loaded.profile.mode == "time_mem_module"
    assert loaded.sr.decode_spatial_splits == 2

    wide = tmp_path / "rtx-5090.yaml"
    write_job_config(_DEVICES / "rtx-5090.yaml", wide)
    pro = load_config(wide)
    assert pro.text_embedder.quantized_qwen is False
    assert pro.text_embedder.release_after_encode is False
    assert pro.sr.decode_spatial_splits == 1


def test_job_result_survives_spawn_pickle() -> None:
    item = JobResult(
        name="rtx-4090",
        gpu=1,
        target_bytes=24 * _GIB,
        free_bytes=24 * _GIB,
        held_bytes=48 * _GIB,
        success=True,
        seconds=1.5,
        profile=None,
        log="log",
    )
    assert pickle.loads(pickle.dumps(item)) == item


def test_memfit_accepts_several_gpus() -> None:
    from kandinsky.cli import app

    _command, bound, _unused = app.parse_args(["memfit", "--gpus", "2", "4", "--seed", "7"])
    assert bound.arguments["gpus"] == [2, 4]
    assert bound.arguments["seed"] == 7


def test_justfile_runs_memfit() -> None:
    text = (_ROOT / "JUSTFILE").read_text(encoding="utf-8")
    assert "uv run kandy memfit" in text
