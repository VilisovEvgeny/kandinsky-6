"""Twelve-cell timing plan: one load per checkpoint, three resolutions each."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kandinsky.bench import CHECKPOINTS, RESOLUTIONS, bind_run_profiles, write_plan
from kandinsky.pipeline.config import load_config
from kandinsky.pipeline.reconfigure import _dit_needs_reload
from kandinsky.runtime.profile import _profile_output_dir

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINTS = _ROOT / "kandinsky" / "configs" / "checkpoints"
_SR = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"


def test_plan_is_twelve_cells_grouped_by_checkpoint(tmp_path: Path) -> None:
    groups = write_plan(tmp_path)
    assert [group.checkpoint for group in groups] == list(CHECKPOINTS)
    assert [resolution.name for resolution in RESOLUTIONS] == ["sd", "hd", "fhd"]
    names = [job.name for group in groups for job in group.jobs]
    assert names == [
        "lite-sd",
        "lite-hd",
        "lite-fhd",
        "lite-distill-sd",
        "lite-distill-hd",
        "lite-distill-fhd",
        "pro-sd",
        "pro-hd",
        "pro-fhd",
        "pro-distill-sd",
        "pro-distill-hd",
        "pro-distill-fhd",
    ]
    assert all(len(group.jobs) == 3 for group in groups)


@pytest.mark.parametrize("checkpoint", ["lite", "pro", "lite-distill", "pro-distill"])
def test_cell_keeps_one_checkpoint_across_resolutions(checkpoint: str, tmp_path: Path) -> None:
    jobs = {job.resolution: job for group in write_plan(tmp_path) for job in group.jobs if job.checkpoint == checkpoint}
    loaded = {name: load_config(job.config_path) for name, job in jobs.items()}
    source = load_config(_CHECKPOINTS / f"{checkpoint}.yaml")
    distilled = checkpoint.endswith("-distill")

    for name, cfg in loaded.items():
        resolution = next(item for item in RESOLUTIONS if item.name == name)
        assert cfg.checkpoint == checkpoint
        assert cfg.generation.height == resolution.height
        assert cfg.generation.width == resolution.width
        assert cfg.generation.sample_frames == 121
        assert cfg.compile.strategy == "torch"
        assert cfg.offload.strategy == "module"
        assert cfg.attention.engine == "flash_attention_3"
        assert cfg.beautifier.name == "qwen25"
        assert cfg.profile.mode == "time_mem_module"
        assert cfg.piflow.enabled is distilled
        if distilled:
            assert cfg.generation.num_steps == 10
            assert cfg.generation.guidance_weight == 1.0
            assert cfg.cache.mode == "none"
            assert cfg.cache.magcache is None
        else:
            assert cfg.generation.num_steps == 50
            assert cfg.generation.guidance_weight == 5.0
            assert cfg.cache.mode == "magcache"
            mag = cfg.cache.magcache
            assert mag is not None and source.cache.magcache is not None
            assert mag.mag_ratios == source.cache.magcache.mag_ratios
            assert mag.thresh == source.cache.magcache.thresh
            assert mag.K == source.cache.magcache.K
            assert mag.retention_ratio == source.cache.magcache.retention_ratio
        if resolution.scale is None:
            assert cfg.sr.enabled is False
        else:
            assert cfg.sr.enabled is True
            assert cfg.sr.checkpoint_path == _SR
            assert cfg.sr.resolution_scale == resolution.scale
            assert cfg.sr.num_steps == 2

    assert _dit_needs_reload(loaded["sd"], loaded["hd"]) is False
    assert _dit_needs_reload(loaded["hd"], loaded["fhd"]) is False
    assert _dit_needs_reload(loaded["sd"], loaded["fhd"]) is False


def test_distill_reloads_the_dit(tmp_path: Path) -> None:
    jobs = {job.name: load_config(job.config_path) for group in write_plan(tmp_path) for job in group.jobs}
    assert _dit_needs_reload(jobs["lite-sd"], jobs["lite-distill-sd"]) is True
    assert _dit_needs_reload(jobs["pro-sd"], jobs["pro-distill-sd"]) is True


def test_profiles_go_to_the_run_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KANDINSKY_PROFILE_DIR", str(tmp_path / "elsewhere"))
    run = tmp_path / "outputs" / "bench_example"
    directory = bind_run_profiles(run)
    assert directory == run / "profiles"
    assert directory.is_dir()
    assert os.environ["KANDINSKY_PROFILE_DIR"] == str(directory)
    assert _profile_output_dir() == directory


def test_justfile_runs_the_bench() -> None:
    text = (_ROOT / "JUSTFILE").read_text(encoding="utf-8")
    assert "uv run kandy bench" in text
    assert "unset KANDINSKY_PROFILE_DIR" not in text
