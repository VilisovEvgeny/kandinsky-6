"""Eight-cell timing plan: one load per checkpoint, SD and Full HD."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kandinsky.tools.bench import (
    CHECKPOINTS,
    RESOLUTIONS,
    _Sample,
    _is_oom,
    _keep_completed,
    _median_sample,
    _next_offload,
    bind_run_profiles,
    initial_offload,
    write_plan,
)
from kandinsky.pipeline.config import load_config
from kandinsky.pipeline.reconfigure import _dit_needs_reload
from kandinsky.runtime.profile import _profile_output_dir

_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINTS = _ROOT / "kandinsky" / "configs" / "checkpoints"
_SR = "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers"


def test_plan_is_eight_cells_grouped_by_checkpoint(tmp_path: Path) -> None:
    groups = write_plan(tmp_path)
    assert [group.checkpoint for group in groups] == list(CHECKPOINTS)
    assert [resolution.name for resolution in RESOLUTIONS] == ["sd", "fhd"]
    names = [job.name for group in groups for job in group.jobs]
    assert names == [
        "lite-sd",
        "lite-fhd",
        "lite-distill-sd",
        "lite-distill-fhd",
        "pro-sd",
        "pro-fhd",
        "pro-distill-sd",
        "pro-distill-fhd",
    ]
    assert all(len(group.jobs) == 2 for group in groups)


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
        assert cfg.offload.strategy == "none"
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


def test_initial_offload_follows_module_bytes() -> None:
    assert initial_offload({"dit": 40, "vae": 10}, 100) == "none"
    assert initial_offload({"dit": 60, "vae": 50}, 100) == "module"
    assert initial_offload({"dit": 200, "vae": 10}, 100) == "block"
    assert initial_offload({}, 100) == "none"
    assert initial_offload({"dit": 10}, 0) == "none"


def test_oom_keeps_the_largest_completed_set(tmp_path: Path) -> None:
    module = [_Sample(1.0, tmp_path / "a.json"), _Sample(2.0, tmp_path / "b.json")]
    block = [_Sample(3.0, tmp_path / "c.json")]
    kept = _keep_completed(None, module, "module")
    assert kept is not None
    assert [sample.profile.name for sample in kept[0]] == ["a.json", "b.json"]
    assert kept[1] == "module"
    kept = _keep_completed(kept, [], "module")
    assert len(kept[0]) == 2
    kept = _keep_completed(kept, block, "block")
    assert len(kept[0]) == 2
    assert kept[1] == "module"
    replaced = _keep_completed(kept, module + block, "block")
    assert replaced is not None
    assert len(replaced[0]) == 3
    assert replaced[1] == "block"


def test_oom_steps_to_the_next_offload() -> None:
    assert _is_oom(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not _is_oom(RuntimeError("missing output"))
    assert _next_offload("none") == "module"
    assert _next_offload("module") == "block"
    assert _next_offload("block") is None


def test_median_sample_is_the_middle_run(tmp_path: Path) -> None:
    samples = [_Sample(float(value), tmp_path / f"{value}.json") for value in (9, 1, 4, 3, 8)]
    chosen = _median_sample(samples)
    assert chosen.elapsed == 4
    assert chosen.profile.name == "4.json"


def test_justfile_runs_the_bench() -> None:
    text = (_ROOT / "JUSTFILE").read_text(encoding="utf-8")
    assert "uv run kandy bench" in text
    assert "unset KANDINSKY_PROFILE_DIR" not in text
