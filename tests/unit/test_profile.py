"""SR pass peaks land on the same profile JSON as component high-water marks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from kandinsky.runtime.profile import NoOpProfile, TimeMemModuleProfile


def _profile(tmp_path, monkeypatch) -> TimeMemModuleProfile:
    monkeypatch.setenv("KANDINSKY_PROFILE_DIR", str(tmp_path))
    profile = TimeMemModuleProfile()
    pipe = SimpleNamespace(device=torch.device("cpu"), config_path=None, launch_overrides={})
    profile.apply(pipe)
    with profile.pipeline():
        pass
    return profile


def test_measure_sr_records_cuda_peaks(tmp_path, monkeypatch) -> None:
    profile = _profile(tmp_path, monkeypatch)
    resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: resets.append(device))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 111)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 222)

    with profile.measure_sr("cuda:0") as timing:
        timing.elapsed = 1.5
        timing.save = True

    payload = json.loads(profile.path.read_text(encoding="utf-8"))
    measurements = payload["measurements"]
    assert len(resets) == 1
    assert measurements["sr_time"] == 1.5
    assert measurements["sr_peak_allocated_mem"] == 111
    assert measurements["sr_peak_reserved_mem"] == 222
    assert profile.report.sr_peak_allocated_mem == 111
    assert profile.report.sr_peak_reserved_mem == 222


def test_measure_sr_on_cpu_records_zero_peaks(tmp_path, monkeypatch) -> None:
    profile = _profile(tmp_path, monkeypatch)

    with profile.measure_sr("cpu") as timing:
        timing.elapsed = 2.0
        timing.save = True

    measurements = json.loads(profile.path.read_text(encoding="utf-8"))["measurements"]
    assert measurements["sr_time"] == 2.0
    assert measurements["sr_peak_allocated_mem"] == 0
    assert measurements["sr_peak_reserved_mem"] == 0


def test_unsaved_sr_stage_does_not_record(tmp_path, monkeypatch) -> None:
    profile = _profile(tmp_path, monkeypatch)

    with profile.measure_sr("cpu") as timing:
        timing.elapsed = 9.0

    measurements = json.loads(profile.path.read_text(encoding="utf-8"))["measurements"]
    assert "sr_time" not in measurements
    assert profile.report.sr_time is None


def test_failed_sr_pass_does_not_record(tmp_path, monkeypatch) -> None:
    profile = _profile(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="sr failed"):
        with profile.measure_sr("cpu"):
            raise RuntimeError("sr failed")

    measurements = json.loads(profile.path.read_text(encoding="utf-8"))["measurements"]
    assert "sr_time" not in measurements
    assert profile.report.sr_time is None


def test_noop_measure_sr_runs_the_pass() -> None:
    profile = NoOpProfile()
    seen = []
    with profile.measure_sr("cpu") as timing:
        timing.elapsed = 1.0
        timing.save = True
        seen.append("ran")
    assert seen == ["ran"]
    assert profile.report.sr_time is None
