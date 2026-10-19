"""Module-time report written at the end of a bench run."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kandinsky.tools.bench import RunResult, run
from kandinsky.tools.bench_report import write_module_time_report

_DIT = "#2563EB"


def _profile(name: str, *, dit: float, vae: float, gpu: str = "NVIDIA H100 80GB HBM3") -> dict:
    total = dit + vae + 1.0
    return {
        "mode": "time_mem_module",
        "run": {
            "width": 864,
            "height": 480,
            "latent_frames": 31,
            "num_steps": 10 if "distill" in name else 50,
            "guidance_weight": 1.0 if "distill" in name else 5.0,
            "cache_mode": "none" if "distill" in name else "magcache",
        },
        "measurements": {
            "total_time": total,
            "modules": {
                "dit": {"time": dit},
                "vae_decoder": {"time": vae},
                "text_encoder": {"time": 0.5},
                "vocoder": {"time": 0.2},
                "audio_vae_decoder": {"time": 0.2},
                "text_encoder_2": {"time": 0.1},
            },
        },
        "environment": {"gpu": gpu},
    }


def _write_profile(directory: Path, name: str, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}_2026-10-13T00-00-00-000000.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_report_orders_distill_then_lite_then_pro(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    for name, dit in (
        ("pro-sd", 90.0),
        ("lite-distill-fhd", 20.0),
        ("pro-distill-hd", 30.0),
        ("lite-sd", 80.0),
    ):
        _write_profile(profiles, name, _profile(name, dit=dit, vae=5.0))

    report = write_module_time_report(tmp_path)
    assert report == tmp_path / "module-time-report.html"
    text = report.read_text(encoding="utf-8")
    assert not re.search(r"[А-Яа-яЁё]", text)
    leads = re.findall(r'<p class="lead">(.*?)</p>', text)
    assert leads[0] == f"Run <code>{tmp_path.name}</code> on NVIDIA H100 80GB HBM3."
    assert "total_time" not in leads[0]
    assert "<h2>Charts</h2>" in text
    assert re.findall(r"<h3>(.*?)</h3>", text) == [
        "lite-distill-fhd",
        "pro-distill-hd",
        "lite-sd",
        "pro-sd",
    ]
    names = [
        "lite-distill-fhd",
        "pro-distill-hd",
        "lite-sd",
        "pro-sd",
    ]
    assert re.findall(r'<span class="run">(.*?)</span>', text) == names + names
    assert text.count(f'fill="{_DIT}"') == 4


def test_sr_is_a_module_in_the_same_total(tmp_path: Path) -> None:
    payload = _profile("lite-hd", dit=50.0, vae=10.0)
    payload["measurements"]["sr_time"] = 40.0
    _write_profile(tmp_path / "profiles", "lite-hd", payload)

    text = write_module_time_report(tmp_path).read_text(encoding="utf-8")
    assert 'fill="#0891B2"' in text
    assert "prefers-color-scheme: dark" in text
    assert "body.vscode-dark" in text
    assert "--heat-dark:" in text
    assert ">39.6<" in text
    assert "<th>sr</th>" in text
    assert "SR is outside" not in text


def test_report_uses_the_recorded_profile_not_the_latest(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    _write_profile(profiles, "lite-sd", _profile("lite-sd", dit=10.0, vae=5.0))
    early = profiles / "lite-sd_2026-10-13T00-00-00-000000.json"
    late = profiles / "lite-sd_2026-10-13T00-00-01-000000.json"
    late.write_text(json.dumps(_profile("lite-sd", dit=80.0, vae=5.0)), encoding="utf-8")
    failed = profiles / "lite-hd_2026-10-13T00-00-02-000000.json"
    failed.write_text(json.dumps(_profile("lite-hd", dit=80.0, vae=5.0)), encoding="utf-8")
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {"name": "lite-sd", "success": True, "profile": str(early)},
                {"name": "lite-hd", "success": False, "profile": str(failed)},
            ]
        ),
        encoding="utf-8",
    )

    text = write_module_time_report(tmp_path).read_text(encoding="utf-8")
    assert "lite-sd" in text
    assert "lite-hd" not in text
    assert "62.5" in text


def test_module_shares_are_medians_across_the_recorded_repetitions(tmp_path: Path) -> None:
    profiles = tmp_path / "profiles"
    recorded = []
    for index, dit in enumerate((10.0, 40.0, 20.0, 100.0)):
        path = profiles / f"lite-sd_2026-10-13T00-00-0{index}-000000.json"
        profiles.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_profile("lite-sd", dit=dit, vae=2.0)), encoding="utf-8")
        recorded.append(path)
    ignored = profiles / "lite-sd_2026-10-13T00-00-09-000000.json"
    ignored.write_text(json.dumps(_profile("lite-sd", dit=1.0, vae=2.0)), encoding="utf-8")
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "name": "lite-sd",
                    "success": True,
                    "seconds": 50.0,
                    "profile": str(recorded[2]),
                    "profiles": [str(path) for path in recorded],
                }
            ]
        ),
        encoding="utf-8",
    )

    text = write_module_time_report(tmp_path).read_text(encoding="utf-8")
    assert "90.9" in text
    assert ">50.0 s<" in text
    assert ">median<" in text


def test_memory_table_uses_the_maximum_call_and_sr(tmp_path: Path) -> None:
    gib = 1024**3
    profiles = tmp_path / "profiles"
    paths = []
    peaks = (
        (40 * gib, 50 * gib, 10 * gib, 12 * gib),
        (10 * gib, 70 * gib, None, None),
    )
    for index, (call_allocated, call_reserved, sr_allocated, sr_reserved) in enumerate(peaks):
        payload = _profile("lite-fhd", dit=10.0 + index, vae=2.0)
        payload["run"]["offload_strategy"] = "module"
        payload["environment"]["gpu_total_mem"] = 80 * gib
        payload["measurements"]["total_peak_allocated_mem"] = call_allocated
        payload["measurements"]["total_peak_reserved_mem"] = call_reserved
        if sr_allocated is not None:
            payload["measurements"]["sr_peak_allocated_mem"] = sr_allocated
            payload["measurements"]["sr_peak_reserved_mem"] = sr_reserved
        path = profiles / f"lite-fhd_2026-10-13T00-00-0{index}-000000.json"
        profiles.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    (tmp_path / "results.json").write_text(
        json.dumps([{"name": "lite-fhd", "success": True, "profiles": [str(path) for path in paths]}]),
        encoding="utf-8",
    )

    text = write_module_time_report(tmp_path).read_text(encoding="utf-8")
    section = text.split("<h2>Peak memory, GiB</h2>", 1)[1]
    assert "<th>dit</th>" not in section
    assert "<th>call</th>" not in section
    assert '<th colspan="2" class="group">generate</th>' in section
    assert '<th colspan="2" class="group split">sr</th>' in section
    assert section.count("<th>allocated</th>") == 1
    assert section.count('<th class="split">allocated</th>') == 1
    assert ">40.0<" in section
    assert ">70.0<" in section
    assert ">10.0<" in section
    assert ">12.0<" in section
    assert ">module<" in section
    assert "this GPU (80 GiB)" in section
    assert not re.search(r"[А-Яа-яЁё]", section)


def test_report_is_skipped_without_profiles(tmp_path: Path) -> None:
    assert write_module_time_report(tmp_path) is None


def test_bench_writes_the_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    monkeypatch.setattr("kandinsky.tools.bench._one_gpu", lambda requested, threshold: 0)

    def execute(groups, prompt, seed, run_dir, gpu_id):
        _write_profile(
            run_dir / "profiles",
            "lite-distill-sd",
            _profile("lite-distill-sd", dit=12.0, vae=4.0),
        )
        return [
            RunResult(
                "lite-distill-sd",
                True,
                16.0,
                run_dir / "logs" / "lite-distill-sd.log",
                run_dir / "profiles" / "lite-distill-sd_2026-10-13T00-00-00-000000.json",
            )
        ]

    monkeypatch.setattr("kandinsky.tools.bench._execute", execute)
    assert run(gpu=0) == 0
    reports = list((tmp_path / "outputs").glob("bench_*/module-time-report.html"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "<h2>Charts</h2>" in text
    assert "lite-distill-sd" in text
