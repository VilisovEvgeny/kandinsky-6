"""Module-time report written at the end of a bench run."""

from __future__ import annotations

import json
import re
from pathlib import Path

from kandinsky.bench import RunResult, run
from kandinsky.bench_report import write_module_time_report

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
    assert re.findall(r'<span class="run">(.*?)</span>', text) == [
        "lite-distill-fhd",
        "pro-distill-hd",
        "lite-sd",
        "pro-sd",
    ]
    assert text.count(f'fill="{_DIT}"') == 4


def test_sr_is_a_module_in_the_same_total(tmp_path: Path) -> None:
    payload = _profile("lite-hd", dit=50.0, vae=10.0)
    payload["measurements"]["sr_time"] = 40.0
    _write_profile(tmp_path / "profiles", "lite-hd", payload)

    text = write_module_time_report(tmp_path).read_text(encoding="utf-8")
    assert 'fill="#0891B2"' in text
    assert ">39.6<" in text
    assert "<th>sr</th>" in text
    assert "SR is outside" not in text


def test_report_is_skipped_without_profiles(tmp_path: Path) -> None:
    assert write_module_time_report(tmp_path) is None


def test_bench_writes_the_report(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KANDINSKY_HOME", str(tmp_path))
    monkeypatch.setattr("kandinsky.bench._one_gpu", lambda requested, threshold: 0)

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

    monkeypatch.setattr("kandinsky.bench._execute", execute)
    assert run(gpu=0) == 0
    reports = list((tmp_path / "outputs").glob("bench_*/module-time-report.html"))
    assert len(reports) == 1
    text = reports[0].read_text(encoding="utf-8")
    assert "<h2>Charts</h2>" in text
    assert "lite-distill-sd" in text
