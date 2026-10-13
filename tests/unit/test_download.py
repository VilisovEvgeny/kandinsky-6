"""``kandy download`` accepts one or more catalog names."""

from __future__ import annotations

from pathlib import Path

import pytest

from kandinsky.cli import download


def test_download_fetches_each_name_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []

    def fake_ensure(name: str, *, cache_dir: str | None = None) -> Path:
        assert cache_dir == str(tmp_path)
        seen.append(name)
        return tmp_path / name

    monkeypatch.setattr("kandinsky.runtime.weights.ensure_checkpoint", fake_ensure)
    download("lite-distill", "lite", "pro", cache_dir=str(tmp_path))
    assert seen == ["lite-distill", "lite", "pro"]


def test_download_rejects_an_unknown_name_before_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ensure(name: str, *, cache_dir: str | None = None) -> Path:
        raise AssertionError(f"should not fetch {name}")

    monkeypatch.setattr("kandinsky.runtime.weights.ensure_checkpoint", fake_ensure)
    with pytest.raises(KeyError, match="unknown checkpoint 'nope'"):
        download("lite", "nope")


def test_download_requires_a_name() -> None:
    with pytest.raises(ValueError, match="at least one checkpoint"):
        download()
