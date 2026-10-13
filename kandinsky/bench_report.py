"""Module-time charts for one bench run.

One chart per profiled cell, plus a table. A sector is that module's share of
profile ``total_time`` plus ``sr_time``. ``sr`` is its own module. Module colors
stay fixed across charts. Rows are
distilled checkpoints first, then the others, lite before pro, and SD, HD,
Full HD within each checkpoint.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

_MODULES = (
    "dit",
    "vae_decoder",
    "text_encoder",
    "vocoder",
    "audio_vae_decoder",
    "text_encoder_2",
    "sr",
)
_COLORS = {
    "dit": "#2563EB",
    "vae_decoder": "#059669",
    "text_encoder": "#EA580C",
    "vocoder": "#DB2777",
    "audio_vae_decoder": "#7C3AED",
    "text_encoder_2": "#CA8A04",
    "sr": "#0891B2",
    "other": "#CBD5E1",
}
_EXTRA_COLORS = ("#0F766E", "#B45309", "#BE123C", "#4338CA", "#0369A1", "#4D7C0F")
_RESOLUTION_ORDER = {"sd": 0, "hd": 1, "fhd": 2}
_REPORT_NAME = "module-time-report.html"

_CSS = """
  :root { color-scheme: light; --ink: #0f172a; --muted: #64748b; --line: #e2e8f0; --paper: #f8fafc; }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.45 "Segoe UI", system-ui, sans-serif; }
  main { max-width: 1180px; margin: 0 auto; padding: 36px 28px 64px; }
  h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: -0.03em; }
  h2 { margin: 36px 0 12px; font-size: 20px; letter-spacing: -0.02em; }
  .lead { margin: 0; max-width: 78ch; color: #334155; }
  .lead + .lead { margin-top: 8px; }
  code { font-size: 0.92em; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px 16px; margin: 18px 0 8px; }
  .legend-item { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; }
  .legend-item i { width: 12px; height: 12px; border-radius: 99px; display: inline-block; }
  .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 14px 14px 8px; }
  .card h3 { margin: 0; font-size: 16px; }
  .meta { margin: 4px 0 0; min-height: 2.8em; color: var(--muted); font-size: 12px; }
  .card svg { display: block; width: min(100%, 250px); margin: 0 auto; }
  .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 14px; background: white; }
  table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
  thead th { font-weight: 650; background: #f8fafc; }
  tbody th { text-align: left; font-weight: 600; background: white; min-width: 168px; }
  .run { display: block; }
  .sub { display: block; margin-top: 2px; color: var(--muted); font-weight: 450; font-size: 11px; }
  td { font-weight: 650; }
  .scale { display: flex; gap: 4px; margin-top: 10px; max-width: 520px; }
  .scale span { flex: 1; text-align: center; font-size: 11px; font-weight: 650; padding: 6px 0; border-radius: 6px; }
  .note { margin: 10px 0 0; color: var(--muted); font-size: 12px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } main { padding: 20px 14px 40px; } }
  @media print { body { background: white; } .card, .table-wrap { break-inside: avoid; } main { max-width: none; } }
"""


def write_module_time_report(run: Path) -> Path | None:
    """Write ``module-time-report.html`` into ``run``. Returns the path, or None when there are no profiles."""
    rows = _load_rows(run)
    if not rows:
        return None
    target = run / _REPORT_NAME
    target.write_text(_page(run.name, rows), encoding="utf-8")
    return target


def _row_order(name: str) -> tuple[int, int, int, str]:
    """Distilled before full, lite before pro, then SD, HD, Full HD."""
    distilled = 0 if "-distill" in name else 1
    if name.startswith("lite"):
        family = 0
    elif name.startswith("pro"):
        family = 1
    else:
        family = 2
    resolution = name.rsplit("-", 1)[-1]
    return (distilled, family, _RESOLUTION_ORDER.get(resolution, 9), name)


def _load_rows(run: Path) -> list[dict]:
    directory = run / "profiles"
    if not directory.is_dir():
        return []
    latest: dict[str, Path] = {}
    for path in sorted(directory.glob("*.json")):
        job, sep, _rest = path.stem.partition("_")
        latest[job if sep else path.stem] = path
    rows = []
    for name, path in latest.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("mode") != "time_mem_module":
            continue
        row = _row(name, payload)
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda item: _row_order(item["name"]))
    return rows


def _row(name: str, payload: dict) -> dict | None:
    measurements = payload.get("measurements")
    if not isinstance(measurements, dict):
        return None
    modules = measurements.get("modules")
    if not isinstance(modules, dict):
        return None
    total = measurements.get("total_time")
    if not isinstance(total, (int, float)) or total <= 0:
        return None
    times = {
        module: float(entry["time"])
        for module, entry in modules.items()
        if isinstance(entry, dict) and isinstance(entry.get("time"), (int, float))
    }
    profile = float(total)
    sr_raw = measurements.get("sr_time")
    sr = float(sr_raw) if isinstance(sr_raw, (int, float)) and sr_raw > 0 else 0.0
    other = max(0.0, profile - sum(times.values()))
    times["sr"] = sr
    times["other"] = other
    span = profile + sr
    shares = {module: value / span * 100.0 for module, value in times.items()}
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
    return {
        "name": name,
        "total": span,
        "times": times,
        "shares": shares,
        "width": run.get("width"),
        "height": run.get("height"),
        "frames": run.get("latent_frames"),
        "steps": run.get("num_steps"),
        "guidance": run.get("guidance_weight"),
        "cache": run.get("cache_mode"),
        "gpu": environment.get("gpu"),
    }


def _columns(rows: list[dict]) -> list[str]:
    present = {module for row in rows for module in row["times"] if module != "other"}
    extras = sorted(module for module in present if module not in _MODULES)
    return [module for module in _MODULES if module in present] + extras + ["other"]


def _colors(columns: list[str]) -> dict[str, str]:
    colors = dict(_COLORS)
    extras = [name for name in columns if name not in colors]
    for index, name in enumerate(extras):
        colors[name] = _EXTRA_COLORS[index % len(_EXTRA_COLORS)]
    return colors


def _pct(value: float) -> str:
    if value <= 0:
        return "0"
    if value < 0.01:
        return "<0.01"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _annular(cx: float, cy: float, inner: float, outer: float, start: float, end: float) -> str:
    sweep = end - start
    if sweep <= 0:
        return ""
    if sweep >= 359.999:
        end = start + 359.999
        sweep = 359.999
    large = 1 if sweep > 180 else 0

    def point(radius: float, angle: float) -> tuple[float, float]:
        radians = math.radians(angle)
        return cx + radius * math.sin(radians), cy - radius * math.cos(radians)

    x0, y0 = point(outer, start)
    x1, y1 = point(outer, end)
    xi1, yi1 = point(inner, end)
    xi0, yi0 = point(inner, start)
    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {outer:.3f} {outer:.3f} 0 {large} 1 {x1:.3f} {y1:.3f} "
        f"L {xi1:.3f} {yi1:.3f} "
        f"A {inner:.3f} {inner:.3f} 0 {large} 0 {xi0:.3f} {yi0:.3f} Z"
    )


def _chart(row: dict, columns: list[str], colors: dict[str, str]) -> str:
    size = 232
    cx = cy = size / 2
    outer, inner = 96, 60
    angle = 0.0
    parts: list[str] = []
    for key in columns:
        share = row["shares"].get(key, 0.0)
        sweep = share / 100.0 * 360.0
        if sweep > 0.05:
            path = _annular(cx, cy, inner, outer, angle, angle + sweep)
            tip = f"{key}: {row['times'].get(key, 0.0):.2f} s, {share:.2f}%"
            parts.append(f'<path d="{path}" fill="{colors[key]}"><title>{html.escape(tip)}</title></path>')
            if share >= 8:
                mid = math.radians(angle + sweep / 2)
                lx = cx + ((inner + outer) / 2) * math.sin(mid)
                ly = cy - ((inner + outer) / 2) * math.cos(mid)
                fill = "#1e293b" if key == "other" else "#ffffff"
                parts.append(
                    f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle" dominant-baseline="central" '
                    f'fill="{fill}" font-size="13" font-weight="650">{_pct(share)}%</text>'
                )
        angle += sweep
    parts.append(
        f'<text x="{cx}" y="{cy - 6}" text-anchor="middle" fill="#0f172a" '
        f'font-size="18" font-weight="680">{row["total"]:.1f} s</text>'
        f'<text x="{cx}" y="{cy + 14}" text-anchor="middle" fill="#64748b" font-size="11">total</text>'
    )
    label = html.escape(row["name"])
    return f'<svg viewBox="0 0 {size} {size}" role="img" aria-label="{label}">{"".join(parts)}</svg>'


def _meta(row: dict) -> str:
    bits = []
    if row["width"] is not None and row["height"] is not None:
        bits.append(f"{row['width']}×{row['height']}")
    if row["frames"] is not None:
        bits.append(f"{row['frames']} latent")
    if row["steps"] is not None:
        bits.append(f"{row['steps']} steps")
    if isinstance(row["guidance"], (int, float)):
        bits.append(f"cfg {row['guidance']:g}")
    if row["cache"]:
        bits.append(str(row["cache"]))
    return " · ".join(bits)


def _heat(share: float) -> tuple[str, float]:
    t = max(0.0, min(1.0, share / 100.0))
    stops = (
        (0.00, (247, 248, 250)),
        (0.08, (219, 234, 254)),
        (0.20, (147, 197, 253)),
        (0.40, (59, 130, 246)),
        (0.70, (29, 78, 216)),
        (1.00, (30, 58, 138)),
    )
    rgb = stops[-1][1]
    for (left, start), (right, end) in zip(stops, stops[1:]):
        if t <= right:
            blend = 0.0 if right == left else (t - left) / (right - left)
            rgb = tuple(round(start[channel] + (end[channel] - start[channel]) * blend) for channel in range(3))
            break
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})", t


def _legend(columns: list[str], colors: dict[str, str]) -> str:
    return "".join(
        f'<span class="legend-item"><i style="background:{colors[name]}"></i>{html.escape(name)}</span>'
        for name in columns
    )


def _cards(rows: list[dict], columns: list[str], colors: dict[str, str]) -> str:
    cards = []
    for row in rows:
        cards.append(
            "<article class=\"card\">"
            f"<h3>{html.escape(row['name'])}</h3>"
            f"<p class=\"meta\">{html.escape(_meta(row))}</p>"
            f"{_chart(row, columns, colors)}"
            "</article>"
        )
    return "".join(cards)


def _table(rows: list[dict], columns: list[str]) -> str:
    header = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
    body = []
    for row in rows:
        cells = []
        for name in columns:
            share = row["shares"].get(name, 0.0)
            color, weight = _heat(share)
            ink = "#ffffff" if weight >= 0.34 else "#0f172a"
            seconds = row["times"].get(name, 0.0)
            cells.append(
                f'<td style="background:{color};color:{ink}" title="{seconds:.2f} s">{_pct(share)}</td>'
            )
        detail = f"{row['total']:.1f} s"
        body.append(
            "<tr>"
            f'<th scope="row"><span class="run">{html.escape(row["name"])}</span>'
            f'<span class="sub">{html.escape(detail)}</span></th>'
            + "".join(cells)
            + "</tr>"
        )
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr><th>run</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        "</table></div>"
    )


def _scale() -> str:
    parts = []
    for value in (0, 10, 25, 50, 75, 100):
        color, weight = _heat(float(value))
        ink = "#ffffff" if weight >= 0.34 else "#0f172a"
        parts.append(f'<span style="background:{color};color:{ink}">{value}%</span>')
    return f'<div class="scale">{"".join(parts)}</div>'


def _page(run_name: str, rows: list[dict]) -> str:
    columns = _columns(rows)
    colors = _colors(columns)
    gpu = next((row["gpu"] for row in rows if row["gpu"]), None)
    intro = f"Run <code>{html.escape(run_name)}</code>"
    if gpu:
        intro += f" on {html.escape(str(gpu))}"
    intro += "."
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Module time share — {html.escape(run_name)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
  <h1>Module time share</h1>
  <p class="lead">{intro}</p>
  <p class="lead">Each sector is that module's share of profile <code>total_time</code> plus <code>sr_time</code>. The same module keeps the same color on every chart. <code>sr</code> is the super-resolution pass. <code>other</code> is profile time outside the measured modules. Sectors under 0.5% are nearly invisible on the ring.</p>
  <div class="legend">{_legend(columns, colors)}</div>
  <h2>Charts</h2>
  <div class="grid">{_cards(rows, columns, colors)}</div>
  <h2>Share of total time, %</h2>
  {_table(rows, columns)}
  {_scale()}
  <p class="note">Cell color is the size of the percentage. Hover a cell or a sector to see seconds. The line under a run name is profile time plus SR.</p>
</main>
</body>
</html>
"""
