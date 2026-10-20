"""One generation per consumer preset, on emulated VRAM.

The host cards are whatever is installed, usually H100. Each consumer preset
in ``kandinsky/configs/devices`` names a 16, 24, or 32 GB card. Before loading
that preset, the process ``cudaMalloc``s enough bytes that ``mem_get_info``
reports that much memory still free. The block stays allocated until the clip
is written. It is not a PyTorch tensor: the caching allocator only sees the
emulated card, the way it would on that card.

Presets run in parallel, one process per GPU. Each process is started with
``CUDA_VISIBLE_DEVICES`` already set, because importing this package
initializes CUDA. A GPU with several presets runs them one after another and
allocates a new block for each budget. Offload, quantization, and the
checkpoint stay as the preset wrote them. The attention engine stays too when
it is installed; SageAttention is not part of the H100 setup, so those presets
run on FlashAttention 3 instead. Profile mode is ``time_mem_module``. Peaks
are the pipeline's own allocations.
"""

from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import yaml

from kandinsky.pipeline.config import load_config
from kandinsky.runtime.home import kandinsky_home

_GIB = 1024**3
_ALIGN = 512
_TOLERANCE = 64 * 1024**2
_CONSUMER_GB = frozenset({16, 24, 32})
_FREE_GPU_THRESHOLD_MIB = 1000
_GB_RE = re.compile(r"(\d+)\s*GB\b", re.IGNORECASE)
_DEVICES = Path(__file__).resolve().parents[1] / "configs" / "devices"
_CUDART = None


@dataclass(frozen=True, slots=True)
class Preset:
    name: str
    path: Path
    memory_bytes: int


@dataclass(frozen=True, slots=True)
class JobResult:
    name: str
    gpu: int
    target_bytes: int
    free_bytes: int | None
    held_bytes: int
    success: bool
    seconds: float | None
    profile: str | None
    log: str
    error: str = ""


def preset_memory_gb(path: Path) -> int | None:
    """Gigabytes named in the preset's first comment, or None when it has none."""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return None
        match = _GB_RE.search(stripped)
        if match is None:
            return None
        return int(match.group(1))
    return None


def consumer_presets(directory: Path | None = None) -> tuple[Preset, ...]:
    """Device presets whose comment names 16, 24, or 32 GB, smallest card first."""
    root = _DEVICES if directory is None else directory
    found: list[Preset] = []
    for path in sorted(root.glob("*.yaml")):
        gigabytes = preset_memory_gb(path)
        if gigabytes not in _CONSUMER_GB:
            continue
        found.append(Preset(path.stem, path, gigabytes * _GIB))
    found.sort(key=lambda item: (item.memory_bytes, item.name))
    return tuple(found)


def next_reservation(free_bytes: int, target_bytes: int, *, align: int = _ALIGN) -> int | None:
    """Bytes to allocate so ``free_bytes`` falls to ``target_bytes``.

    Rounds down to ``align``. A gap smaller than ``align`` returns None.
    """
    gap = free_bytes - target_bytes
    if gap <= 0:
        return None
    size = gap - (gap % align)
    if size <= 0:
        return None
    return size


def _aligned_down(nbytes: int, align: int) -> int:
    if nbytes <= 0:
        return 0
    return nbytes - (nbytes % align)


def _in_band(free_bytes: int, target_bytes: int, tolerance: int) -> bool:
    return target_bytes - tolerance <= free_bytes <= target_bytes + tolerance


def fill_down_to(
    free_bytes: Callable[[], int],
    allocate: Callable[[int], None],
    release_last: Callable[[], None],
    target_bytes: int,
    *,
    align: int = _ALIGN,
    tolerance: int = _TOLERANCE,
) -> int:
    """Allocate until free memory is within ``tolerance`` of ``target_bytes``.

    ``allocate(n)`` reserves ``n`` bytes and may consume more, or raise
    ``MemoryError`` when the request does not fit. An overshoot larger than
    ``tolerance`` releases the last block and replaces it with a smaller one.
    Returns the free byte count.
    """
    if target_bytes < 0:
        raise ValueError("target_bytes must be >= 0")
    if align <= 0:
        raise ValueError("align must be positive")

    for _ in range(8):
        free = free_bytes()
        if _in_band(free, target_bytes, tolerance):
            return free
        if free < target_bytes - tolerance:
            raise RuntimeError(
                f"only {free} bytes free; cannot leave {target_bytes} bytes free"
            )
        size = next_reservation(free, target_bytes, align=align)
        if size is None:
            raise RuntimeError(
                f"{free} bytes free cannot be moved to {target_bytes} at {align}-byte alignment"
            )
        requested = _allocate_fitting(allocate, size, target_bytes, align=align, tolerance=tolerance)
        after = free_bytes()
        if _in_band(after, target_bytes, tolerance):
            return after
        if after > target_bytes + tolerance:
            continue
        shortfall = target_bytes - after
        release_last()
        smaller = _aligned_down(requested - shortfall, align)
        if smaller <= 0 or smaller >= requested:
            raise RuntimeError(
                f"reservation of {requested} bytes left {after} free, "
                f"{shortfall} under {target_bytes}, and cannot shrink"
            )
        allocate(smaller)
    free = free_bytes()
    raise RuntimeError(f"left {free} bytes free, target {target_bytes}")


def _allocate_fitting(
    allocate: Callable[[int], None],
    size: int,
    target_bytes: int,
    *,
    align: int,
    tolerance: int,
) -> int:
    """Allocate ``size`` bytes. On ``MemoryError``, retry a smaller block once."""
    try:
        allocate(size)
    except MemoryError:
        smaller = _aligned_down(size - max(tolerance, align), align)
        if smaller <= 0 or smaller >= size:
            raise RuntimeError(
                f"could not reserve {size} bytes while leaving {target_bytes} free"
            ) from None
        allocate(smaller)
        return smaller
    return size


class DriverBlock:
    """Bytes held with ``cudaMalloc``, outside the PyTorch caching allocator.

    A tensor from ``torch.empty`` shares that allocator with the model. Its
    segment splits the emulated card, so a later request can fail while enough
    bytes are still reserved in holes. ``cudaFree`` is idempotent.
    """

    def __init__(self, nbytes: int, device) -> None:
        import ctypes

        import torch

        self.nbytes = nbytes
        self._ptr = ctypes.c_void_p()
        torch.cuda.set_device(device)
        err = _cudart().cudaMalloc(ctypes.byref(self._ptr), ctypes.c_size_t(nbytes))
        if err != 0:
            raise MemoryError(f"cudaMalloc({nbytes}) failed with error {err}")

    def free(self) -> None:
        import ctypes

        ptr = getattr(self, "_ptr", None)
        if ptr is None or not ptr.value:
            return
        err = _cudart().cudaFree(ptr)
        self._ptr = ctypes.c_void_p()
        if err != 0:
            raise RuntimeError(f"cudaFree failed with error {err}")


def _cudart():
    """The CUDA runtime already loaded by PyTorch, not a second copy."""
    import ctypes
    from pathlib import Path

    global _CUDART
    if _CUDART is not None:
        return _CUDART
    import torch

    del torch
    mapped = [
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
        if "libcudart.so" in line
    ]
    if not mapped:
        raise RuntimeError("libcudart is not loaded")
    lib = ctypes.CDLL(mapped[0])
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    _CUDART = lib
    return lib


def _free_blocks(blocks: list[DriverBlock]) -> None:
    for block in blocks:
        block.free()
    blocks.clear()


def hold_free_memory(target_bytes: int, device) -> tuple[list[DriverBlock], int]:
    """Reserve driver memory so ``mem_get_info`` free bytes equal ``target_bytes``.

    The returned blocks must stay alive for the whole generation. ``empty_cache``
    does not release them, and they do not count as PyTorch reserved memory.
    """
    import torch

    cuda = torch.device(device)
    torch.cuda.set_device(cuda)
    held: list[DriverBlock] = []

    def free_bytes() -> int:
        torch.cuda.synchronize(cuda)
        return int(torch.cuda.mem_get_info(cuda)[0])

    def allocate(nbytes: int) -> None:
        held.append(DriverBlock(nbytes, cuda))

    def release_last() -> None:
        if not held:
            raise RuntimeError("no reservation to shrink")
        held.pop().free()

    gc.collect()
    torch.cuda.synchronize(cuda)
    torch.cuda.empty_cache()
    try:
        left = fill_down_to(free_bytes, allocate, release_last, target_bytes)
    except BaseException:
        _free_blocks(held)
        raise
    return held, left


def write_job_config(preset: Path, dest: Path) -> None:
    """Copy a device preset and record module times and peaks for the run."""
    raw = yaml.safe_load(preset.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"preset root must be a mapping: {preset}")
    profile = dict(raw.get("profile") or {})
    profile["mode"] = "time_mem_module"
    raw["profile"] = profile
    gigabytes = preset_memory_gb(preset)
    body = yaml.safe_dump(raw, sort_keys=False)
    dest.write_text(f"# {preset.stem}, emulated {gigabytes} GB\n{body}", encoding="utf-8")


def runnable_attention(requested: str, *, sage_installed: bool, flash3_installed: bool) -> str:
    """Attention engine this host can construct for a preset.

    Consumer presets ask for SageAttention. An H100 environment has
    FlashAttention 3 and does not install Sage. The memory test still runs;
    the substitution is printed and stored in ``launch.json``.
    """
    if requested == "sage" and not sage_installed:
        return "flash_attention_3" if flash3_installed else "sdpa"
    if requested == "flash_attention_3" and not flash3_installed:
        return "sdpa"
    return requested


def _installed_attention() -> tuple[bool, bool]:
    from kandinsky.runtime.kernels.flash_attention_3 import flash_attention_3
    from kandinsky.runtime.kernels.sage import sage

    return sage is not None, flash_attention_3 is not None


def assign_gpus(names: list[str], gpus: list[int]) -> dict[int, list[str]]:
    """Round-robin ``names`` across ``gpus``. Each GPU keeps that order."""
    if not gpus:
        raise ValueError("no GPUs to assign")
    buckets: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    order = list(buckets)
    for index, name in enumerate(names):
        buckets[order[index % len(order)]].append(name)
    return buckets


def _unique(gpus: list[int]) -> list[int]:
    seen: list[int] = []
    for gpu in gpus:
        if gpu not in seen:
            seen.append(gpu)
    return seen


def _free_gpus(threshold_mib: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        try:
            index, used = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if used < threshold_mib:
            result.append(index)
    return result


def _open_run(home: Path) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root = home / "outputs" / f"memfit_{stamp}"
    if root.exists():
        root = home / "outputs" / f"memfit_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S-%f')}"
    for name in ("generations", "logs", "configs", "profiles"):
        (root / name).mkdir(parents=True)
    return root


def _gb(nbytes: int) -> str:
    whole, frac = divmod(nbytes, _GIB)
    if frac == 0:
        return f"{whole} GB"
    return f"{nbytes / _GIB:.2f} GB"


def _is_oom(exc: BaseException) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _one_job(spec: dict) -> JobResult:
    import torch

    from kandinsky.pipeline.factory import get_pipeline

    log_path = Path(spec["log"])
    held: list[DriverBlock] = []
    pipe = None
    free_bytes: int | None = None
    held_bytes = 0
    try:
        target = int(spec["target_bytes"])
        held, free_bytes = hold_free_memory(target, torch.device("cuda:0"))
        held_bytes = sum(block.nbytes for block in held)
        print(
            f"  [cuda:{spec['gpu']}] {spec['name']} reserved {_gb(held_bytes)}, {_gb(free_bytes)} free",
            flush=True,
        )
        started = time.perf_counter()
        pipe = get_pipeline(spec["config"], device="cuda:0", attention_engine=spec["attention"])
        result = pipe(
            text=spec["prompt"],
            seed=int(spec["seed"]),
            save_path=spec["out"],
            sample_audio=True,
            show_progress=True,
        )
        elapsed = time.perf_counter() - started
        # Keep ``held`` referenced until the clip and the profile are on disk.
        if not held:
            raise RuntimeError("reservation was released before generation finished")
        out_path = Path(spec["out"])
        if not out_path.is_file():
            raise RuntimeError(f"missing output {out_path}")
        profile_path = pipe.profile.path
        profile = None if profile_path is None else str(profile_path)
        kind = "frames + audio" if result.audio is not None else "frames"
        print(
            f"  [cuda:{spec['gpu']}] {spec['name']} saved {result.frames.shape[2]} {kind} → {result.path}",
            flush=True,
        )
        log_path.write_text(
            f"OK {elapsed:.3f}s free={free_bytes} held={held_bytes} target={target} "
            f"attention={spec['attention']}\nprofile {profile}\n",
            encoding="utf-8",
        )
        return JobResult(
            name=spec["name"],
            gpu=int(spec["gpu"]),
            target_bytes=target,
            free_bytes=free_bytes,
            held_bytes=held_bytes,
            success=True,
            seconds=elapsed,
            profile=profile,
            log=str(log_path),
        )
    except Exception as exc:
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        message = f"{type(exc).__name__}: {exc}"
        if _is_oom(exc):
            message = f"{message} (does not fit in emulated {int(spec['target_bytes']) // _GIB} GB)"
        return JobResult(
            name=spec["name"],
            gpu=int(spec["gpu"]),
            target_bytes=int(spec["target_bytes"]),
            free_bytes=free_bytes,
            held_bytes=held_bytes,
            success=False,
            seconds=None,
            profile=None,
            log=str(log_path),
            error=message,
        )
    finally:
        del pipe
        _free_blocks(held)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _require_visible_gpu(gpu: int) -> None:
    """The worker process must already see only ``gpu``.

    Importing ``kandinsky`` initializes CUDA. Setting ``CUDA_VISIBLE_DEVICES``
    inside this function cannot hide cards that were visible at import time.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(gpu):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES is {visible!r}; expected {gpu}. "
            "Start the worker with that variable set before Python imports kandinsky."
        )


def _run_gpu(specs: list[dict]) -> list[JobResult]:
    """Run one GPU's presets. The process was started with one visible card."""
    if not specs:
        return []
    gpu = int(specs[0]["gpu"])
    if any(int(item["gpu"]) != gpu for item in specs):
        raise RuntimeError("a worker was given more than one GPU")
    _require_visible_gpu(gpu)
    os.environ["KANDINSKY_LOG_DIR"] = str(specs[0]["log_dir"])
    os.environ["KANDINSKY_PROFILE_DIR"] = str(specs[0]["profile_dir"])
    import torch

    from kandinsky.cli import configure_logging

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible GPU for cuda:{gpu}, found {torch.cuda.device_count()}"
        )
    configure_logging()
    return [_one_job(spec) for spec in specs]


def _worker_entry(spec_path: Path, result_path: Path) -> int:
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    results = _run_gpu(specs)
    result_path.write_text(json.dumps([asdict(item) for item in results]) + "\n", encoding="utf-8")
    return 0 if all(item.success for item in results) else 1


def _worker_env(gpu: int) -> dict[str, str]:
    """Environment for one GPU worker.

    PyTorch 2.14 logs a C++ warning for every ``cudaMalloc`` that fails,
    including the first attempt the caching allocator recovers by releasing
    cached blocks and retrying. On a tight VAE decode that line says "OOM"
    even though the retry succeeds and the clip is saved. ``ERROR`` hides
    that warning. A real failure is still ``torch.OutOfMemoryError``. An
    explicit ``TORCH_CPP_LOG_LEVEL`` from the parent process is kept.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    return env


def _launch_workers(groups: list[list[dict]], run_dir: Path) -> list[JobResult]:
    """One subprocess per GPU, started with that card already selected."""
    procs: list[tuple[subprocess.Popen, Path, list[dict]]] = []
    for specs in groups:
        gpu = int(specs[0]["gpu"])
        spec_path = run_dir / "logs" / f"cuda{gpu}-specs.json"
        result_path = run_dir / "logs" / f"cuda{gpu}-results.json"
        spec_path.write_text(json.dumps(specs) + "\n", encoding="utf-8")
        env = _worker_env(gpu)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "kandinsky.tools.memfit", "--worker", str(spec_path), str(result_path)],
            env=env,
        )
        procs.append((proc, result_path, specs))
    results: list[JobResult] = []
    for proc, result_path, specs in procs:
        code = proc.wait()
        if result_path.is_file():
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            results.extend(JobResult(**item) for item in payload)
            continue
        message = f"worker exited {code} without results"
        for spec in specs:
            results.append(
                JobResult(
                    name=spec["name"],
                    gpu=int(spec["gpu"]),
                    target_bytes=int(spec["target_bytes"]),
                    free_bytes=None,
                    held_bytes=0,
                    success=False,
                    seconds=None,
                    profile=None,
                    log=str(spec["log"]),
                    error=message,
                )
            )
    return results


def _main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "--worker":
        return _worker_entry(Path(argv[1]), Path(argv[2]))
    print("use kandy memfit", file=sys.stderr)
    return 2


def _print_summary(results: list[JobResult], wall: float, run: Path) -> bool:
    print(f"\n{'Preset':<16} {'GPU':>4} {'Target':>8} {'Free':>10} {'Result':>6} {'Time':>8}")
    print("-" * 58)
    all_ok = True
    for result in results:
        status = "OK" if result.success else "FAIL"
        target = f"{result.target_bytes // _GIB} GB"
        free = "n/a" if result.free_bytes is None else _gb(result.free_bytes)
        shown = "n/a" if result.seconds is None else f"{result.seconds:7.1f}s"
        hint = "" if result.success else f"  <- {result.log}"
        print(f"{result.name:<16} {result.gpu:>4} {target:>8} {free:>10} {status:>6} {shown:>8}{hint}")
        all_ok = all_ok and result.success
    print("-" * 58)
    verdict = "All presets fit." if all_ok else "Some presets did not fit."
    print("The reservation is driver memory, outside the PyTorch allocator.")
    print(f"Wall time: {wall:.0f}s   {verdict}")
    print(f"Run: {run}")
    (run / "results.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return all_ok


def run(
    *,
    gpus: list[int] | None = None,
    prompt: str | None = None,
    seed: int = 0,
    mem_threshold: int = _FREE_GPU_THRESHOLD_MIB,
) -> int:
    """Run every consumer preset. Returns a process status code."""
    from kandinsky.tools.bench import DEFAULT_PROMPT

    text = DEFAULT_PROMPT if prompt is None else prompt
    home = kandinsky_home()
    home.mkdir(parents=True, exist_ok=True)
    presets = consumer_presets()
    if not presets:
        print(f"No 16, 24, or 32 GB presets in {_DEVICES}", file=sys.stderr)
        return 1
    selected = _unique(gpus) if gpus else _free_gpus(mem_threshold)
    if not selected:
        print(
            f"No free GPUs found (memory.used < {mem_threshold} MiB). Pass --gpus.",
            file=sys.stderr,
        )
        return 1

    run_dir = _open_run(home)
    by_name = {preset.name: preset for preset in presets}
    buckets = assign_gpus([preset.name for preset in presets], selected)
    sage_installed, flash3_installed = _installed_attention()
    written: dict[str, Path] = {}
    engines: dict[str, str] = {}
    for preset in presets:
        dest = run_dir / "configs" / f"{preset.name}.yaml"
        write_job_config(preset.path, dest)
        written[preset.name] = dest
        loaded = load_config(dest)
        requested = loaded.attention.engine
        engine = runnable_attention(
            requested, sage_installed=sage_installed, flash3_installed=flash3_installed
        )
        engines[preset.name] = engine
        shown = requested if engine == requested else f"{requested} -> {engine}"
        print(
            f"{preset.name}: emulate {_gb(preset.memory_bytes)}, "
            f"{loaded.checkpoint} {loaded.generation.height}x{loaded.generation.width} "
            f"offload={loaded.offload.strategy} quant={loaded.text_embedder.quantized_qwen} "
            f"attention={shown}"
        )

    groups: list[list[dict]] = []
    recorded = []
    for gpu, names in buckets.items():
        specs = []
        for name in names:
            preset = by_name[name]
            specs.append(
                {
                    "name": name,
                    "gpu": gpu,
                    "config": str(written[name]),
                    "target_bytes": preset.memory_bytes,
                    "attention": engines[name],
                    "prompt": text,
                    "seed": seed,
                    "out": str(run_dir / "generations" / f"{name}.mp4"),
                    "log": str(run_dir / "logs" / f"{name}.log"),
                    "log_dir": str(run_dir / "logs"),
                    "profile_dir": str(run_dir / "profiles"),
                }
            )
            recorded.append(
                {
                    "name": name,
                    "gpu": gpu,
                    "target_bytes": preset.memory_bytes,
                    "attention": engines[name],
                    "config": f"configs/{name}.yaml",
                }
            )
        if specs:
            groups.append(specs)
            print(f"cuda:{gpu}: {', '.join(names)}")

    payload = {
        "name": "memfit",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt": text,
        "seed": seed,
        "gpus": selected,
        "free_tolerance_bytes": _TOLERANCE,
        "profile_dir": str(run_dir / "profiles"),
        "jobs": recorded,
    }
    (run_dir / "launch.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(presets)} presets on {len(groups)} GPU(s)")
    print(f"Run: {run_dir}")
    print(f"KANDINSKY_HOME={home}")
    print()

    wall_started = time.monotonic()
    results = _launch_workers(groups, run_dir)
    order = {preset.name: index for index, preset in enumerate(presets)}
    results.sort(key=lambda item: order[item.name])
    ok = _print_summary(results, time.monotonic() - wall_started, run_dir)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
