"""Pick the least-loaded GPU and pin CUDA_VISIBLE_DEVICES to it.

Call pin_freest_gpu() BEFORE any CUDA-touching import (torch, triton, or our
generator which transitively imports them). Once CUDA is initialized in a
process, the visible-devices mapping is locked.

Subprocess workers spawned by KernelAgent inherit the env var, so a single
call at the top of our entry script is enough.

Respects an existing CUDA_VISIBLE_DEVICES (from .env or shell) as a user
override unless force=True is passed.
"""

from __future__ import annotations

import os
import subprocess
import sys

from dotenv import load_dotenv

_dotenv_loaded = False


def _rank_candidates() -> list[int]:
    """Rank GPU indices by nvidia-smi (lowest memory used, then lowest util)."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout

    rows: list[tuple[int, int, int]] = []  # (memory_used, util, index)
    for line in out.strip().splitlines():
        idx, util, mem_used = [int(s.strip()) for s in line.split(",")]
        rows.append((mem_used, util, idx))

    if not rows:
        raise RuntimeError("nvidia-smi returned no GPUs")
    rows.sort()
    return [r[2] for r in rows]


def _probe_gpu(idx: int, timeout_s: int = 20) -> bool:
    """Try allocating a tiny CUDA tensor on `idx` in a fresh subprocess.

    nvidia-smi can report a GPU as idle (0 MiB / 0% util) while another
    tenant still holds it in a way that blocks new contexts. The only
    reliable check is to attempt an actual allocation.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(idx)}
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; torch.randn(8, device='cuda')"],
        env=env,
        capture_output=True,
        timeout=timeout_s,
    )
    return probe.returncode == 0


def pick_usable_gpu(max_probe: int = 4) -> int:
    """Return index of the first nvidia-smi-ranked GPU that passes an alloc probe.

    Probes at most `max_probe` candidates from the top of the ranking so we
    don't waste time hammering every GPU when most are unusable.
    """
    ranked = _rank_candidates()
    for idx in ranked[:max_probe]:
        if _probe_gpu(idx):
            return idx
        print(f"[gpu_pick] GPU {idx} ranked free but failed allocation probe; trying next")
    raise RuntimeError(
        f"No usable GPU among top {max_probe} candidates: {ranked[:max_probe]}"
    )


def pin_freest_gpu(*, force: bool = False) -> str:
    """Set CUDA_VISIBLE_DEVICES to a usable GPU; return whatever ends up set.

    If CUDA_VISIBLE_DEVICES is already in the environment (from .env or shell
    export), respect it and skip auto-pick. Pass force=True to override.
    """
    global _dotenv_loaded
    if not _dotenv_loaded:
        load_dotenv()
        _dotenv_loaded = True

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing and not force:
        print(f"[gpu_pick] respecting CUDA_VISIBLE_DEVICES={existing} (user override)")
        return existing

    idx = pick_usable_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    print(f"[gpu_pick] auto-pinned CUDA_VISIBLE_DEVICES={idx} (probed and usable)")
    return str(idx)
