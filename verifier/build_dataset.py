"""CLI: generate kernels for a list of problems and save them as dataset entries.

Run once (or whenever you want fresh kernels). After that, verifier.run reads
from the dataset offline — no KernelAgent calls, no $$$, no GPU contention.

Sources of problems:
  - PROBLEMS dict: hand-written inline prompts (small, for quick iteration)
  - KernelBench files: read directly from KernelBench/KernelBench/levelN/*.py
  - CURATED_KB: a hand-picked 10-problem progression from KernelBench level 1

Usage:
    uv run kv-build                                     # build PROBLEMS (default)
    uv run kv-build --problem elem_add
    uv run kv-build --kernelbench level1/19_ReLU.py    # one KB file
    uv run kv-build --curated                          # all 10 from CURATED_KB
    uv run kv-build --list                             # show what would be built, don't run
"""

from __future__ import annotations

# MUST run before any torch / triton / generator import.
from .gpu_pick import pin_freest_gpu

pin_freest_gpu()

import argparse  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from triton_kernel_agent import TritonKernelAgent  # noqa: E402

from . import dataset  # noqa: E402
from .generator import generate_kernel  # noqa: E402

# Root of the KernelBench problem files (note: the repo nests
# KernelBench/KernelBench/levelN/*.py).
KB_ROOT = Path("KernelBench/KernelBench")

# Hand-picked 10-problem progression: trivial elem-wise -> reductions ->
# norms -> matmul -> tricky (cumsum). Order is roughly easy-to-hard so a
# rate-limited run that gets killed mid-way still produces useful entries.
CURATED_KB: list[str] = [
    "level1/19_ReLU.py",
    "level1/21_Sigmoid.py",
    "level1/26_GELU_.py",
    "level1/47_Sum_reduction_over_a_dimension.py",
    "level1/23_Softmax.py",
    "level1/36_RMSNorm_.py",
    "level1/40_LayerNorm.py",
    "level1/1_Square_matrix_multiplication_.py",
    "level1/88_MinGPTNewGelu.py",
    "level1/89_cumsum.py",
]

PROBLEMS: dict[str, str] = {
    "elem_add": """
Write a Triton kernel for element-wise addition of two vectors:

import torch

class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b

vector_size = 1024
dtype = torch.float32

def get_inputs():
    a = torch.randn(vector_size, dtype=dtype, device='cuda')
    b = torch.randn(vector_size, dtype=dtype, device='cuda')
    return [a, b]

def get_init_inputs():
    return []
""".strip(),
    "softmax": """
Write a Triton kernel for row-wise softmax along the last dimension:

import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)

batch, dim = 32, 1024
dtype = torch.float32

def get_inputs():
    return [torch.randn(batch, dim, dtype=dtype, device='cuda')]

def get_init_inputs():
    return []
""".strip(),
}


def kb_entry_name(rel_path: str) -> str:
    """Derive a stable, lowercased entry name from a KernelBench file path.

    Examples:
        level1/19_ReLU.py                          -> kb_level1_19_relu
        level1/47_Sum_reduction_over_a_dimension.py -> kb_level1_47_sum_reduction_over_a_dimension
        level1/26_GELU_.py                         -> kb_level1_26_gelu  (trailing _ stripped)
    """
    stem = Path(rel_path).with_suffix("")  # drop .py
    name = "kb_" + str(stem).replace("/", "_").replace("\\", "_").lower()
    name = re.sub(r"_+", "_", name).rstrip("_")
    return name


def load_kb(rel_path: str) -> str:
    """Read a KernelBench problem file's full text, to be fed as problem_description."""
    full = KB_ROOT / rel_path
    if not full.exists():
        raise FileNotFoundError(f"KernelBench file not found: {full}")
    return full.read_text()


def _resolve_jobs(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return list of (entry_name, problem_description) to build."""
    jobs: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(name: str, body: str) -> None:
        if name in seen:
            return
        seen.add(name)
        jobs.append((name, body))

    if args.curated:
        for rel in CURATED_KB:
            add(kb_entry_name(rel), load_kb(rel))

    for rel in args.kernelbench or []:
        add(kb_entry_name(rel), load_kb(rel))

    if args.problem:
        for name in args.problem:
            add(name, PROBLEMS[name])

    # If nothing was specified, default to all of PROBLEMS (preserves old behavior).
    if not jobs and not (args.curated or args.kernelbench or args.problem):
        for name in sorted(PROBLEMS):
            add(name, PROBLEMS[name])

    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--problem",
        action="append",
        choices=sorted(PROBLEMS),
        help="Inline problem name; repeat to build multiple.",
    )
    ap.add_argument(
        "--kernelbench",
        action="append",
        metavar="REL_PATH",
        help=f"Path under {KB_ROOT}/ (e.g. level1/19_ReLU.py); repeatable.",
    )
    ap.add_argument(
        "--curated",
        action="store_true",
        help="Build the 10 hand-picked KernelBench problems in CURATED_KB.",
    )
    ap.add_argument(
        "--out",
        default="dataset",
        help="Dataset root directory (default: dataset/)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print what would be built and exit (no LLM calls).",
    )
    args = ap.parse_args()

    try:
        jobs = _resolve_jobs(args)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not jobs:
        print("nothing to build (PROBLEMS is empty and no flags given)", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    print(f"Target dataset dir: {out_dir.resolve()}")
    print(f"Jobs ({len(jobs)}):")
    for name, body in jobs:
        first_line = body.splitlines()[0][:70]
        print(f"  - {name:<55s} | {first_line}")

    if args.list:
        return 0

    agent = TritonKernelAgent()
    summary: dict[str, int] = {}
    try:
        for name, body in jobs:
            print(f"\n=== {name} ===")
            t0 = time.time()
            try:
                artifact = generate_kernel(body, agent=agent)
            except Exception as e:
                print(f"  GENERATION FAILED: {type(e).__name__}: {e}")
                summary["exception"] = summary.get("exception", 0) + 1
                continue
            dt = time.time() - t0
            status = artifact.get("status", "unknown")
            rounds = artifact.get("rounds")
            print(
                f"  passed={artifact['passed']}  status={status}  "
                f"rounds={rounds}  time={dt:.1f}s"
            )

            path = dataset.save_entry(name, artifact, dataset_dir=out_dir)
            print(f"  saved -> {path}")
            summary[status] = summary.get(status, 0) + 1
    finally:
        agent.cleanup()

    print("\n=== SUMMARY (generation status) ===")
    total = sum(summary.values())
    for status, count in sorted(summary.items()):
        print(f"  {status:<25s} {count}")
    print(f"  {'total':<25s} {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
