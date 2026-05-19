"""Run KernelAgent on a small problem and dump everything it produced.

Usage:
    uv run python tests/inspect_generator.py
    uv run python tests/inspect_generator.py --problem softmax
"""

from __future__ import annotations

# MUST run before any torch / triton / generator import so the env var is set
# while CUDA is still uninitialized in this process AND in worker subprocesses.
from verifier.gpu_pick import pin_freest_gpu

pin_freest_gpu()

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from verifier.generator import generate_kernel  # noqa: E402

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


def _hr(title: str) -> None:
    print(f"\n{'=' * 6} {title} {'=' * (72 - len(title))}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="elem_add", choices=sorted(PROBLEMS))
    args = ap.parse_args()

    print(f"problem: {args.problem}")
    print(f"(this will call KernelAgent — expect 3-10 min on a fresh run)\n")

    t0 = time.time()
    artifact = generate_kernel(PROBLEMS[args.problem])
    dt = time.time() - t0

    _hr("SUMMARY")
    print(f"  passed         = {artifact['passed']}")
    print(f"  session_dir    = {artifact['session_dir']}")
    print(f"  kernel_code    = {len(artifact['kernel_code'])} chars")
    print(f"  test_code      = {len(artifact['test_code'])} chars")
    print(f"  generation_time = {dt:.1f}s")

    sd = Path(artifact["session_dir"])
    if sd.exists():
        _hr(f"SESSION DIR FILES ({sd})")
        for p in sorted(sd.iterdir()):
            size = p.stat().st_size
            print(f"  {size:>8}  {p.name}")

    _hr("KERNEL CODE")
    print(artifact["kernel_code"] or "(empty — generator failed)")

    _hr("TEST CODE")
    print(artifact["test_code"] or "(empty)")

    _hr("RAW RETURN")
    for k, v in artifact["raw"].items():
        if isinstance(v, str) and len(v) > 200:
            print(f"  {k}: <{len(v)} chars>")
        else:
            print(f"  {k}: {v}")

    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
