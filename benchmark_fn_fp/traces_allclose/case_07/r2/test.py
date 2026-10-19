import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import stochastic_round_to_grid, STEP  # real Triton Philox RNG


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"

    n = 4096
    x = torch.full((n,), 0.137, device=device)

    run_a = stochastic_round_to_grid(x, seed=1)
    run_b = stochastic_round_to_grid(x, seed=2)
    naive_pass = torch.allclose(run_a, run_b, rtol=1e-2, atol=1e-2)

    mean_estimate = run_a.mean().item()
    unbiased_pass = abs(mean_estimate - 0.137) < 4 * STEP / (n ** 0.5)

    print(f"real Triton Philox RNG (GPU): two independent seeds, elementwise allclose = {naive_pass} (expected False)")
    print(f"mean over {n} draws = {mean_estimate:.5f} vs true 0.13700; unbiased check = {unbiased_pass}")
    print("FP DEMONSTRATED" if (not naive_pass) and unbiased_pass else "tune constants")
    # What a conventional CI test would conclude: 把一次运行当作另一次的精确参考
    print(f"NAIVE_ALLCLOSE_VERDICT: {naive_pass}")
    return (not naive_pass) and unbiased_pass


if __name__ == "__main__":
    test_kernel()
