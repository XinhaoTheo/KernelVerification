import torch
from kernel import blocked_sum

RUNS = 20
N = 1 << 16


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(N, device=device, dtype=torch.float32)

    values = [blocked_sum(x).item() for _ in range(RUNS)]
    bits = {torch.tensor(v, dtype=torch.float32).view(torch.int32).item() for v in values}
    bitwise_reproducible = len(bits) == 1
    spread = max(values) - min(values)

    # The value is never far off: only the last bits move, which is why an
    # accuracy check reports nothing while the stated contract is not met.
    ref_exact = x.double().sum().item()
    budget = 1e-6 * x.abs().sum().item()
    all_runs_accurate = all(abs(v - ref_exact) <= budget for v in values)

    # How much of the disorder the grid size accounts for: fewer blocks, fewer
    # landing orders. Recorded so a stable result at 64 blocks is visible as a
    # property of the launch, not a property of atomics.
    per_block = {}
    for block in (256, 1024, 4096):
        seen = {blocked_sum(x, block=block).view(torch.int32).item() for _ in range(RUNS)}
        per_block[(x.numel() + block - 1) // block] = len(seen)

    print(f"{RUNS} runs bitwise identical = {bitwise_reproducible} (expected False), "
          f"{len(bits)} distinct bit pattern(s), spread={spread:.3e}")
    print(f"every run within {budget:.3e} of the float64 value = {all_runs_accurate} "
          f"(expected True), ref={ref_exact:.6f}")
    print(f"distinct bit patterns by grid size: {per_block}")
    # A conventional CI test compares against a stored value within a tolerance.
    print(f"NAIVE_ALLCLOSE_VERDICT: {all_runs_accurate}")
    return (not bitwise_reproducible) and all_runs_accurate


if __name__ == "__main__":
    test_kernel()
