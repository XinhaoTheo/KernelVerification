import torch
from kernel import splitk_dot


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    n = 1 << 16
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)

    cand = splitk_dot(a, b).item()
    ref_seq = torch.dot(a, b).item()             # one sequential order
    ref_exact = torch.dot(a.double(), b.double()).item()

    # A CI check that requires reproducibility stores a golden value and demands
    # the kernel reproduce it exactly.
    runs = [splitk_dot(a, b).item() for _ in range(5)]
    bitwise_reproducible = len(set(runs)) == 1

    # What the contract actually asks for: the inner product, to within the
    # rounding the summation order can account for.
    budget = 1e-6 * (a.abs() * b.abs()).sum().item()
    all_runs_accurate = all(abs(r - ref_exact) <= budget for r in runs)
    spread = max(runs) - min(runs)

    print(f"5 runs bitwise identical = {bitwise_reproducible} (expected False), spread={spread:.3e}")
    print(f"every run within the reordering budget of the float64 value = {all_runs_accurate} "
          f"(expected True), budget={budget:.3e}, ref={ref_exact:.6f}")
    print(f"one sequential order for comparison: {ref_seq:.6f}")
    # What a conventional CI test would conclude: it demands a reproducible value.
    print(f"NAIVE_ALLCLOSE_VERDICT: {bitwise_reproducible}")
    return (not bitwise_reproducible) and all_runs_accurate


if __name__ == "__main__":
    test_kernel()
