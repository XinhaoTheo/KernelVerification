import torch
from kernel import blocked_sum

RUNS = 20
N = 1 << 16
SCALE = 2.0 ** -7


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"

    codes = torch.randint(-127, 128, (N,), device=device, dtype=torch.int8)
    x = codes.to(torch.float32) * SCALE

    bits = {blocked_sum(x).view(torch.int32).item() for _ in range(RUNS)}
    bitwise_reproducible = len(bits) == 1

    # Exact in float64 as well: the grid is what makes the order irrelevant.
    ref_exact = x.double().sum().item()
    value = blocked_sum(x).item()
    exact = value == ref_exact

    # Why the order cannot matter here: the accumulator only ever holds integer
    # multiples of SCALE, and the largest integer it can reach stays inside the
    # range fp32 represents exactly.
    reachable = 127 * N
    headroom = 2 ** 24 / reachable

    # The same sum on a scale that is not a power of two, for contrast.
    off_grid = codes.to(torch.float32) * 0.1
    off_grid_bits = {blocked_sum(off_grid).view(torch.int32).item() for _ in range(RUNS)}

    print(f"{RUNS} runs bitwise identical = {bitwise_reproducible} (expected True), "
          f"{len(bits)} distinct bit pattern(s)")
    print(f"value equals the float64 sum = {exact} (expected True), "
          f"value={value!r}, ref={ref_exact!r}")
    print(f"largest reachable integer {reachable} vs 2^24 = {2 ** 24}, headroom={headroom:.2f}x")
    print(f"same data on a non-power-of-two scale: {len(off_grid_bits)} distinct bit pattern(s) "
          f"(expected > 1 -- the grid, not the kernel, is what fixes the bits)")
    # A conventional CI test compares against a stored value within a tolerance.
    print(f"NAIVE_ALLCLOSE_VERDICT: {bitwise_reproducible}")
    return bitwise_reproducible and exact


if __name__ == "__main__":
    test_kernel()
