import torch
from kernel import requantize


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    torch.manual_seed(0)
    device = "cuda"
    x0 = torch.randn(4096, device=device)

    # Conventional test: one round trip, compared with a tolerance.
    once = requantize(x0.clone())
    # An INT8 round trip is only accurate to about absmax/127 per element, so a
    # test written for this kernel uses a tolerance matched to the format.
    level = x0.abs().max().item() / 127.0
    short_pass = torch.allclose(once, x0, rtol=0.0, atol=2 * level)
    one_step_bias = (once.abs().sum() / x0.abs().sum()).item()

    # Long-horizon behaviour: one optimizer step is "add the gradient, then store
    # the parameter back in INT8". A per-step rounding bias compounds here.
    grad = torch.full_like(x0, 1e-3)
    x = x0.clone()
    x_ref = x0.clone()
    for _ in range(2000):
        x = requantize(x + grad)
        x_ref = x_ref + grad
    long_pass = torch.allclose(x, x_ref, rtol=0.0, atol=2 * level)
    tracked = ((x - x0).mean() / (x_ref - x0).mean()).item()

    print(f"1 round trip: allclose={short_pass}, magnitude ratio={one_step_bias:.6f}")
    print(f"2000 optimizer steps: allclose={long_pass}, fraction of the update actually tracked={tracked:.4f}")
    # What a conventional CI test would conclude: a kernel unit test runs the
    # operation once.
    print(f"NAIVE_ALLCLOSE_VERDICT: {short_pass}")
    return short_pass and not long_pass


if __name__ == "__main__":
    test_kernel()
