import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from kernel import state_passing  # the real, unmodified kernel -- used for BOTH sides


def test_kernel():
    assert torch.cuda.is_available(), "this case requires a CUDA GPU (Triton)"
    device = "cuda"
    torch.manual_seed(0)

    def run_fine_and_coarse(n_coarse, m_fine):
        dim = 4
        coarse_decay = -0.05
        coarse_new_states = torch.randn(n_coarse, dim, device=device) * 0.1

        # Coarse: n_coarse real chunks.
        dA_coarse = torch.full((n_coarse,), coarse_decay, device=device)
        coarse_out = state_passing(coarse_new_states, dA_coarse)

        # Fine: each coarse chunk becomes m_fine real chunks whose decays sum
        # to the same coarse_decay, with the new_states contribution placed
        # entirely on the LAST fine sub-step -- exactly equivalent in real
        # arithmetic (intermediate zero-contribution steps only rescale).
        fine_decay_step = coarse_decay / m_fine
        dA_fine = torch.full((n_coarse * m_fine,), fine_decay_step, device=device)
        fine_new_states = torch.zeros(n_coarse * m_fine, dim, device=device)
        fine_new_states[m_fine - 1 :: m_fine] = coarse_new_states
        fine_out = state_passing(fine_new_states, dA_fine)

        return coarse_out, fine_out

    coarse_short, fine_short = run_fine_and_coarse(n_coarse=3, m_fine=4)
    short_pass = torch.allclose(fine_short, coarse_short, rtol=1e-5, atol=1e-5)
    short_diff = (fine_short - coarse_short).abs().max().item()

    coarse_long, fine_long = run_fine_and_coarse(n_coarse=200, m_fine=500)
    long_pass = torch.allclose(fine_long, coarse_long, rtol=1e-5, atol=1e-5)
    long_diff = (fine_long - coarse_long).abs().max().item()

    print(f"short (n_coarse=3, m_fine=4): fine vs coarse allclose(tight)={short_pass}, diff={short_diff:.8f}")
    print(f"long (n_coarse=200, m_fine=500 => 100000 fine steps): fine vs coarse allclose(tight)={long_pass}, diff={long_diff:.8f}")
    grows_as_expected = long_diff >= short_diff
    print("FP DEMONSTRATED" if grows_as_expected else "informational: fp32 drift magnitude depends on inputs")
    # What a conventional CI test would conclude: 严容差 + 真实长度；短跑漂移还没显现（这个选择可争议）
    print(f"NAIVE_ALLCLOSE_VERDICT: {long_pass}")
    return grows_as_expected


if __name__ == "__main__":
    test_kernel()
