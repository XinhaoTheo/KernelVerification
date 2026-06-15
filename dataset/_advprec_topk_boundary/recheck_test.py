import sys
import torch
import torch.nn as nn

from kernel import kernel_function
from kverify_compare import compare_outputs


class Model(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(scores, self.k, dim=-1)
        return indices


k = 8
num_keys = 256


def ref(scores, k=8):
    model = Model(k)
    return model(scores)


results = {}

# CASE 1: standard
case = "standard"
try:
    torch.manual_seed(0)
    scores = torch.randn(16, num_keys)
    out = kernel_function(scores, k)
    r = ref(scores, k)
    matches, max_diff, detail = compare_outputs(out, r)
    if matches:
        print(f"CASE {case}: PASS")
        results[case] = True
    else:
        print(f"CASE {case}: FAIL {detail}")
        results[case] = False
except Exception as e:
    print(f"CASE {case}: FAIL raised {type(e).__name__}: {e}")
    results[case] = False

# CASE 2: noncontig_stride2
case = "noncontig_stride2"
try:
    torch.manual_seed(1)
    big = torch.randn(16, num_keys * 2)
    scores = big[:, ::2]  # non-contiguous, same logical shape
    assert scores.shape == (16, num_keys)
    assert not scores.is_contiguous()
    out = kernel_function(scores, k)
    r = ref(scores, k)
    matches, max_diff, detail = compare_outputs(out, r)
    if matches:
        print(f"CASE {case}: PASS")
    else:
        print(f"CASE {case}: FAIL {detail}")
except Exception as e:
    print(f"CASE {case}: FAIL raised {type(e).__name__}: {e}")

# CASE 3: noncontig_transpose
case = "noncontig_transpose"
try:
    torch.manual_seed(2)
    # Input is 2D (16, num_keys). Transpose gives (num_keys, 16).
    # We run top-k on last dim, so k must be <= 16 here.
    scores_orig = torch.randn(16, num_keys)
    scores = scores_orig.t()  # shape (num_keys, 16), non-contiguous
    k_t = min(k, 16)
    out = kernel_function(scores, k_t)
    r = ref(scores, k_t)
    matches, max_diff, detail = compare_outputs(out, r)
    if matches:
        print(f"CASE {case}: PASS")
    else:
        print(f"CASE {case}: FAIL {detail}")
except Exception as e:
    print(f"CASE {case}: FAIL raised {type(e).__name__}: {e}")

# CASE 4: odd_size (num_keys + 1)
case = "odd_size"
try:
    torch.manual_seed(3)
    scores = torch.randn(16, num_keys + 1)
    out = kernel_function(scores, k)
    r = ref(scores, k)
    matches, max_diff, detail = compare_outputs(out, r)
    if matches:
        print(f"CASE {case}: PASS")
    else:
        print(f"CASE {case}: FAIL {detail}")
except Exception as e:
    print(f"CASE {case}: FAIL raised {type(e).__name__}: {e}")

# CASE 5: empty
case = "empty"
try:
    scores = torch.randn(0, num_keys)
    out = kernel_function(scores, k)
    r = ref(scores, k)
    matches, max_diff, detail = compare_outputs(out, r)
    if matches:
        print(f"CASE {case}: PASS")
    else:
        print(f"CASE {case}: FAIL {detail}")
except Exception as e:
    print(f"CASE {case}: FAIL raised {type(e).__name__}: {e}")


standard_passed = results.get("standard", False)
if __name__ == "__main__":
    sys.exit(0 if standard_passed else 1)