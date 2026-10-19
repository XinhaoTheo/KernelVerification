
import sys, json, random
sys.path.insert(0, "/root/cases/case_159")
import torch
from kernel import run, reference, validate_inputs

device = "cuda"
fail = None
n_tests = 0
for trial in range(300):
    B = random.randint(1, 4)
    N = random.randint(1, 32)
    g = torch.Generator().manual_seed(trial)
    if trial % 3 == 0:
        # extreme values: -32768/32767 bias & initial, matrix -1/1
        matrix = torch.randint(0, 2, (B, N, 2, 2), generator=g, dtype=torch.int8) * 2 - 1
        bias = torch.randint(0, 2, (B, N, 2), generator=g, dtype=torch.int16) * 65535 - 32768
        initial = torch.randint(0, 2, (B, 2), generator=g, dtype=torch.int16) * 65535 - 32768
    else:
        matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
        bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
        initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
    m, b, i = matrix.to(device), bias.to(device), initial.to(device)
    out = run(m, b, i)
    ref = reference(m, b, i)
    n_tests += 1
    if not torch.equal(out, ref):
        fail = {"trial": trial, "B": B, "N": N}
        break
# also full B=64 N=32 and every N in 1..32
for N in range(1, 33):
    B = 64
    g = torch.Generator().manual_seed(1000 + N)
    matrix = torch.randint(-1, 2, (B, N, 2, 2), generator=g, dtype=torch.int8)
    bias = torch.randint(-32768, 32768, (B, N, 2), generator=g, dtype=torch.int16)
    initial = torch.randint(-32768, 32768, (B, 2), generator=g, dtype=torch.int16)
    m, b, i = matrix.to(device), bias.to(device), initial.to(device)
    out = run(m, b, i); ref = reference(m, b, i)
    n_tests += 1
    if not torch.equal(out, ref):
        fail = {"B": 64, "N": N}
        break
print(json.dumps({"metric": "exact equality (torch.equal) vs sequential reference", "tests": n_tests, "failure": fail}))
