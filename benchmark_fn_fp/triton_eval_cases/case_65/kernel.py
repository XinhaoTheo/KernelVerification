import itertools
import numpy as np
import torch
import triton
import triton.language as tl

SEED = 941200
N, D = 8, 32
SMOKE_WORDS = ((2,), (0, 2), (1, 2), (0, 0, 2), (1, 1, 2))


@triton.jit
def _reorder(Order, Permutation, Next, N: tl.constexpr):
    row = tl.arange(0, N)
    source = tl.load(Permutation + row)
    tl.store(Next + row, tl.load(Order + source))


@triton.jit
def _append(Cache, Order, Delta, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)[:, None]
    col = tl.arange(0, D)[None, :]
    source = tl.load(Order + row)
    current = tl.load(Cache + row * D + col)
    increment = tl.load(Delta + source * D + col)
    tl.store(Cache + row * D + col, current + increment)


@triton.jit
def _materialize(Cache, Order, Output, N: tl.constexpr, D: tl.constexpr):
    row = tl.arange(0, N)[:, None]
    col = tl.arange(0, D)[None, :]
    physical = tl.load(Order + row)
    tl.store(Output + row * D + col, tl.load(Cache + physical * D + col))


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    initial = (rng.integers(-64, 65, size=(N, D)) / 16).astype(np.float32)
    increment = (rng.integers(-32, 33, size=(N, D)) / 16).astype(np.float32)
    permutations = []
    for _ in range(2):
        pairs = rng.permutation(N).reshape(-1, 2)
        permutation = np.arange(N, dtype=np.int32)
        for a, b in pairs:
            permutation[a], permutation[b] = b, a
        permutations.append(permutation)
    return initial, increment, np.stack(permutations)


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())


def run_sequence(initial, increment, permutations, operations):
    physical = initial.clone()
    order = torch.arange(N, device=initial.device, dtype=torch.int32)
    for operation in tuple(operations):
        if operation in (0, 1):
            next_order = torch.empty_like(order)
            _reorder[(1,)](order, permutations[operation], next_order, N=N, num_warps=1)
            order = next_order
        elif operation == 2:
            _append[(1,)](physical, order, increment, N=N, D=D, num_warps=1)
        else:
            raise ValueError("operations must contain only 0, 1, or 2")
    output = torch.empty_like(initial)
    _materialize[(1,)](physical, order, output, N=N, D=D, num_warps=1)
    return output


def run(initial, increment, permutations, operations=(0, 1, 2)):
    return run_sequence(initial, increment, permutations, operations)


def initial_probe(device="cuda"):
    cpu = make_inputs_numpy()
    gpu = make_inputs(device)
    rows = []
    for operations in SMOKE_WORDS:
        reference = cpu[0].astype(np.float64).copy()
        for operation in operations:
            reference = (reference[cpu[2][operation]].copy() if operation < 2
                         else reference + cpu[1].astype(np.float64))
        actual = run_sequence(*gpu, operations).detach().cpu().numpy()
        error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        passed = actual.shape == (N, D) and actual.dtype == np.float32 and bool(
            np.isfinite(actual).all()) and error <= 0.00001
        rows.append({"operations": list(operations), "max_absolute_error": error,
                     "passed": bool(passed)})
    return {"scope": "five smoke words, each from fresh state",
            "all_passed": all(row["passed"] for row in rows), "records": rows}
