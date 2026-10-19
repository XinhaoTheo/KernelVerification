import itertools
import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _canonicalize(Features, Anchors, Out):
    view = tl.program_id(0)
    column = tl.program_id(1)
    anchor = tl.load(Anchors + view * 4 + column)
    a0 = tl.abs(tl.load(Anchors + view * 4))
    a1 = tl.abs(tl.load(Anchors + view * 4 + 1))
    a2 = tl.abs(tl.load(Anchors + view * 4 + 2))
    a3 = tl.abs(tl.load(Anchors + view * 4 + 3))
    magnitude = tl.abs(anchor)
    slot = ((a0 < magnitude).to(tl.int32) + (a1 < magnitude).to(tl.int32)
            + (a2 < magnitude).to(tl.int32) + (a3 < magnitude).to(tl.int32))
    rows = tl.arange(0, 8)
    values = tl.load(Features + view * 32 + rows * 4 + column)
    sign = tl.where(anchor >= 0, 1.0, -1.0)
    tl.store(Out + view * 32 + rows * 4 + slot, values * sign)


def run(features, anchors):
    output = torch.empty_like(features)
    _canonicalize[(2, 4)](features, anchors, output, num_warps=1)
    return output


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(171249))
    features = rng.normal(0.0, 1.0, (2, 8, 4)).astype(np.float32)
    features[0, :, 2:] = 0
    features[1, :, (0, 3)] = 0
    anchors = rng.normal(0.0, 1.0, (2, 4)).astype(np.float32)
    return features, anchors


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(array).to(device) for array in make_inputs_numpy())


def initial_probe():
    features, anchors = make_inputs()
    output = run(features, anchors)
    x = features.detach().cpu().numpy().astype(np.float64)
    y = output.detach().cpu().numpy().astype(np.float64)
    local = []
    for view in range(2):
        best_error = float("inf")
        best_transform = None
        for permutation in itertools.permutations(range(4)):
            for signs in itertools.product((-1, 1), repeat=4):
                expected = x[view][:, permutation] * np.asarray(signs)
                error = float(np.linalg.norm(y[view] - expected) /
                              max(np.linalg.norm(x[view]), 1e-12))
                if error < best_error:
                    best_error = error
                    best_transform = {"permutation": list(permutation), "signs": list(signs)}
        local.append({"view": view, "relative_error": best_error,
                      "chosen_transform": best_transform})
    gram_error = float(np.linalg.norm(y @ y.transpose(0, 2, 1) -
                                      x @ x.transpose(0, 2, 1)))
    return {"per_view_alignment": local, "within_view_gram_error": gram_error,
            "tolerance": 0.05,
            "passed": bool(all(row["relative_error"] <= 0.05 for row in local))}
