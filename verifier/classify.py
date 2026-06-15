"""Operator-class router — the 分诊台 that runs BEFORE we judge correctness.

The core insight of precision_verification.md: "small numerical error ⇔ correct"
is only valid for SOME operators. Before picking a judge (J1 allclose / J2
selection / J3 downstream), we must first decide which CLASS the operator is in.

Classification is a property of the PROBLEM (the trusted PyTorch reference), NOT
of the candidate kernel — a buggy kernel doesn't change which judge to use. So
this module runs ONLY the reference from problem.txt, on small CPU inputs. It
never touches kernel.py, Triton, or the GPU.

Two axes (see precision_verification.md §1):
  - op_class: "preserve" (matmul/elementwise) | "compress" (softmax/activation)
              | "select" (sort/topk)
  - precision: "normal" | "low_bit" (fp8/fp4/int4)

Detection layers:
  D1  dtype           -> low_bit axis
  D2  output-type     -> integer output OR "output values are a subset of input
                         values" => select
  D3  gain probe      -> ‖Δoutput‖/ε at several input locations; near-zero gain
                         regions => compress; roughly uniform => preserve
  D4  fingerprints    -> softmax (sum-to-1 + range), saturation sweep (flat
                         output regions), discreteness (piecewise-constant jumps)

Judge mapping (precision_verification.md §5.2 / coverage matrix):
  preserve            -> J1   (allclose, high-precision reference)
  compress            -> J1 + adversarial distribution / compare pre-activation
  select              -> J2   (value-gap weighted / downstream output)
  low_bit (any class) -> J3 / abstain

Usage:
    uv run python -m verifier.classify                 # classify all entries
    uv run python -m verifier.classify kb_level1_23_softmax
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import dataset

SMALL = 16  # cap every tensor dimension to this when building probe inputs


# --------------------------------------------------------------------------- #
# Reconstruct the reference from problem.txt on small CPU inputs               #
# --------------------------------------------------------------------------- #

class _CapSizes:
    """Context manager that monkeypatches torch tensor constructors to cap every
    dimension to SMALL, so calling the spec's get_inputs() on a benchmark-sized
    problem (e.g. softmax 4096x393216) yields tiny probe tensors instead of OOM.

    The cap is deterministic per value (min(d, SMALL)), so a feature dim that
    appears in BOTH get_inputs() and get_init_inputs() shrinks to the same size
    and the shapes still line up.
    """

    # randint omitted on purpose: its leading ints are low/high, not dims.
    _FNS = ("randn", "rand", "zeros", "ones", "empty", "full")

    def __init__(self, torch_mod):
        self.t = torch_mod
        self._orig = {}

    def _cap(self, size):
        return tuple(min(int(d), SMALL) for d in size)

    def __enter__(self):
        for name in self._FNS:
            orig = getattr(self.t, name)
            self._orig[name] = orig

            def make(orig=orig):
                def wrapped(*args, **kwargs):
                    kwargs["device"] = "cpu"  # force CPU: specs often pin 'cuda'
                    if args and isinstance(args[0], (tuple, list, self.t.Size)):
                        return orig(self._cap(args[0]), *args[1:], **kwargs)
                    # varargs form: leading ints are dims, stop at first non-int
                    dims, rest = [], []
                    for i, a in enumerate(args):
                        if isinstance(a, int) and not rest:
                            dims.append(min(a, SMALL))
                        else:
                            rest = list(args[i:])
                            break
                    return orig(*dims, *rest, **kwargs)
                return wrapped

            setattr(self.t, name, make())
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self.t, name, orig)


def _extract_python(problem: str) -> str:
    """Some specs (hand-written ones) prefix the code with a prose line like
    'Write a Triton kernel for ...'. Drop everything before the first code line."""
    lines = problem.splitlines()
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if s.startswith(("import ", "from ", "class ", "def ")):
            return "\n".join(lines[i:])
    return problem


def _load_reference(problem: str):
    """exec problem.txt and return (model, probe_inputs) on CPU, small.

    Returns (model_callable, list_of_input_tensors). Raises on malformed spec.
    """
    import torch

    ns: dict = {}
    exec(compile(_extract_python(problem), "<problem>", "exec"), ns)  # noqa: S102 (trusted local spec)
    Model = ns["Model"]
    get_inputs = ns["get_inputs"]
    get_init = ns.get("get_init_inputs", lambda: [])

    with _CapSizes(torch):
        init_args = get_init() or []
        model = Model(*init_args)
        inputs = get_inputs()

    model = model.to("cpu").eval()
    inputs = [x.to("cpu") if torch.is_tensor(x) else x for x in inputs]
    return model, inputs


def _primary(inputs):
    """Index of the first floating-point tensor input (the one we probe)."""
    import torch
    for i, x in enumerate(inputs):
        if torch.is_tensor(x) and x.is_floating_point() and x.numel() > 1:
            return i
    return 0


# --------------------------------------------------------------------------- #
# Detection layers                                                            #
# --------------------------------------------------------------------------- #

def _forward(model, inputs):
    import torch
    with torch.no_grad():
        return model(*[x.clone() if torch.is_tensor(x) else x for x in inputs])


def _as_tensors(out):
    """Flatten a reference output (tensor / tuple / namedtuple) to a list."""
    import torch
    if torch.is_tensor(out):
        return [out]
    if isinstance(out, (tuple, list)):
        return [o for o in out if torch.is_tensor(o)]
    return []


def d1_precision(inputs, out_tensors) -> dict:
    """Low-bit detection: any fp8 dtype present (int4/fp4 aren't native torch)."""
    import torch
    low = set()
    fp8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n)}
    for x in inputs:
        if torch.is_tensor(x) and x.dtype in fp8:
            low.add(str(x.dtype))
    for o in out_tensors:
        if o.dtype in fp8:
            low.add(str(o.dtype))
    return {"low_bit": bool(low), "low_bit_dtypes": sorted(low)}


def d2_selection(inputs, out_tensors, pidx) -> dict:
    """Selection detection. A selection op either returns indices (integer), is a
    permutation of its input (sort), or returns a strict subset (topk/pool).

    The naive 'output values ⊆ input values' test alone false-positives on
    identity-like elementwise ops (e.g. ReLU on all-positive input == identity).
    So we require: integer output OR a non-trivial permutation OR a
    cardinality-REDUCING subset. ReLU preserves cardinality and (on mixed signs)
    changes the multiset, so it is correctly excluded.
    """
    import torch
    integer_out = any(not o.is_floating_point() for o in out_tensors)

    is_perm = subset_reduce = False
    subset_frac = 0.0
    prim = inputs[pidx] if pidx < len(inputs) and torch.is_tensor(inputs[pidx]) else None
    if out_tensors and prim is not None and prim.is_floating_point():
        o = out_tensors[0].float().flatten()
        p = prim.float().flatten()
        if o.numel() == p.numel():
            multiset_equal = bool(torch.allclose(o.sort().values, p.sort().values, atol=1e-4))
            identity = bool(torch.allclose(o, p, atol=1e-4))
            is_perm = multiset_equal and not identity        # sort reorders; identity doesn't
        elif o.numel() < p.numel():
            in_vals = set(p.tolist())
            hits = sum(1 for v in o.tolist() if v in in_vals)
            subset_frac = hits / max(o.numel(), 1)
            subset_reduce = subset_frac >= 0.95

    is_select = integer_out or is_perm or subset_reduce
    return {"integer_output": integer_out, "is_permutation": is_perm,
            "subset_reduce": subset_reduce, "subset_frac": round(subset_frac, 3),
            "is_select": is_select}


def d3_gain(model, inputs, pidx) -> dict:
    """Gain probe: ‖Δoutput‖/ε at representative input coordinates.

    Near-zero gain somewhere (with non-zero elsewhere) => the op has a region
    where input errors are invisible at the output => compress class. Roughly
    uniform gain => preserve class. Also a big-perturbation tail test: slamming
    a small coordinate that barely moves the output is the compress signature.
    """
    import torch
    x = inputs[pidx]
    if not (torch.is_tensor(x) and x.is_floating_point()):
        return {"ok": False}
    flat = x.float().flatten()
    eps = 1e-2

    def out_of(vec):
        return torch.cat([t.float().flatten() for t in
                          _as_tensors(_forward(model, _swap(inputs, pidx, vec.view_as(x).to(x.dtype))))])

    def gain_at_value(i, setval):
        a = flat.clone(); a[i] = setval
        b = flat.clone(); b[i] = setval + eps
        return round(float((out_of(b) - out_of(a)).norm() / eps), 4)

    y0 = out_of(flat)
    order = flat.abs().argsort()
    coords = {"max_val": int(flat.argmax()), "min_val": int(flat.argmin()),
              "near_zero": int(order[0]), "median_abs": int(order[len(order) // 2])}
    gains = {}
    for name, i in coords.items():
        xp = flat.clone(); xp[i] += eps
        gains[name] = round(float((out_of(xp) - y0).norm() / eps), 4)

    # Extreme-region probe: drive a coordinate to large +/- values and measure
    # local gain THERE. This directly tests the note's "dead zone" — a saturating
    # op (sigmoid/tanh, ReLU's negative half) has ~0 gain in its extreme region;
    # a magnitude-preserving op (matmul/elementwise) keeps comparable gain. randn
    # probe points alone never reach saturation, so this is what catches sigmoid.
    i = coords["median_abs"]
    extreme = {"pos8": gain_at_value(i, 8.0), "neg8": gain_at_value(i, -8.0)}

    gvals = list(gains.values())
    live = max(gvals + list(extreme.values())) if gvals else 0.0
    floor = min(gvals + list(extreme.values())) if gvals else 0.0
    uniformity = round(min(gvals) / max(gvals), 3) if gvals and max(gvals) > 1e-9 else 0.0
    has_dead_region = live > 1e-6 and floor < 0.02 * live
    return {"ok": True, "gains": gains, "extreme_gains": extreme,
            "uniformity": uniformity, "has_dead_region": bool(has_dead_region)}


def _swap(inputs, idx, new):
    out = list(inputs)
    out[idx] = new
    return out


def d4_fingerprints(model, inputs, pidx, out_tensors) -> dict:
    """softmax (sum-to-1 + range), saturation sweep (flat output regions),
    discreteness (piecewise-constant jumps under increasing noise)."""
    import torch
    fp = {}

    # softmax / probability fingerprint: some axis sums to ~1 and all in [0,1]
    is_softmax = False
    if out_tensors:
        o = out_tensors[0].float()
        in01 = bool((o >= -1e-3).all() and (o <= 1 + 1e-3).all())
        if in01 and o.ndim >= 1:
            for ax in range(o.ndim):
                if torch.allclose(o.sum(dim=ax), torch.ones_like(o.sum(dim=ax)), atol=1e-2):
                    is_softmax = True
                    break
    fp["softmax_like"] = is_softmax

    # saturation sweep: sweep one coordinate across [-6, 6]; count flat steps
    flat_frac = 0.0
    x = inputs[pidx]
    if torch.is_tensor(x) and x.is_floating_point():
        xf = x.float().flatten()
        i = int(xf.abs().argsort()[len(xf) // 2])  # a median coordinate
        ys = []
        for v in torch.linspace(-6, 6, 25):
            xp = xf.clone(); xp[i] = v
            yp = torch.cat([t.float().flatten()
                            for t in _as_tensors(_forward(model, _swap(inputs, pidx, xp.view_as(x).to(x.dtype))))])
            ys.append(yp)
        diffs = [float((ys[k + 1] - ys[k]).norm()) for k in range(len(ys) - 1)]
        if diffs:
            scale = max(diffs) or 1.0
            flat_frac = round(sum(1 for d in diffs if d < 0.02 * scale) / len(diffs), 3)
    fp["saturation_flat_frac"] = flat_frac

    # discreteness: increasing common-mode noise; selection output is piecewise
    # constant (stays exactly 0 then jumps), continuous output grows smoothly.
    jumpiness = 0.0
    if torch.is_tensor(x) and x.is_floating_point():
        xf = x.float().flatten()
        g = torch.Generator().manual_seed(0)
        delta = torch.randn(xf.shape, generator=g)
        y0 = torch.cat([t.float().flatten() for t in _as_tensors(_forward(model, inputs))])
        steps = []
        for t in torch.linspace(0.0, 0.5, 12)[1:]:
            xp = xf + t * delta
            yp = torch.cat([t2.float().flatten()
                            for t2 in _as_tensors(_forward(model, _swap(inputs, pidx, xp.view_as(x).to(x.dtype))))])
            steps.append(float((yp - y0).norm()))
        # fraction of steps that are exactly flat (no change at all)
        if steps:
            jumpiness = round(sum(1 for s in steps if s < 1e-9) / len(steps), 3)
    fp["flat_under_noise_frac"] = jumpiness
    return fp


# --------------------------------------------------------------------------- #
# Decision                                                                     #
# --------------------------------------------------------------------------- #

def _decide(d1, d2, d3, d4) -> tuple[str, str, str]:
    """Return (op_class, precision, judge)."""
    precision = "low_bit" if d1["low_bit"] else "normal"

    if d2["is_select"]:
        op_class = "select"
    elif (d4.get("softmax_like")
          or d4.get("saturation_flat_frac", 0) >= 0.25
          or d3.get("has_dead_region")):
        op_class = "compress"
    else:
        op_class = "preserve"

    if precision == "low_bit":
        judge = "J3/abstain"
    elif op_class == "select":
        judge = "J2"
    elif op_class == "compress":
        judge = "J1+advdist"
    else:
        judge = "J1"
    return op_class, precision, judge


def classify_problem(problem: str) -> dict:
    """Classify a problem spec (the PyTorch reference). Pure CPU, no kernel."""
    model, inputs = _load_reference(problem)
    out = _forward(model, inputs)
    out_tensors = _as_tensors(out)
    pidx = _primary(inputs)

    d1 = d1_precision(inputs, out_tensors)
    d2 = d2_selection(inputs, out_tensors, pidx)
    try:
        d3 = d3_gain(model, inputs, pidx)
    except Exception as e:
        d3 = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        d4 = d4_fingerprints(model, inputs, pidx, out_tensors)
    except Exception as e:
        d4 = {"error": f"{type(e).__name__}: {e}"}

    op_class, precision, judge = _decide(d1, d2, d3, d4)
    return {
        "op_class": op_class,
        "precision": precision,
        "judge": judge,
        "signals": {"D1": d1, "D2": d2, "D3": d3, "D4": d4},
    }


def classify_entry(name: str, *, dataset_dir: Path | None = None) -> dict:
    base = (dataset_dir or dataset.DEFAULT_DATASET_DIR) / name
    problem = (base / "problem.txt").read_text()
    return classify_problem(problem)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", help="classify one entry; omit for all")
    ap.add_argument("--verbose", action="store_true", help="dump raw D1-D4 signals")
    args = ap.parse_args()

    names = [args.entry] if args.entry else list(dataset.iter_entries())
    for name in names:
        try:
            r = classify_entry(name)
        except Exception as e:
            print(f"  {name:<48s} ERROR {type(e).__name__}: {e}")
            continue
        print(f"  {name:<48s} {r['op_class']:<9s} {r['precision']:<8s} -> {r['judge']}")
        if args.verbose:
            print(json.dumps(r["signals"], indent=4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
