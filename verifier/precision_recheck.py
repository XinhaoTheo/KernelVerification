"""Precision-aware recheck — the new verification dimension from
precision_verification.md, wired into the real pipeline.

It is the "先判算子哪类 → 选裁判 → 判不了就 abstain" path made concrete:
  1. verifier.classify decides the operator class (from the trusted reference),
  2. we feed a class-appropriate ADVERSARIAL distribution (pillar X) — not just
     benign randn, which is the送分 input that hides these bugs,
  3. verifier.compare.judge picks the judge for that class (J1 / J2 / abstain),
  4. result is a 3-state verdict: match / mismatch / abstain.

This runs alongside the existing recheck battery (standard correctness + shape
robustness in recheck.py). It is recorded under meta.json["precision_recheck"].

Unlike recheck.py's LLM-written allclose test, this battery is DETERMINISTIC:
the adversarial distributions and the judge are fixed code, so the verdict is
reproducible run to run (addresses README §7 limitation #1).

Usage:
    uv run kv-precision-recheck                 # all entries
    uv run kv-precision-recheck _advprec_topk_boundary
    uv run kv-precision-recheck --list
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from . import classify, compare, dataset

WORK_N = 256  # working last-dim size: big enough to expose block / tail bugs


def _load_kernel_fn(base: Path):
    path = base / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"_pr_{base.name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


def _device():
    """Run on GPU when one is visible (real kernels are Triton/CUDA); fall back
    to CPU (the red-team kernels are pure-torch and run either way).

    Pin a FREE GPU first (a card can be visible but busy) — same as recheck.py.
    Must run before any CUDA context is created, so callers should not touch
    torch.cuda before classification (which is CPU-only) and this call.
    """
    import os

    import torch
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"  # caller forced CPU (e.g. the CPU-only regression test)
    try:
        from .gpu_pick import pin_freest_gpu
        pin_freest_gpu()
    except Exception:
        pass
    return "cuda" if torch.cuda.is_available() else "cpu"


def _benign(rows: int, n: int, dtype, device):
    import torch
    return torch.randn(rows, n, device=device).to(dtype)


def _adversarial(op_class: str, rows: int, n: int, dtype, device):
    """Class-appropriate distribution that exposes the class-specific blind spot
    (precision_verification.md §5.1). Benign randn hides these; this is pillar X."""
    import torch
    if op_class == "select":
        # winners concentrated in one block (attention sink / recent tokens):
        # defeats per-block-quota and other "winners are spread" assumptions.
        s = torch.randn(rows, n, device=device)
        s[:, 0:8] = torch.linspace(6.0, 5.3, 8, device=device)
        return s.to(dtype)
    if op_class == "compress":
        # one peak + a heavy band carrying real aggregate tail mass: defeats
        # "the tail is negligible" assumptions (truncation / cheap tail exp).
        s = torch.full((rows, n), -30.0, device=device)
        s[:, 0] = 8.5
        s[:, 1:200] = 0.0
        return s.to(dtype)
    return torch.randn(rows, n, device=device).to(dtype)  # preserve: unused


def _primary_meta(problem: str):
    """Learn the primary input's rank/dtype + the (already-built) reference model
    by loading it on a small capped probe (classify reuses this)."""
    model, inputs = classify._load_reference(problem)
    pidx = classify._primary(inputs)
    x = inputs[pidx]
    return model, x.ndim, x.dtype


def _ref_out(model, x):
    import torch
    with torch.no_grad():
        out = model(x)
    if isinstance(out, (tuple, list)):  # e.g. topk -> (values, indices)
        ints = [o for o in out if torch.is_tensor(o) and not o.is_floating_point()]
        return ints[0] if ints else out[0]
    return out


def _verdict_on(model, kf, op_class, x):
    """Run reference + kernel on x and route through compare.judge."""
    ref = _ref_out(model, x)
    cand = kf(x)
    scores = x if op_class == "select" else None
    return compare.judge(cand, ref, op_class=op_class, scores=scores)


def precision_recheck(name: str, *, dataset_dir: Path | None = None) -> dict:
    base = (dataset_dir or dataset.DEFAULT_DATASET_DIR) / name
    problem = (base / "problem.txt").read_text()

    cls = classify.classify_problem(problem)
    op_class, precision = cls["op_class"], cls["precision"]

    import torch

    # D — low-bit: numerical comparison is structurally unreliable. Abstain
    # BEFORE running anything: a confident wrong PASS/FAIL is worse than "I can't".
    if precision == "low_bit":
        return {
            "op_class": op_class, "precision": precision,
            "verdict": "abstain", "judge": "J3/abstain",
            "reason": "low-bit output (fp8/fp4/int4): sub-resolution errors round "
                      "to 0, so no tolerance/distribution can separate correct "
                      "from buggy. Route to a downstream / task-level check.",
        }

    # A — preserve: numerical error is a faithful proxy here, so there is no
    # compression/selection blind spot to probe. The standard benign recheck
    # (recheck.py) already covers it; skip the adversarial battery.
    if op_class == "preserve":
        return {"op_class": op_class, "precision": precision, "verdict": "skipped",
                "reason": "preserve class: no compression/selection blind spot; "
                          "covered by the standard recheck battery (J1 on benign)."}

    device = _device()
    model, _ndim, dtype = _primary_meta(problem)
    model = model.to(device)
    rows = 4
    if dtype not in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
        dtype = torch.float32
    kf = _load_kernel_fn(base)

    try:
        benign = _verdict_on(model, kf, op_class, _benign(rows, WORK_N, dtype, device))
        adv = _verdict_on(model, kf, op_class, _adversarial(op_class, rows, WORK_N, dtype, device))
    except Exception as e:
        return {"op_class": op_class, "precision": precision, "verdict": "error",
                "reason": f"could not run kernel on device={device}: "
                          f"{type(e).__name__}: {e}"}

    # Overall: a kernel must be right on BOTH benign and adversarial. The whole
    # point is that benign alone passes (false accept) — the adversarial verdict
    # is what catches the class-specific bug.
    if benign["verdict"] == "mismatch":
        overall = "mismatch"          # broken even on benign
    elif adv["verdict"] == "mismatch":
        overall = "mismatch"          # caught by the adversarial distribution
    else:
        overall = "match"
    return {"op_class": op_class, "precision": precision, "verdict": overall,
            "judge": adv.get("judge"), "benign": benign, "adversarial": adv}


def to_claim(result: dict) -> dict | None:
    """Turn a precision_recheck result into a PRE-FILED debate claim (design B),
    so the judge weighs precision's empirical finding alongside the skeptic's
    instead of re-discovering it. Returns None when there is nothing to add
    (match / skipped / error). The claim is pre-resolved, so the verifier (which
    only probes status=="open") leaves it untouched.
    """
    v = result.get("verdict")
    op = result.get("op_class")
    judge = result.get("judge")
    if v == "mismatch":
        adv = result.get("adversarial") or {}
        ev = {k: adv[k] for k in ("value_gap", "max_diff", "recall") if k in adv}
        return {
            "id": "p1",
            "type": f"precision_{op}",
            "statement": (
                f"precision_recheck empirically caught a {op}-class defect on its "
                f"adversarial distribution ({judge}): {ev}. Benign inputs hide it; "
                "the divergence is real on the class-specific stress distribution."
            ),
            "raised_by": "precision_recheck",
            "round": -1,
            "status": "confirmed",
            "evidence": [{"by": "precision_recheck",
                          "verdict_line": f"adversarial {judge} -> mismatch {ev}"}],
        }
    if v == "abstain":
        return {
            "id": "p1",
            "type": f"precision_{op}_abstain",
            "statement": (
                f"precision_recheck ABSTAINS: {result.get('reason')} Numerical "
                "comparison cannot decide correctness here; it needs a downstream / "
                "task-level check rather than a PASS."
            ),
            "raised_by": "precision_recheck",
            "round": -1,
            "status": "inconclusive",
            "evidence": [{"by": "precision_recheck",
                          "verdict_line": "abstain (low-bit; error is sub-resolution)"}],
        }
    return None


def _record(base: Path, result: dict):
    meta = json.loads((base / "meta.json").read_text())
    meta["precision_recheck"] = result
    (base / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", help="one entry; omit for all")
    ap.add_argument("--list", action="store_true", help="show stored precision verdicts")
    ap.add_argument("--no-save", action="store_true", help="don't write meta.json")
    args = ap.parse_args()

    if args.list:
        for name in dataset.iter_entries():
            meta = json.loads((dataset.DEFAULT_DATASET_DIR / name / "meta.json").read_text())
            pr = meta.get("precision_recheck", {})
            print(f"  {name:<46s} {pr.get('op_class','?'):<9s} "
                  f"{pr.get('precision','?'):<8s} -> {pr.get('verdict','untested')}")
        return 0

    names = [args.entry] if args.entry else list(dataset.iter_entries())
    for name in names:
        try:
            r = precision_recheck(name)
        except Exception as e:
            print(f"  {name:<46s} ERROR {type(e).__name__}: {e}")
            continue
        det = ""
        if "adversarial" in r:
            a = r["adversarial"]
            det = f"  adv={{{', '.join(f'{k}={v}' for k,v in a.items() if k in ('value_gap','max_diff','recall'))}}}"
        print(f"  {name:<46s} {r['op_class']:<9s} {r['precision']:<8s} "
              f"-> {r['verdict']:<9s} ({r.get('judge')}){det}")
        if not args.no_save:
            _record(dataset.DEFAULT_DATASET_DIR / name, r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
