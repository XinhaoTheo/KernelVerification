"""End-to-end check of the wired path: classify -> judge dispatch -> verdict.

For each red-team entry it runs the FULL routing the upgraded pipeline will use:
  1. verifier.classify decides the operator class (from the trusted reference),
  2. verifier.compare.judge() picks the judge for that class,
  3. the judge returns match / mismatch / abstain on benign vs adversarial inputs.

This proves the standalone-demo judges now live in compare.py and are selected
automatically by class — the "通电" step before recheck.py calls them.

Run: CUDA_VISIBLE_DEVICES="" uv run python tests/advprec_pipeline_test.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from verifier import compare
from verifier.classify import classify_entry

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "dataset"
N = 256


def load_kernel(entry: str):
    path = DS / entry / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"_k_{entry}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


def line(entry, op_class, label, verdict_dict):
    extra = {k: v for k, v in verdict_dict.items() if k in ("value_gap", "recall", "max_diff")}
    print(f"    {label:<26s} -> {verdict_dict['verdict']:<8s} ({verdict_dict['judge']})  {extra}")


# --- per-entry runners: build benign + adversarial, route through judge() ---

def run_topk():
    entry = "_advprec_topk_boundary"
    kf = load_kernel(entry)
    op = classify_entry(entry)["op_class"]
    print(f"\n  {entry}  (classify -> {op})")

    def go(label, scores):
        ref = torch.topk(scores, 8, dim=-1).indices
        cand = kf(scores, 8)
        line(entry, op, label, compare.judge(cand, ref, op_class=op, scores=scores))

    g = torch.Generator().manual_seed(0)
    go("benign (randn)", torch.randn(1, N, generator=g))
    adv = torch.randn(1, N, generator=torch.Generator().manual_seed(0))
    adv[0, 0:8] = torch.linspace(6.0, 5.3, 8)  # winners concentrated in block 0
    go("adversarial (concentrated)", adv)


def run_softmax():
    entry = "_advprec_softmax_tail"
    kf = load_kernel(entry)
    op = classify_entry(entry)["op_class"]
    print(f"\n  {entry}  (classify -> {op})")

    def go(label, x):
        ref = torch.softmax(x, dim=-1)
        line(entry, op, label, compare.judge(kf(x), ref, op_class=op))

    go("benign (randn)", torch.randn(1, N, generator=torch.Generator().manual_seed(0)))
    adv = torch.full((1, N), -30.0)
    adv[0, 0] = 8.5
    adv[0, 1:200] = 0.0  # heavy tail just past the cutoff
    go("adversarial (heavy tail)", adv)


def run_fp8():
    entry = "_advprec_softmax_fp8_tail"
    kf = load_kernel(entry)
    op = classify_entry(entry)  # full result: needs precision too
    op_class, precision = op["op_class"], op["precision"]
    eff = "low_bit" if precision == "low_bit" else op_class
    print(f"\n  {entry}  (classify -> {op_class}, {precision})")

    x = torch.full((1, N), -30.0)
    x[0, 0] = 7.0
    x[0, 1:200] = 0.0
    ref = torch.softmax(x, dim=-1)
    line(entry, eff, "peaked (FP8 output)", compare.judge(kf(x), ref, op_class=eff))


def main() -> int:
    print("Wired path: classify -> compare.judge -> verdict\n"
          "  expect: topk catches via J2, softmax catches via J1+advdist, FP8 abstains")
    run_topk()
    run_softmax()
    run_fp8()
    print("\nAll three routed and judged automatically by operator class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
