"""Our own independent correctness recheck — a second opinion on each kernel.

KernelAgent generates its own test (self-judged: same LLM writes kernel + test,
and forces bf16). This module adds OUR test: we ask an LLM to write a fresh
torch.allclose test from the problem spec + the final kernel, then run it.

Design choices (see discussion):
- We let the LLM write the test (it sees kernel.py, so it knows the call
  signature — sidesteps the per-kernel signature problem).
- We keep the kernel's dtype (bf16, consistent with KernelAgent) so we test
  algorithmic correctness, not dtype-assertion noise.
- Result is recorded on the dataset entry: meta.json["recheck"] + a
  recheck_test.py (always) + recheck_error.txt (on failure), mirroring the
  existing status/error.txt convention.

Usage:
    uv run kv-recheck                 # recheck all entries not yet rechecked
    uv run kv-recheck elem_add        # recheck one (force re-run)
    uv run kv-recheck --list          # show recheck status of every entry
    uv run kv-recheck --all           # force re-run every entry
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import dataset, llm_client
from .gpu_pick import pin_freest_gpu

# bf16 tolerances (matches KernelAgent's convention; bf16 has ~7 mantissa bits).
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 2e-2

TEST_GEN_SYSTEM = """You write a single standalone Python correctness test for a
Triton GPU kernel. It is an INDEPENDENT second opinion — be faithful to the
reference, do not rubber-stamp.

You are given:
  1. PROBLEM SPEC: a PyTorch reference (Model + get_inputs + get_init_inputs).
  2. KERNEL CODE: a module exporting `kernel_function`.

Setup (shared by all cases):
  - `from kernel import kernel_function`
  - `from kverify_compare import compare_outputs`  (provided helper, always available)
  - Reconstruct the reference Model from the spec.
  - INFER the call signature from the KERNEL CODE (some take (x), some (a, b),
    some (x, dim)). Match it exactly.
  - Use the dtype the kernel expects. If the kernel asserts a dtype, build
    inputs AND compute the reference in that dtype (test the algorithm, not the
    guard).
  - Decide each case with `matches, max_diff, detail = compare_outputs(out, ref)`.
    NEVER invent your own tolerance — compare_outputs handles it.

You MUST run this FIXED battery of cases and print ONE line per case, exactly:
    CASE <name>: PASS
    CASE <name>: FAIL <short detail>
    CASE <name>: SKIP <why>     # only if the transform cannot apply to this kernel

The fixed case names (run them all, in this order):
  1. standard          — the spec's get_inputs() exactly. THE core correctness case.
  2. noncontig_stride2 — make a tensor input non-contiguous: allocate 2x the size
                         and slice [::2] (1D) or [:, ::2] (2D+), keeping the SAME
                         logical values; reference computed on the same values.
  3. noncontig_transpose — for a 2D+ input, pass a transposed (.t()/.mT) view
                         (SKIP if input is 1D).
  4. odd_size          — re-run standard but with the main size +1 (non-aligned),
                         if the spec exposes a size you can bump (else SKIP).
  5. empty             — a zero-element input of the right rank/dtype
                         (SKIP if shape is fixed by the spec and cannot be empty).

Rules:
  - Wrap EACH case in its own try/except; an exception in one case prints
    `CASE <name>: FAIL raised <ExcType>` and must NOT abort the others.
  - If the kernel deliberately ASSERTS a precondition a case violates (e.g.
    asserts contiguity), that is a defensive guard for that case:
    print `CASE <name>: SKIP kernel asserts <precondition>`.
  - For `standard`, an exception or mismatch is a real FAIL (not skippable).
  - End with `if __name__ == "__main__":` that runs all cases, then
    `sys.exit(0 if standard_passed else 1)` — exit code reflects ONLY the
    standard case (core correctness); adversarial results are reported via the
    CASE lines, they do NOT change the exit code.

Output ONLY the Python code in a single ```python fenced block. No prose.
"""


def generate_test(problem: str, kernel_code: str, *, rtol: float, atol: float) -> str:
    """Ask the LLM to write an allclose test for this kernel."""
    system = TEST_GEN_SYSTEM
    user = (
        "=== PROBLEM SPEC ===\n"
        f"{problem}\n\n"
        "=== KERNEL CODE (exports kernel_function) ===\n"
        f"{kernel_code}\n"
    )
    raw = llm_client.oneshot(system, user, max_tokens=4096)
    return _extract_code(raw)


def run_test(
    kernel_code: str,
    test_code: str,
    *,
    timeout_s: int = 60,
) -> tuple[bool, str, str]:
    """Write kernel.py + test in a temp dir and run it as a subprocess.

    Returns (passed, stdout, stderr). passed == (exit code 0).
    """
    with tempfile.TemporaryDirectory(prefix="recheck_") as tmp:
        d = Path(tmp)
        (d / "kernel.py").write_text(kernel_code)
        # Drop the fixed comparison helper alongside so the generated test /
        # probe can `from kverify_compare import compare_outputs` instead of
        # improvising its own (badly-calibrated) tolerance.
        shutil.copy(Path(__file__).parent / "compare.py", d / "kverify_compare.py")
        (d / "recheck_test.py").write_text(test_code)
        try:
            proc = subprocess.run(
                [sys.executable, "recheck_test.py"],
                cwd=str(d),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            return False, e.stdout or "", f"TIMEOUT after {timeout_s}s\n{e.stderr or ''}"
        return proc.returncode == 0, proc.stdout, proc.stderr


def recheck_entry(
    name: str,
    *,
    dataset_dir: Path | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Generate + run our test for one entry, record results into the entry."""
    base = (dataset_dir or dataset.DEFAULT_DATASET_DIR) / name
    meta_path = base / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"dataset entry not found: {base}")

    meta = json.loads(meta_path.read_text())
    problem = _read(base / "problem.txt")
    kernel_code = _read(base / "kernel.py")

    if not kernel_code.strip():
        result = {"status": "error", "reason": "empty kernel.py"}
        _write_recheck(base, meta, result, test_code="", stdout="", stderr="")
        return result

    test_code = generate_test(problem, kernel_code, rtol=rtol, atol=atol)
    passed, stdout, stderr = run_test(kernel_code, test_code)
    cases = _parse_cases(stdout)

    # status reflects ONLY the standard case (core correctness). If the test
    # didn't emit a parseable `standard` line, fall back to the exit code.
    std = cases.get("standard")
    if std is not None:
        status = "passed" if std == "pass" else "failed"
    else:
        status = "passed" if passed else "failed"

    # robustness = the adversarial cases (everything except `standard`).
    robustness = {k: v for k, v in cases.items() if k != "standard"}

    result = {
        "status": status,
        "robustness": robustness,
        "rtol": rtol,
        "atol": atol,
        "model": llm_client._DEFAULT_MODEL,
        "stdout_len": len(stdout),
        "stderr_len": len(stderr),
    }
    _write_recheck(base, meta, result, test_code=test_code, stdout=stdout, stderr=stderr)
    return result


_CASE_RE = re.compile(r"^CASE\s+(\w+):\s*(PASS|FAIL|SKIP)", re.MULTILINE)


def _parse_cases(stdout: str) -> dict[str, str]:
    """Parse `CASE <name>: PASS|FAIL|SKIP` lines into {name: pass|fail|skip}."""
    out: dict[str, str] = {}
    for name, status in _CASE_RE.findall(stdout):
        out[name] = status.lower()
    return out


def get_recheck(
    name: str,
    *,
    dataset_dir: Path | None = None,
    force: bool = False,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> dict:
    """Return recheck info for an entry, running it if absent (or forced).

    This is the entry point the verification path (kv-run) calls: it makes
    kv-run self-contained — given any kernel, we generate + run OUR test rather
    than trusting whatever the upstream generator may or may not have provided.

    Returns {status, rtol, atol, test_code, error_text}.
    """
    base = (dataset_dir or dataset.DEFAULT_DATASET_DIR) / name
    if not (base / "meta.json").exists():
        raise FileNotFoundError(f"dataset entry not found: {base}")

    meta = json.loads((base / "meta.json").read_text())
    if force or "recheck" not in meta:
        recheck_entry(name, dataset_dir=dataset_dir, rtol=rtol, atol=atol)
        meta = json.loads((base / "meta.json").read_text())

    rc = meta.get("recheck", {})
    return {
        "status": rc.get("status", "unknown"),
        "robustness": rc.get("robustness", {}),
        "rtol": rc.get("rtol"),
        "atol": rc.get("atol"),
        "test_code": _read(base / "recheck_test.py"),
        "output_text": _read(base / "recheck_output.txt"),
        "error_text": _read(base / "recheck_error.txt"),
    }


def iter_untested(*, dataset_dir: Path | None = None):
    """Yield entry names that have no recheck result yet."""
    for name in dataset.iter_entries(dataset_dir=dataset_dir):
        base = (dataset_dir or dataset.DEFAULT_DATASET_DIR) / name
        meta = json.loads((base / "meta.json").read_text())
        if "recheck" not in meta:
            yield name


def _write_recheck(base: Path, meta: dict, result: dict, *, test_code, stdout, stderr):
    if test_code:
        (base / "recheck_test.py").write_text(test_code)
    # Always save the run output (pass or fail) for observability.
    (base / "recheck_output.txt").write_text(
        f"=== RECHECK STATUS: {result['status']} ===\n\n"
        f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n"
    )
    if result["status"] != "passed" and (stdout or stderr):
        (base / "recheck_error.txt").write_text(
            f"=== RECHECK STATUS: {result['status']} ===\n\n"
            f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n"
        )
    meta["recheck"] = result
    (base / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


def _read(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def main() -> int:
    # Pin a usable GPU before the test subprocess (which loads torch) runs.
    # Done here, not at import, so importing recheck as a library (e.g. from the
    # verifier agent) has no heavy side effect.
    pin_freest_gpu()

    ap = argparse.ArgumentParser()
    ap.add_argument("entry", nargs="?", help="recheck one entry (force). Omit to do all untested.")
    ap.add_argument("--all", action="store_true", help="force re-run every entry")
    ap.add_argument("--list", action="store_true", help="show recheck status of all entries")
    ap.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    args = ap.parse_args()

    if args.list:
        for name in dataset.iter_entries():
            meta = json.loads((dataset.DEFAULT_DATASET_DIR / name / "meta.json").read_text())
            rc = meta.get("recheck", {})
            print(f"  {name:<50s} recheck={rc.get('status', 'untested')}")
        return 0

    if args.entry:
        names = [args.entry]
    elif args.all:
        names = list(dataset.iter_entries())
    else:
        names = list(iter_untested())

    if not names:
        print("nothing to recheck (all entries already have a recheck result)")
        return 0

    print(f"rechecking {len(names)} entries (rtol={args.rtol}, atol={args.atol})")
    summary: dict[str, int] = {}
    for name in names:
        print(f"\n=== {name} ===")
        try:
            result = recheck_entry(name, rtol=args.rtol, atol=args.atol)
        except Exception as e:
            print(f"  RECHECK ERROR: {type(e).__name__}: {e}")
            summary["error"] = summary.get("error", 0) + 1
            continue
        print(f"  recheck status = {result['status']}")
        summary[result["status"]] = summary.get(result["status"], 0) + 1

    print("\n=== SUMMARY ===")
    for status, count in sorted(summary.items()):
        print(f"  {status:<12s} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
