"""Rebuild benchmark_fn_fp/eval_cases/ as an answer-free, opaquely-named copy.

Two separate leaks have to be closed for a case to be fair:

1. File contents. Only problem.txt and kernel.py are copied; test.py holds a
   reference implementation and meta.json holds the answer key, so neither is
   carried over.

2. The case NAME. `fn7_liger_rmsnorm_eps_placement` states the group (FN means a
   real defect), and names the defect. The debate orchestrator puts `state.entry`
   -- the directory name -- into every agent prompt, and `list_artifact_files`
   exposes the directory, so a descriptive name is read by the verifier on every
   turn. Directories here are therefore `case_NN`, assigned in a shuffled order
   so the numbering does not group FN before FP either.

The name mapping lives in benchmark_fn_fp/case_map.json, OUTSIDE eval_cases, so
the scorer can resolve a verdict back to its answer key while nothing under
eval_cases reveals it.

Usage (from repo root):
    python benchmark_fn_fp/generation/generators/build_eval_cases.py
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO / "benchmark_fn_fp" / "triton"
EVAL_DIR = REPO / "benchmark_fn_fp" / "eval_cases"
MAP_PATH = REPO / "benchmark_fn_fp" / "case_map.json"

# Fixed so a rebuild produces the same assignment; a case keeps its id across
# runs and results stay comparable.
SHUFFLE_SEED = 20260906

COPIED_FILES = ("problem.txt", "kernel.py")

# Words that would give the answer away if they appeared in a file the verifier
# reads. "exceed"/"within" are excluded on purpose: they are ordinary contract
# vocabulary ("must not exceed 1e-4") and appear identically in both halves of a
# matched pair, so they carry no signal.
BANNED = re.compile(
    r"\b(bug|buggy|violat|defect|unmodified|injected|mutat|not a bug|"
    r"correct kernel|both are valid|false[ _-]?(positive|negative))\b",
    re.IGNORECASE,
)


def _strip_name_header(text: str, case_name: str) -> tuple[str, str | None]:
    """Drop the leading docstring that names the case.

    Every kernel.py opens with `\"\"\"Triton kernel under test: <case name>.\"\"\"`,
    which hands the verifier both the group (fn = a real defect) and, in most
    cases, the defect itself. Only a first-line docstring is removed, and only
    when it is the sole occurrence of the name; anything else is reported rather
    than silently rewritten, since a name buried in real code would mean the
    kernel needs editing, not filtering.
    """
    if case_name not in text:
        return text, None
    lines = text.splitlines(keepends=True)
    first = lines[0] if lines else ""
    is_one_line_docstring = first.startswith('"""') and first.rstrip().endswith('"""')
    if is_one_line_docstring and case_name in first and case_name not in "".join(lines[1:]):
        return "".join(lines[1:]).lstrip("\n"), None
    return text, f"{case_name}: case name appears outside the leading docstring; edit the source"


def main() -> int:
    cases = sorted(
        d.name for d in SOURCE_DIR.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )
    if not cases:
        print(f"no cases found under {SOURCE_DIR}", file=sys.stderr)
        return 1

    # Ids already handed out are kept: results from earlier runs are stored under
    # case ids, so re-shuffling on every rebuild would silently misalign them with
    # the cases they scored. Only genuinely new cases get new ids.
    existing: dict[str, str] = {}
    if MAP_PATH.exists():
        existing = json.loads(MAP_PATH.read_text()).get("cases", {})
    mapping = {cid: name for cid, name in existing.items() if name in set(cases)}
    dropped = {cid: name for cid, name in existing.items() if name not in set(cases)}
    for cid, name in dropped.items():
        print(f"case removed from the source set, id retired: {cid} was {name}")

    fresh = [n for n in cases if n not in set(mapping.values())]
    random.Random(SHUFFLE_SEED).shuffle(fresh)
    next_id = max((int(c.split("_")[1]) for c in mapping), default=0) + 1
    for name in fresh:
        mapping[f"case_{next_id:02d}"] = name
        next_id += 1
    mapping = dict(sorted(mapping.items()))

    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True)

    problems: dict[str, str] = {}
    failures: list[str] = []
    for case_id, name in mapping.items():
        out = EVAL_DIR / case_id
        out.mkdir()
        for filename in COPIED_FILES:
            src = SOURCE_DIR / name / filename
            if not src.exists():
                failures.append(f"{name}: missing {filename}")
                continue
            text, note = _strip_name_header(src.read_text(), name)
            if note:
                failures.append(note)
            (out / filename).write_text(text)
        # The stub meta.json carries the opaque id only -- never the real name.
        (out / "meta.json").write_text(
            json.dumps({"name": case_id, "status": "under_test", "passed": None}, indent=2) + "\n"
        )
        problems[case_id] = (out / "problem.txt").read_text() if (out / "problem.txt").exists() else ""

    MAP_PATH.write_text(json.dumps({
        "note": "Maps opaque eval_cases ids to their answer-key directory under benchmark_fn_fp/triton/. "
                "Read only by the scorer; never shown to a verifier.",
        "shuffle_seed": SHUFFLE_SEED,
        "cases": mapping,
    }, indent=2) + "\n")

    # --- leak checks -------------------------------------------------------
    for path in sorted(EVAL_DIR.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        for match in BANNED.finditer(text):
            failures.append(f"{path.relative_to(REPO)}: banned word {match.group(0)!r}")
        for real_name in cases:
            if real_name in text:
                failures.append(f"{path.relative_to(REPO)}: leaks real case name {real_name!r}")
        if re.search(r"\b(fn|fp)\d", path.name, re.IGNORECASE):
            failures.append(f"{path.relative_to(REPO)}: filename encodes the group")

    # A matched pair is only a fair test if its two halves are indistinguishable
    # from the kernel alone; if two cases share a kernel, that must stay true.
    kernels: dict[str, list[str]] = {}
    for case_id in mapping:
        kernel = EVAL_DIR / case_id / "kernel.py"
        if kernel.exists():
            kernels.setdefault(kernel.read_text(), []).append(case_id)
    for shared in (ids for ids in kernels.values() if len(ids) > 1):
        joined = ", ".join(shared)
        if len({problems[i] for i in shared}) != len(shared):
            failures.append(f"pair {joined}: identical kernel AND identical problem.txt")
        else:
            print(f"matched pair sharing one kernel: {joined}")

    print(f"\nwrote {len(mapping)} cases to {EVAL_DIR.relative_to(REPO)}")
    print(f"wrote mapping to {MAP_PATH.relative_to(REPO)}")
    if failures:
        print(f"\nLEAK CHECK FAILED ({len(failures)}):", file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    print("leak check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
