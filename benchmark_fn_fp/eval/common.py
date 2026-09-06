"""Shared helpers for the three-way FN/FP benchmark evaluation.

Every baseline emits the SAME verdict vocabulary as the debate system's Judge
(verifier/agentic/tools/verdict.py) so the three are directly comparable:

    trust                -> "this kernel is correct"
    reject               -> "this kernel has a bug"
    needs_more_evidence  -> "inconclusive"

Ground truth comes from each case's meta.json `expected.correct_verdict`:
a BUGGY case must be `reject`, a CORRECT case must be `trust`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# What a verifier is allowed to see: answer-free problem.txt + kernel.py only,
# under opaque `case_NN` directory names. The descriptive names live only in the
# answer key: `fn7_liger_rmsnorm_eps_placement` states both the group (fn = a
# real defect) and the defect, and the debate orchestrator puts the directory
# name into every agent prompt as `state.entry`.
CASES_DIR = Path(__file__).resolve().parent.parent / "eval_cases"
# The answer key, read only by the scorer, never fed to any verifier.
ANSWER_KEY_DIR = Path(__file__).resolve().parent.parent / "triton"
# case_NN -> answer-key directory. Lives outside eval_cases on purpose.
CASE_MAP_PATH = Path(__file__).resolve().parent.parent / "case_map.json"

TRUST = "trust"
REJECT = "reject"
INCONCLUSIVE = "needs_more_evidence"
# A verifier that produced no answer at all. Distinct from INCONCLUSIVE,
# which is a judgement the system actually made.
NO_VERDICT = "no_verdict"


@dataclass(frozen=True)
class Case:
    name: str           # the opaque id the verifier sees, e.g. "case_07"
    real_name: str      # the answer-key directory; never shown to a verifier
    group: str          # "FN" or "FP"
    seed_class: str     # "FN1".."FP6"
    kernel_family: str
    reference: str
    ground_truth: str   # TRUST or REJECT
    problem_txt: str
    kernel_py: str
    path: Path


def load_cases(cases_dir: Path | None = None) -> list[Case]:
    """Inputs come from the answer-free copy; labels from the answer key."""
    root = cases_dir or CASES_DIR
    case_map = json.loads(CASE_MAP_PATH.read_text())["cases"]
    cases: list[Case] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "meta.json").exists():
            continue
        real_name = case_map.get(d.name)
        if real_name is None:
            raise KeyError(f"{d.name} is not in {CASE_MAP_PATH}; rerun build_eval_cases.py")
        key_path = ANSWER_KEY_DIR / real_name / "meta.json"
        if not key_path.exists():
            raise FileNotFoundError(f"no answer key for {d.name} at {key_path}")
        meta = json.loads(key_path.read_text())
        verdict_text = str(meta["expected"]["correct_verdict"]).upper()
        if verdict_text.startswith("BUGGY"):
            ground_truth = REJECT
        elif verdict_text.startswith("CORRECT"):
            ground_truth = TRUST
        else:
            raise ValueError(f"{d.name}: unrecognized correct_verdict {verdict_text!r}")
        cases.append(
            Case(
                name=d.name,
                real_name=real_name,
                group=meta["group"],
                seed_class=meta["seed_class"],
                kernel_family=meta["kernel_family"],
                reference=meta["reference"],
                ground_truth=ground_truth,
                problem_txt=(d / "problem.txt").read_text(),
                kernel_py=(d / "kernel.py").read_text(),
                path=d,
            )
        )
    return cases


def score(results: dict[str, str], cases: list[Case]) -> dict:
    """results: case name -> verdict. Returns accuracy plus FN/FP error counts."""
    by_name = {c.name: c for c in cases}
    correct = wrong = inconclusive = no_answer = 0
    missed_bugs: list[str] = []      # ground truth REJECT, baseline said trust
    false_alarms: list[str] = []     # ground truth TRUST, baseline said reject
    for name, verdict in results.items():
        gt = by_name[name].ground_truth
        if verdict == NO_VERDICT:
            no_answer += 1
        elif verdict == INCONCLUSIVE:
            inconclusive += 1
        elif verdict == gt:
            correct += 1
        else:
            wrong += 1
            if gt == REJECT:
                missed_bugs.append(name)
            else:
                false_alarms.append(name)
    return {
        "n": len(results),
        "correct": correct,
        "wrong": wrong,
        "inconclusive": inconclusive,
        "no_answer": no_answer,
        "missed_bugs": missed_bugs,
        "false_alarms": false_alarms,
    }


def print_report(title: str, results: dict[str, str], cases: list[Case]) -> None:
    by_name = {c.name: c for c in cases}
    print(f"\n=== {title} ===")
    for name in sorted(results):
        c = by_name[name]
        verdict = results[name]
        if verdict == c.ground_truth:
            mark = "OK "
        elif verdict == NO_VERDICT:
            mark = "-- "
        elif verdict == INCONCLUSIVE:
            mark = "?? "
        else:
            mark = "XX "
        print(f"  {mark} {c.group} {c.seed_class:4s} {name:8s} {c.real_name:44s} "
              f"gt={c.ground_truth:6s} got={verdict}")
    s = score(results, cases)
    print(f"  -> {s['correct']}/{s['n']} correct, {s['wrong']} wrong, "
          f"{s['inconclusive']} inconclusive, {s['no_answer']} produced no verdict")
    if s["missed_bugs"]:
        print(f"     missed bugs (said trust, really buggy): {len(s['missed_bugs'])} {s['missed_bugs']}")
    if s["false_alarms"]:
        print(f"     false alarms (said reject, really fine): {len(s['false_alarms'])} {s['false_alarms']}")
