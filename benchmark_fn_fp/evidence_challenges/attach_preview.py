"""Expose real initial measurements to every arm, before the source freeze."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MARKER = "\n\nRecorded initial-probe execution on NVIDIA T4 (not an oracle verdict):\n"


def main(path):
    source = Path(path).resolve()
    relative_source = str(source.relative_to(ROOT))
    preview = json.loads(source.read_text())
    if preview.get("failures"):
        raise ValueError("Do not publish a failed preview")
    frozen = json.loads((ROOT / "private_data" / "validation_gpu.json").read_text()).get("cases", {}) if (
        ROOT / "private_data" / "validation_gpu.json").exists() else {}
    keys = {p: json.loads(p.read_text()) for p in (ROOT / "private_data").glob("answer_key_*.json")}
    changes = []
    for name, row in preview["cases"].items():
        if name in frozen or (ROOT.parent / "traces_glm" / name).exists():
            raise ValueError(f"Cannot amend frozen or evaluated case: {name}")
        path = ROOT / "eval_cases" / name / "problem.txt"
        old = path.read_text()
        if MARKER in old:
            raise ValueError(f"Initial results already attached: {name}")
        if hashlib.sha256(old.encode()).hexdigest() != row["problem_sha256"]:
            raise ValueError(f"Preview source changed: {name}")
        text = old.rstrip() + MARKER + json.dumps(row["initial_probe"], indent=2) + "\n"
        if len(text) > 12000:
            raise ValueError(f"Contract exceeds existing prompt limit: {name}")
        owners = [p for p, key in keys.items() if name in key["cases"]]
        if len(owners) != 1:
            raise ValueError(f"Ambiguous CPU key: {name}")
        changes.append((name, path, text, owners[0], row["problem_sha256"]))
    provenance = []
    for name, path, text, owner, before in changes:
        path.write_text(text)
        after = hashlib.sha256(text.encode()).hexdigest()
        keys[owner]["cases"][name]["problem_sha256"] = after
        provenance.append({"case": name, "preview": relative_source,
                           "before_problem_sha256": before, "after_problem_sha256": after,
                           "label_unchanged": True})
    for owner in {change[3] for change in changes}:
        owner.write_text(json.dumps(keys[owner], indent=2) + "\n")
    log = ROOT / "private_data" / "validation_attempts" / f"{source.stem}_public_attachment.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Published real initial measurements for {len(changes)} cases; final GPU freeze still required")


if __name__ == "__main__":
    main(sys.argv[1])
