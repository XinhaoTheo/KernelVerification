"""Number each GLM case/arm's trials without rewriting historical payloads.

Dry-run writes one reviewable manifest. --apply executes that exact snapshot;
the same command resumes an interrupted two-phase rename. Only trace_meta.json
changes. Its original text and every original file hash remain in the manifest.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import uuid

BENCHMARK = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = Path("/private/tmp/kv-normalize-trials-plan.json")
ARMS = {"single_call", "solo", "debate"}
META = "trace_meta.json"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hashes(directory):
    """Inventory regular files; refuse aliases before moving any trace."""
    if directory.is_symlink():
        raise ValueError(f"Symlink in trace inventory: {directory}")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in trace inventory: {path}")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = _digest(path.read_bytes())
        elif not path.is_dir():
            raise ValueError(f"Non-regular trace artifact: {path}")
    return result


def _directories(root):
    base = root / "traces_glm"
    if not base.is_dir() or base.is_symlink():
        raise ValueError(f"Expected a real traces_glm directory: {base}")
    result = []
    for case in sorted(base.glob("case_*")):
        if not re.fullmatch(r"case_\d+", case.name) or not case.is_dir() or case.is_symlink():
            raise ValueError(f"Unexpected case directory: {case}")
        for arm in sorted(case.iterdir()):
            if not arm.is_dir() or arm.is_symlink() or arm.name not in ARMS:
                raise ValueError(f"Unexpected arm entry: {arm}")
            for trial in sorted(arm.iterdir()):
                if not trial.is_dir() or trial.is_symlink():
                    raise ValueError(f"Unexpected trial entry: {trial}")
                result.append(trial)
    return result


def _sort_key(meta, path):
    key = meta.get("selection_sort_key", [meta.get("created_at") or "", path])
    if not isinstance(key, list) or len(key) != 2 or not all(isinstance(v, str) for v in key):
        raise ValueError(f"Invalid selection_sort_key for {path}")
    return key


def _metadata_text(row):
    metadata = json.loads(row["original_metadata_text"])
    metadata.update(row["metadata_updates"])
    return json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"


def build_plan(root):
    root = Path(root).resolve()
    grouped = defaultdict(list)
    for directory in _directories(root):
        original = (directory / META).read_bytes().decode("utf-8")
        meta = json.loads(original)
        relative = directory.relative_to(root).as_posix()
        grouped[directory.parent].append((directory, meta, original, _sort_key(meta, relative)))
    runs = []
    for parent, trials in sorted(grouped.items(), key=lambda item: (
            int(item[0].parent.name.removeprefix("case_")), item[0].name)):
        for sequence, (source, meta, original, key) in enumerate(sorted(trials, key=lambda item: (
                item[3], item[0].name)), 1):
            trial = f"r{sequence}"
            relative = source.relative_to(root).as_posix()
            updates = {
                "trial": trial,
                "trial_sequence": sequence,
                "original_trial": meta.get("original_trial", meta.get("trial", source.name)),
                "original_trace_path": meta.get("original_trace_path", relative),
                "selection_sort_key": key,
            }
            row = {
                "source": relative,
                "destination": (parent / trial).relative_to(root).as_posix(),
                "case": parent.parent.name,
                "arm": parent.name,
                "trial": trial,
                "original_sha256": file_hashes(source),
                "original_metadata_text": original,
                "metadata_updates": updates,
            }
            row["metadata_changed"] = original != _metadata_text(row)
            runs.append(row)
    return {
        "schema_version": 1,
        "migration": "glm_numeric_trials",
        "status": "planned",
        "benchmark_root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "planned_runs": len(runs),
        "renamed_runs": sum(row["source"] != row["destination"] for row in runs),
        "metadata_updates": sum(row["metadata_changed"] for row in runs),
        "ordering": "Preserved selection_sort_key, otherwise created_at (missing first), then original path; no verdict filtering.",
        "runs": runs,
    }


def _write_plan(path, plan):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"Refusing symlink manifest: {path}")
    # A killed process can leave a temporary file behind. Unique, exclusive
    # names let the last durable phase resume without overwriting that file.
    temporary = path.with_name(path.name + f".tmp_{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _path(root, relative):
    path = Path(relative)
    if path.is_absolute() or len(path.parts) != 4 or path.parts[0] != "traces_glm" or ".." in path.parts:
        raise ValueError(f"Invalid manifest trace path: {relative}")
    if not re.fullmatch(r"case_\d+", path.parts[1]) or path.parts[2] not in ARMS:
        raise ValueError(f"Invalid manifest trace identity: {relative}")
    result = root / path
    if any(p.is_symlink() for p in (result, result.parent, result.parent.parent, root / "traces_glm")):
        raise ValueError(f"Symlink in manifest trace path: {relative}")
    return result


def _expected(row, *, updated):
    result = dict(row["original_sha256"])
    if updated:
        result[META] = _digest(_metadata_text(row).encode())
    return result


def _check(directory, row, *, allow_updated=False):
    actual = file_hashes(directory)
    allowed = [_expected(row, updated=False)]
    if allow_updated:
        allowed.append(_expected(row, updated=True))
    if actual not in allowed:
        raise ValueError(f"Trace changed since reviewed manifest: {directory}")


def _move(source, destination):
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite trace destination: {destination}")
    source.rename(destination)


def _validate(root, plan):
    if plan.get("migration") != "glm_numeric_trials" or plan.get("benchmark_root") != str(root):
        raise ValueError("Manifest belongs to a different migration or benchmark root")
    sources, destinations = set(), set()
    for row in plan["runs"]:
        source, destination = _path(root, row["source"]), _path(root, row["destination"])
        if source.parent != destination.parent or not re.fullmatch(r"r[1-9]\d*", destination.name):
            raise ValueError("Trial normalization cannot change a case/arm or use a nonnumeric trial")
        if source in sources or destination in destinations:
            raise ValueError("Duplicate source or destination in manifest")
        sources.add(source)
        destinations.add(destination)
        old = json.loads(row["original_metadata_text"])
        updates = row["metadata_updates"]
        if updates != {"trial": destination.name, "trial_sequence": int(destination.name[1:]),
                       "original_trial": old.get("original_trial", old.get("trial", source.name)),
                       "original_trace_path": old.get("original_trace_path", row["source"]),
                       "selection_sort_key": _sort_key(old, row["source"])}:
            raise ValueError("Manifest metadata updates differ from the allowed provenance fields")
        if row["original_sha256"].get(META) != _digest(row["original_metadata_text"].encode()):
            raise ValueError("Original metadata text does not match its saved hash")
    return sources, destinations


def apply_plan(root, plan_path):
    """Apply or resume the reviewed manifest, checking all payloads at each phase."""
    root, plan_path = Path(root).resolve(), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    sources, destinations = _validate(root, plan)
    rows = plan["runs"]
    if plan["status"] == "completed":
        if set(_directories(root)) != destinations:
            raise ValueError("Completed manifest no longer matches trial directories")
        for row in rows:
            if file_hashes(root / row["destination"]) != _expected(row, updated=True):
                raise ValueError(f"Completed trace changed: {row['destination']}")
        return plan
    if plan["status"] == "planned":
        if set(_directories(root)) != sources:
            raise ValueError("Trial inventory changed since reviewed manifest")
        for row in rows:
            _check(root / row["source"], row)
        for dest in destinations - sources:
            if dest.exists() or dest.is_symlink():
                raise FileExistsError(f"Unplanned destination exists: {dest}")
        staging_name = f"traces_glm/.normalize_trials_{uuid.uuid4().hex}"
        if (root / staging_name).exists() or (root / staging_name).is_symlink():
            raise FileExistsError("Migration staging path already exists")
        plan.update(status="in_progress", phase="staging", staging_dir=staging_name)
        _write_plan(plan_path, plan)
    elif plan["status"] != "in_progress":
        raise ValueError(f"Cannot apply manifest status: {plan['status']}")

    staging_relative = plan.get("staging_dir", "")
    if not re.fullmatch(r"traces_glm/\.normalize_trials_[0-9a-f]{32}", staging_relative):
        raise ValueError("Invalid migration staging path")
    staging = root / staging_relative
    if staging.is_symlink():
        raise ValueError("Migration staging path is a symlink")
    staging.mkdir(exist_ok=True)
    staged = [staging / f"{i:06d}" for i in range(len(rows))]
    allowed_staging = {p.name for p in staged}
    if plan["phase"] == "metadata":
        allowed_staging.update(f".metadata_{i:06d}.tmp" for i in range(len(rows)))
    if any(p.is_symlink() or p.name not in allowed_staging for p in staging.iterdir()):
        raise ValueError("Unexpected artifacts in migration staging directory")
    # Phase boundaries are persisted before advancing. A restart can determine
    # each row's location from its private staging slot, even for r1/r2 cycles.
    if plan["phase"] == "staging":
        expected_live = {root / row["source"] for row, stage in zip(rows, staged) if not stage.exists()}
        if set(_directories(root)) != expected_live:
            raise ValueError("Unexpected trace directories during staging")
        for row, stage in zip(rows, staged):
            _check(stage if stage.exists() else root / row["source"], row)
        for row, stage in zip(rows, staged):
            if not stage.exists():
                _move(root / row["source"], stage)
        plan["phase"] = "placing"
        _write_plan(plan_path, plan)
    if plan["phase"] == "placing":
        expected_live = {root / row["destination"] for row, stage in zip(rows, staged) if not stage.exists()}
        if set(_directories(root)) != expected_live:
            raise ValueError("Unexpected trace directories during placement")
        for row, stage in zip(rows, staged):
            _check(stage if stage.exists() else root / row["destination"], row)
        for row, stage in zip(rows, staged):
            if stage.exists():
                _move(stage, root / row["destination"])
        plan["phase"] = "metadata"
        _write_plan(plan_path, plan)
    if plan["phase"] != "metadata":
        raise ValueError(f"Unknown migration phase: {plan['phase']}")
    if set(_directories(root)) != destinations:
        raise ValueError("Unexpected trace directories before metadata update")
    for row in rows:
        _check(root / row["destination"], row, allow_updated=True)
    for i, row in enumerate(rows):
        dest = root / row["destination"] / META
        updated = _metadata_text(row)
        if dest.read_text() != updated:
            temporary = staging / f".metadata_{i:06d}.tmp"
            if temporary.is_symlink():
                raise ValueError("Metadata temporary path is a symlink")
            temporary.write_text(updated)
            temporary.replace(dest)
    for row in rows:
        if file_hashes(root / row["destination"]) != _expected(row, updated=True):
            raise ValueError(f"Final payload verification failed: {row['destination']}")
    staging.rmdir()  # Never recursively delete unexpected artifacts.
    plan.update(status="completed", phase="completed", completed_at=datetime.now(timezone.utc).isoformat())
    _write_plan(plan_path, plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=BENCHMARK)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--apply", action="store_true", help="apply/resume the existing reviewed plan")
    args = parser.parse_args(argv)
    if args.apply:
        plan = apply_plan(args.benchmark_dir, args.plan)
    else:
        if args.plan.exists():
            previous = json.loads(args.plan.read_text())
            if previous.get("status") == "in_progress":
                raise ValueError("An interrupted migration must be resumed with --apply, not replanned")
            if previous.get("status") == "completed":
                raise ValueError("Keep the completed audit manifest; choose a different --plan for a new preview")
        plan = build_plan(args.benchmark_dir)
        _write_plan(args.plan, plan)
    print(json.dumps({"status": plan["status"], "planned_runs": plan["planned_runs"],
                      "renamed_runs": plan["renamed_runs"], "metadata_updates": plan["metadata_updates"],
                      "manifest": str(args.plan)}, indent=2))


if __name__ == "__main__":
    main()
