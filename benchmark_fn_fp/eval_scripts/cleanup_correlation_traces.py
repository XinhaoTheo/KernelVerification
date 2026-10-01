"""Move historical Opus traces and remove verified duplicate GLM archives.

One manifest stores every old/new path and original SHA-256. No raw payload is
rewritten, and absent historical API captures or timestamps are not fabricated.
The default is a dry-run; --apply applies or resumes the saved manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

try:
    from .normalize_glm_trials import file_hashes
except ImportError:
    from normalize_glm_trials import file_hashes

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "traces_opus5/migration_20260930_correlation_cleanup.json"
ALIASES = {"case_a": "case_36", "case_b": "case_37"}
TRIAL_ORDER = {"r1": 1, "r2": 2, "neutral1": 3}


def _canonical_path(value):
    return Path(*(ALIASES.get(part, part) for part in Path(value).parts)).as_posix()


def _write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Refusing existing migration temporary file: {temporary}")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _metadata(source, destination, imported_at):
    case, arm, old_trial = source.parts[-3:]
    new_trial = destination.name
    meta = {"schema_version": 2, "case": case, "arm": arm, "trial": new_trial,
            "trial_sequence": int(new_trial[1:]), "original_trial": old_trial,
            "original_trace_path": source.as_posix(), "source_path": source.as_posix(),
            "dataset": "correlation_pair", "provider": "anthropic", "model": "claude-opus-5",
            "status": "historical", "raw_api_capture": False,
            "capture_note": "Original files are byte-preserved. Historical full API requests/responses were not captured; importing does not reconstruct them.",
            "imported_at": imported_at}
    return meta


def build_plan(root):
    root = Path(root).resolve()
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    opus_base = root / "correlation_pair/traces"
    for source in sorted(p for p in opus_base.glob("case_*/*/*") if p.is_dir()):
        case, arm, trial = source.parts[-3:]
        if case not in ALIASES.values() or arm not in {"single_call", "solo", "debate"} or trial not in TRIAL_ORDER:
            raise ValueError(f"Unexpected historical Opus trial: {source}")
        dest = root / "traces_opus5" / case / arm / f"r{TRIAL_ORDER[trial]}"
        if dest.exists() or dest.is_symlink():
            raise FileExistsError(f"Refusing existing Opus destination: {dest}")
        hashes = file_hashes(source)
        if "trace_meta.json" in hashes:
            raise ValueError(f"Historical source already has metadata: {source}")
        relative = source.relative_to(root)
        meta = _metadata(relative, dest.relative_to(root), timestamp)
        usage_file = source / "usage.json"
        if usage_file.exists():
            usage = json.loads(usage_file.read_text())
            if usage.get("model") != "claude-opus-5":
                raise ValueError(f"Unexpected recorded Opus model: {source}")
            for key in ("max_tokens", "prompt_variant", "kernel_sha256", "problem_sha256",
                        "source_problem_sha256", "stop_reason"):
                if key in usage:
                    meta[key] = usage[key]
            meta["model_provenance"] = "Explicit model in archived usage.json; complete API payload was not saved."
        else:
            if "provider: anthropic" not in (source / "runner_stdout.txt").read_text():
                raise ValueError(f"Missing historical Anthropic provenance: {source}")
            meta["model_provenance"] = "Model inferred from historical correlation_pair runner/report; Anthropic provider recorded in runner_stdout.txt. No full API payload."
        rows.append({"action": "move", "source": relative.as_posix(),
                     "destination": dest.relative_to(root).as_posix(),
                     "original_sha256": hashes, "metadata": meta})
    mapped_opus = {root / row["source"] / name for row in rows for name in row["original_sha256"]}
    if mapped_opus != {p for p in opus_base.rglob("*") if p.is_file()}:
        raise ValueError("Unmapped files in historical Opus archive")

    sources = {}
    for case in ALIASES.values():
        for metadata_path in (root / "traces_glm" / case).glob("*/*/trace_meta.json"):
            meta = json.loads(metadata_path.read_text())
            path = _canonical_path(meta.get("source_path", ""))
            if path.startswith("correlation_pair/traces_fireworks/"):
                if path in sources:
                    raise ValueError(f"Duplicate canonical GLM source: {path}")
                sources[path] = metadata_path.parent
    old_manifest = json.loads((root / "traces_glm/migration_20260923T050809Z.json").read_text())
    historical_hashes = {_canonical_path(r["source"]): r["original_sha256"]
                         for r in old_manifest["runs"] if r["source"].startswith("correlation_pair/")}
    for source in sorted(p for p in (root / "correlation_pair/traces_fireworks").glob("case_*/*/*") if p.is_dir()):
        relative = source.relative_to(root).as_posix()
        hashes = file_hashes(source)
        dest = sources.get(relative)
        if dest is None or hashes != historical_hashes.get(relative):
            raise ValueError(f"GLM archive does not match its historical migration: {source}")
        canonical = file_hashes(dest)
        if any(canonical.get(name) != digest for name, digest in hashes.items()):
            raise ValueError(f"Canonical GLM copy differs: {dest}")
        rows.append({"action": "deduplicate", "source": relative,
                     "destination": dest.relative_to(root).as_posix(),
                     "original_sha256": hashes, "canonical_sha256": canonical})
    mapped_glm = {root / row["source"] / name for row in rows if row["action"] == "deduplicate"
                  for name in row["original_sha256"]}
    if mapped_glm != {p for p in (root / "correlation_pair/traces_fireworks").rglob("*") if p.is_file()}:
        raise ValueError("Unmapped files in historical GLM archive")
    if not rows:
        raise ValueError("No historical traces to clean up")
    return {"schema_version": 1, "migration": "correlation_trace_cleanup", "status": "planned",
            "benchmark_root": str(root), "created_at": timestamp,
            "trial_order_note": "Original r1/r2 retain their sequence; neutral1 is the separately labelled prompt ablation and becomes r3. Missing API timestamps are not reconstructed.",
            "runs": rows}


def _expected_move(row):
    expected = dict(row["original_sha256"])
    text = json.dumps(row["metadata"], indent=2, ensure_ascii=False) + "\n"
    expected["trace_meta.json"] = hashlib.sha256(text.encode()).hexdigest()
    return expected


def _validate(root, plan):
    if plan.get("benchmark_root") != str(root) or plan.get("migration") != "correlation_trace_cleanup":
        raise ValueError("Manifest belongs to a different benchmark or migration")
    for row in plan["runs"]:
        source, dest = root / row["source"], root / row["destination"]
        if any(Path(row[k]).is_absolute() or ".." in Path(row[k]).parts for k in ("source", "destination")):
            raise ValueError("Unsafe migration path")
        if row["action"] == "move":
            if source.exists():
                if dest.exists() or file_hashes(source) != row["original_sha256"]:
                    raise ValueError(f"Opus source/destination changed: {source}")
            elif not dest.is_dir() or file_hashes(dest) not in (row["original_sha256"], _expected_move(row)):
                raise ValueError(f"Cannot recover moved Opus trace: {dest}")
        elif row["action"] == "deduplicate":
            if file_hashes(dest) != row["canonical_sha256"]:
                raise ValueError(f"Canonical GLM trace changed: {dest}")
            existing = file_hashes(source) if source.exists() else {}
            if plan["status"] == "planned" and existing != row["original_sha256"]:
                raise ValueError(f"GLM archive changed: {source}")
            if any(row["original_sha256"].get(k) != v for k, v in existing.items()):
                raise ValueError(f"Unrecognized GLM archive artifact: {source}")
        else:
            raise ValueError("Unknown migration action")


def apply_plan(root, manifest_path):
    root = Path(root).resolve()
    plan = json.loads(manifest_path.read_text())
    _validate(root, plan)  # Check every duplicate before moving/deleting anything.
    if plan["status"] == "completed":
        return plan
    plan["status"] = "in_progress"
    _write_json(manifest_path, plan)
    for row in plan["runs"]:
        source, dest = root / row["source"], root / row["destination"]
        if row["action"] == "move":
            if source.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists() or dest.is_symlink():
                    raise FileExistsError(f"Refusing to replace trace: {dest}")
                source.rename(dest)
            if not (dest / "trace_meta.json").exists():
                _write_json(dest / "trace_meta.json", row["metadata"])
            if file_hashes(dest) != _expected_move(row):
                raise ValueError(f"Moved raw trace verification failed: {dest}")
        else:
            for name, digest in row["original_sha256"].items():
                original, canonical = source / name, dest / name
                if original.exists():
                    if hashlib.sha256(original.read_bytes()).hexdigest() != digest or hashlib.sha256(canonical.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Duplicate changed before deletion: {original}")
                    original.unlink()
            if file_hashes(dest) != row["canonical_sha256"]:
                raise ValueError(f"Canonical GLM trace changed during cleanup: {dest}")
    for old_root in (root / "correlation_pair/traces", root / "correlation_pair/traces_fireworks"):
        if old_root.exists():
            for directory in sorted((p for p in old_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                directory.rmdir()
            old_root.rmdir()
    plan.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    _write_json(manifest_path, plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = args.benchmark_dir / MANIFEST
    if not manifest_path.exists():
        plan = build_plan(args.benchmark_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(manifest_path, plan)
    else:
        plan = json.loads(manifest_path.read_text())
    if args.apply:
        plan = apply_plan(args.benchmark_dir, manifest_path)
    print(json.dumps({"status": plan["status"], "manifest": str(manifest_path),
                      "moved_trials": sum(r["action"] == "move" for r in plan["runs"]),
                      "deduplicated_trials": sum(r["action"] == "deduplicate" for r in plan["runs"]),
                      "original_files": sum(len(r["original_sha256"]) for r in plan["runs"])}, indent=2))


if __name__ == "__main__":
    main()
