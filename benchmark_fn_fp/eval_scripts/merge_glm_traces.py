"""Flatten GLM traces without replacing paid-run records.

Dry run by default. --apply verifies every original file before removing moved
source files. Pair archives and historical migration manifests remain in place.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from traces import BENCHMARK_DIR, single_call_readable_files, update_trace_metadata

MODELS = {"openrouter": "z-ai/glm-5.3-flash", "fireworks": "accounts/fireworks/models/glm-5p3"}
MARKERS = ("run.json", "usage.json", "trace_meta.json")


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def is_trace(directory):
    return directory.is_dir() and any((directory / name).exists() for name in MARKERS)


def metadata(directory):
    path = directory / "trace_meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def source_hashes(source):
    # A legacy flat arm can already contain newer trial leaves. They are not
    # part of that old run and must never be moved into its own legacy child.
    nested_trials = [p for p in source.iterdir() if p.name not in {"probes", "llm_calls"} and is_trace(p)]
    out = {}
    for path in sorted(source.rglob("*")):
        if any(path.is_relative_to(trial) for trial in nested_trials):
            continue
        if path.is_symlink():
            raise ValueError(f"Refusing a symlink in migration source: {path}")
        if path.is_file():
            out[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def plan(root):
    registry_path = root / "case_map.json"
    if registry_path.exists() and json.loads(registry_path.read_text()).get("case_details"):
        raise RuntimeError("This historical provider-layout migration is disabled after numeric case migration. "
                           "Use the existing traces_glm tree; re-importing pair archives would duplicate paid runs.")
    rows = []
    reserved = set()

    def add(source, provider, case, arm, trial, *, dataset="benchmark_fn_fp", move=True):
        meta = metadata(source)
        provider = meta.get("provider", provider)
        preferred = root / "traces_glm" / case / arm / trial
        dest = preferred
        suffix = 2
        while dest.exists() or dest in reserved:
            dest = preferred.with_name(f"{trial}__{suffix}")
            suffix += 1
        reserved.add(dest)
        rows.append(dict(source=source, dest=dest, provider=provider,
            model=meta.get("model", MODELS[provider]), case=case, arm=arm,
            trial=dest.name, dataset=meta.get("dataset", dataset), move=move,
            original_sha256=source_hashes(source)))

    # The immediately preceding layout separated endpoint directories.
    for provider in MODELS:
        base = root / "traces_glm" / provider
        for source in sorted(base.glob("case_*/*/*")):
            if is_trace(source):
                add(source, provider, source.parts[-3], source.parts[-2], source.name)

    # Original runs stored payload files directly in the arm directory.
    for base, provider in ((root / "traces_glm", "openrouter"),
                           (root / "traces_glm_fireworks", "fireworks")):
        for source in sorted(base.glob("case_*/*")):
            if is_trace(source):
                add(source, provider, source.parent.name, source.name, "legacy")

    for source in sorted((root / "correlation_pair" / "traces_fireworks").glob("case_*/*/*")):
        if not is_trace(source):
            continue
        relative = str(source.relative_to(root))
        # Existing imports may still be in provider leaves awaiting this move.
        imported = [row["source"] for row in rows
                    if metadata(row["source"]).get("source_path") == relative]
        imported += [p for p in (root / "traces_glm" / source.parts[-3] / source.parts[-2]).glob("*")
                     if is_trace(p) and metadata(p).get("source_path") == relative]
        if imported:
            before = source_hashes(source)
            for existing in imported:
                after = hashes(existing)
                assert all(after.get(k) == v for k, v in before.items()), f"Changed imported source: {source}"
            continue
        add(source, "fireworks", source.parts[-3], source.parts[-2], source.name,
            dataset="correlation_pair", move=False)
    return rows


def remove_original_files(source, original_hashes, root):
    # Never rmtree(source): for old flat runs, destination is source/legacy/.
    for name, digest in original_hashes.items():
        path = source / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"Source changed: {path}"
        path.unlink()
    directories = {source, *(p.parent for p in (source / name for name in original_hashes))}
    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        while directory != root and directory.is_relative_to(root):
            if not directory.exists() or any(directory.iterdir()):
                break
            directory.rmdir()
            directory = directory.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = BENCHMARK_DIR
    rows = plan(root)
    print(json.dumps({"planned_runs": len(rows), "sources": [str(r["source"].relative_to(root)) for r in rows]}, indent=2))
    if not args.apply or not rows:
        return
    manifest = []
    manifest_path = root / "traces_glm" / ("migration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    for row in rows:
        source, dest, before = row["source"], row["dest"], row["original_sha256"]
        dest.mkdir(parents=True, exist_ok=False)
        for name in before:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        assert hashes(dest) == before, f"Copy hash mismatch: {source}"
        preserved_paths = {name: name for name in before}
        old_meta = metadata(dest)
        if old_meta and old_meta.get("trial") != row["trial"]:
            backup = dest / "trace_meta.before_migration.json"
            suffix = 2
            while backup.exists():
                backup = dest / f"trace_meta.before_migration_{suffix}.json"
                suffix += 1
            shutil.copyfile(dest / "trace_meta.json", backup)
            preserved_paths["trace_meta.json"] = backup.name
            update_trace_metadata(dest, trial=row["trial"])
        elif not old_meta:
            update_trace_metadata(dest, **{k: row[k] for k in ("case", "arm", "trial", "provider", "model", "dataset")},
                schema_version=2, status="historical", source_path=str(source.relative_to(root)),
                model_provenance="inferred from legacy tree/profile" if row["move"] else "explicit request/usage",
                raw_api_capture=row["arm"] == "single_call",
                imported_at=datetime.now(timezone.utc).isoformat())
        if row["arm"] == "single_call":
            raw_path = dest / "raw_response.json"
            raw = json.loads(raw_path.read_text()) if raw_path.exists() else {}
            message = ((raw.get("choices") or [{}])[0].get("message") or {})
            extras = single_call_readable_files(system=(dest / "system_prompt.txt").read_text(),
                user=(dest / "user_prompt.txt").read_text(), response=(dest / "response_text.json").read_text(),
                thinking=message.get("reasoning_content") or "", usage=json.loads((dest / "usage.json").read_text()))
            for name, value in extras.items():
                if not (dest / name).exists():
                    (dest / name).write_text(value)
        after = hashes(dest)
        assert all(after[preserved_paths[k]] == v for k, v in before.items())
        manifest.append({"source": str(source.relative_to(root)), "destination": str(dest.relative_to(root)),
                         "original_sha256": before, "preserved_paths": preserved_paths, "moved": row["move"]})
        manifest_path.write_text(json.dumps({"schema_version": 2, "status": "in_progress", "planned_runs": len(rows), "runs": manifest}, indent=2) + "\n")
        if row["move"]:
            remove_original_files(source, before, root)
    manifest_path.write_text(json.dumps({"schema_version": 2, "status": "completed", "runs": manifest}, indent=2) + "\n")
    print(f"Verified {len(rows)} runs; manifest {manifest_path}")


if __name__ == "__main__":
    main()
