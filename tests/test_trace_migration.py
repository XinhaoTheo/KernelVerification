"""Migration preserves paid-run bytes and keeps its manifest separate from prose."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def _migration(tmp_path, monkeypatch):
    eval_dir = Path(__file__).resolve().parents[1] / "benchmark_fn_fp" / "eval_scripts"
    monkeypatch.syspath_prepend(str(eval_dir))
    spec = importlib.util.spec_from_file_location("trace_migration_under_test", eval_dir / "merge_glm_traces.py")
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    monkeypatch.setattr(migration, "BENCHMARK_DIR", tmp_path)
    return migration


def test_glm_migration_preserves_files_records_manifest_and_is_idempotent(tmp_path, monkeypatch):
    migration = _migration(tmp_path, monkeypatch)

    legacy = tmp_path / "traces_glm_fireworks" / "case_35" / "debate"
    legacy.mkdir(parents=True)
    (legacy / "run.json").write_bytes(b'{"history":[],"verdict":{"verdict":"trust"}}\n')
    (legacy / "transcript.md").write_text("# Original tool transcript\n\nMeasured result.\n")
    (legacy / "probes").mkdir()
    (legacy / "probes" / "t1_stdout.txt").write_bytes(b"binary-preservation-check\x00\r\n")

    pair = tmp_path / "correlation_pair" / "traces_fireworks" / "case_a" / "single_call" / "r1"
    pair.mkdir(parents=True)
    response = '{"verdict":"reject","confidence":0.8,"reason":"Measured budget violation."}'
    originals = {
        "system_prompt.txt": "Verify the supplied kernel.\n",
        "user_prompt.txt": "A synthetic frozen contract and kernel.\n",
        "response_text.json": response,
        "usage.json": json.dumps({"usage": {"input_tokens": 12, "output_tokens": 7}, "stop_reason": "stop"}),
        "raw_response.json": json.dumps({"choices": [{"message": {
            "content": response, "reasoning_content": "Original provider reasoning."}, "finish_reason": "stop"}]}),
    }
    for name, text in originals.items():
        (pair / name).write_text(text)

    def hashes(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob("*") if path.is_file()}

    before = {str(source.relative_to(tmp_path)): hashes(source) for source in (legacy, pair)}
    monkeypatch.setattr(sys, "argv", ["merge_glm_traces.py", "--apply"])
    migration.main()

    destinations = {
        str(legacy.relative_to(tmp_path)): tmp_path / "traces_glm" / "case_35" / "debate" / "legacy",
        str(pair.relative_to(tmp_path)): tmp_path / "traces_glm" / "case_a" / "single_call" / "r1",
    }
    for source, original_hashes in before.items():
        copied = hashes(destinations[source])
        assert all(copied[name] == digest for name, digest in original_hashes.items())
    assert not legacy.exists()
    assert not (tmp_path / "traces_glm_fireworks").exists()
    assert hashes(pair) == before[str(pair.relative_to(tmp_path))]

    single_dest = destinations[str(pair.relative_to(tmp_path))]
    transcript = (single_dest / "transcript.md").read_text()
    assert transcript.startswith("# Single-call trace\n")
    assert "## User prompt" in transcript
    assert "Original provider reasoning." in transcript
    assert "original_sha256" not in transcript
    assert "planned_runs" not in transcript

    manifests = list((tmp_path / "traces_glm").glob("migration_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["runs"]) == 2
    assert {row["source"]: row["original_sha256"] for row in manifest["runs"]} == before
    for row in manifest["runs"]:
        assert tmp_path / row["destination"] == destinations[row["source"]]

    stable = hashes(tmp_path)
    monkeypatch.setattr(sys, "argv", ["merge_glm_traces.py"])
    migration.main()
    assert hashes(tmp_path) == stable
    monkeypatch.setattr(sys, "argv", ["merge_glm_traces.py", "--apply"])
    migration.main()
    assert hashes(tmp_path) == stable


def test_provider_leaves_flatten_with_neutral_collision_trial_and_preserved_metadata(tmp_path, monkeypatch):
    migration = _migration(tmp_path, monkeypatch)
    old_manifest = tmp_path / "traces_glm" / "migration_previous.json"
    old_manifest.parent.mkdir()
    old_manifest.write_bytes(b'{"status":"completed","runs":[]}\n')
    old_manifest_bytes = old_manifest.read_bytes()
    sources = []
    for provider, model in migration.MODELS.items():
        source = tmp_path / "traces_glm" / provider / "case_01" / "solo" / "r1"
        source.mkdir(parents=True)
        (source / "run.json").write_text(json.dumps({"provider_specific_evidence": provider}))
        (source / "trace_meta.json").write_text(json.dumps({
            "case": "case_01", "arm": "solo", "trial": "r1", "provider": provider,
            "model": model, "dataset": "benchmark_fn_fp", "status": "completed",
        }))
        sources.append((source, migration.hashes(source)))
    monkeypatch.setattr(sys, "argv", ["merge_glm_traces.py", "--apply"])
    migration.main()
    destination_root = tmp_path / "traces_glm" / "case_01" / "solo"
    assert sorted(p.name for p in destination_root.iterdir()) == ["r1", "r1__2"]
    for provider in migration.MODELS:
        assert not (tmp_path / "traces_glm" / provider).exists()
    assert old_manifest.read_bytes() == old_manifest_bytes
    manifests = [p for p in old_manifest.parent.glob("migration_*.json") if p != old_manifest]
    manifest = json.loads(manifests[0].read_text())
    assert manifest["status"] == "completed"
    assert len(manifest["runs"]) == 2
    for row in manifest["runs"]:
        dest = tmp_path / row["destination"]
        after = migration.hashes(dest)
        assert all(after[row["preserved_paths"][name]] == digest
                   for name, digest in row["original_sha256"].items())
        meta = json.loads((dest / "trace_meta.json").read_text())
        assert meta["trial"] == dest.name
        assert meta["provider"] in migration.MODELS
    assert (destination_root / "r1__2" / "trace_meta.before_migration.json").exists()
    stable = migration.hashes(tmp_path)
    migration.main()
    assert migration.hashes(tmp_path) == stable


def test_flat_legacy_migrates_into_own_child_without_removing_existing_trials(tmp_path, monkeypatch):
    migration = _migration(tmp_path, monkeypatch)
    arm = tmp_path / "traces_glm" / "case_02" / "solo"
    arm.mkdir(parents=True)
    original = b'{"history":[],"verdict":{"verdict":"reject"}}\n'
    (arm / "run.json").write_bytes(original)
    (arm / "probes").mkdir()
    (arm / "probes" / "t1_stdout.txt").write_bytes(b"original probe\x00\n")
    newer = arm / "legacy"
    newer.mkdir()
    (newer / "run.json").write_bytes(b'{"existing":"do not replace"}')
    newer_hashes = migration.hashes(newer)
    monkeypatch.setattr(sys, "argv", ["merge_glm_traces.py", "--apply"])
    migration.main()
    assert not (arm / "run.json").exists()
    assert not (arm / "probes").exists()
    assert migration.hashes(newer) == newer_hashes
    destination = arm / "legacy__2"
    assert (destination / "run.json").read_bytes() == original
    assert (destination / "probes" / "t1_stdout.txt").read_bytes() == b"original probe\x00\n"
    assert not (destination / "legacy").exists()
    stable = migration.hashes(tmp_path)
    migration.main()
    assert migration.hashes(tmp_path) == stable
