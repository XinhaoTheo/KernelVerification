"""Historical trace cleanup preserves raw evidence, including failed attempts."""
import json
from pathlib import Path

import pytest

from benchmark_fn_fp.eval_scripts import cleanup_correlation_traces as cleanup


def fixture(root):
    old = root / "correlation_pair/traces/case_36/single_call/neutral1"
    old.mkdir(parents=True)
    (old / "usage.json").write_text(json.dumps({"model": "claude-opus-5", "max_tokens": 32768,
        "prompt_variant": "neutral_contract", "problem_sha256": "actual-prompt-hash"}))
    (old / "response_text.json").write_bytes(b'{"verdict":"reject"}\n\n')
    fw = root / "correlation_pair/traces_fireworks/case_37/single_call/r1"
    fw.mkdir(parents=True)
    (fw / "raw_response.json").write_text('{"choices":[{"finish_reason":"length"}]}')
    canonical = root / "traces_glm/case_37/single_call/r1"
    canonical.mkdir(parents=True)
    (canonical / "raw_response.json").write_bytes((fw / "raw_response.json").read_bytes())
    (canonical / "trace_meta.json").write_text(json.dumps({
        "source_path": "correlation_pair/traces_fireworks/case_b/single_call/r1"}))
    (root / "traces_glm/migration_20260923T050809Z.json").write_text(json.dumps({"runs": [{
        "source": "correlation_pair/traces_fireworks/case_b/single_call/r1",
        "original_sha256": cleanup.file_hashes(fw)}]}))
    return old, fw, canonical


def plan(root):
    manifest = root / cleanup.MANIFEST
    manifest.parent.mkdir(parents=True)
    cleanup._write_json(manifest, cleanup.build_plan(root))
    return manifest


def test_moves_opus_and_removes_only_verified_glm_copies(tmp_path):
    old, fw, canonical = fixture(tmp_path)
    before = {p.relative_to(old): p.read_bytes() for p in old.iterdir()}
    glm_before = cleanup.file_hashes(canonical)
    manifest = plan(tmp_path)
    result = cleanup.apply_plan(tmp_path, manifest)
    assert result["status"] == "completed"
    assert not (tmp_path / "correlation_pair/traces").exists()
    assert not (tmp_path / "correlation_pair/traces_fireworks").exists()
    dest = tmp_path / "traces_opus5/case_36/single_call/r3"
    assert {name: (dest / name).read_bytes() for name in before} == before
    assert cleanup.file_hashes(canonical) == glm_before
    assert json.loads((canonical / "raw_response.json").read_text())["choices"][0]["finish_reason"] == "length"
    meta = json.loads((dest / "trace_meta.json").read_text())
    assert meta["original_trial"] == "neutral1" and meta["trial"] == "r3"
    assert meta["prompt_variant"] == "neutral_contract" and meta["problem_sha256"] == "actual-prompt-hash"
    assert not meta["raw_api_capture"] and "created_at" not in meta
    assert cleanup.apply_plan(tmp_path, manifest)["status"] == "completed"


def test_changed_glm_copy_stops_before_any_opus_move_or_deletion(tmp_path):
    old, fw, canonical = fixture(tmp_path)
    manifest = plan(tmp_path)
    (canonical / "raw_response.json").write_text("changed")
    with pytest.raises(ValueError, match="Canonical GLM trace changed"):
        cleanup.apply_plan(tmp_path, manifest)
    assert old.exists() and fw.exists()
    assert not (tmp_path / "traces_opus5/case_36").exists()


def test_interruption_after_rename_can_resume_without_duplicate_archives(tmp_path, monkeypatch):
    _, _, canonical = fixture(tmp_path)
    manifest = plan(tmp_path)
    original_write = cleanup._write_json

    def interrupted(path, value):
        if path.name == "trace_meta.json":
            raise OSError("simulated metadata write failure")
        original_write(path, value)

    monkeypatch.setattr(cleanup, "_write_json", interrupted)
    with pytest.raises(OSError, match="simulated"):
        cleanup.apply_plan(tmp_path, manifest)
    monkeypatch.setattr(cleanup, "_write_json", original_write)
    assert cleanup.apply_plan(tmp_path, manifest)["status"] == "completed"
    assert (canonical / "raw_response.json").exists()


def test_unmapped_archive_files_are_not_silently_discarded(tmp_path):
    old, _, _ = fixture(tmp_path)
    (old.parents[2] / "unmapped.txt").write_text("historical evidence")
    with pytest.raises(ValueError, match="Unmapped files"):
        cleanup.build_plan(tmp_path)
