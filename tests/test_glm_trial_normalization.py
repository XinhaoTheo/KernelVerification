"""Offline checks for a resumable, byte-preserving GLM trial rename."""
import json
from pathlib import Path

import pytest

from benchmark_fn_fp.eval_scripts import normalize_glm_trials as migration


def add_trial(root, name, *, created_at=None, case="case_01", arm="solo", verdict="trust"):
    path = root / "traces_glm" / case / arm / name
    path.mkdir(parents=True)
    meta = {"trial": name, "case": case, "arm": arm, "status": "completed",
            "model": "historical-model", "provider": "historical-provider"}
    if created_at is not None:
        meta["created_at"] = created_at
    (path / "trace_meta.json").write_text(json.dumps(meta, indent=4) + "\n")
    (path / "run.json").write_text(json.dumps({"trial": name, "verdict": verdict}))
    (path / "transcript.md").write_bytes(b"Old raw transcript\n\n  \xff")
    (path / "llm_calls" / "raw").mkdir(parents=True)
    (path / "llm_calls" / "raw" / "response.json").write_bytes(b'{"raw": "unchanged"}\n')
    return path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def save_plan(root, tmp_path):
    plan = migration.build_plan(root)
    path = tmp_path / "plan.json"
    migration._write_plan(path, plan)
    return path, plan


def test_dry_run_maps_every_trial_and_changes_no_payload(tmp_path, capsys):
    root = tmp_path / "benchmark"
    add_trial(root, "r1", created_at="2026-09-30")
    add_trial(root, "legacy", verdict="reject")
    add_trial(root, "earlier", created_at="2026-09-01")
    before = snapshot(root)
    plan_path = tmp_path / "plan.json"
    migration.main(["--benchmark-dir", str(root), "--plan", str(plan_path)])
    plan = json.loads(plan_path.read_text())
    assert snapshot(root) == before
    assert [r["source"].split("/")[-1] for r in plan["runs"]] == ["legacy", "earlier", "r1"]
    assert [r["trial"] for r in plan["runs"]] == ["r1", "r2", "r3"]
    assert plan["status"] == "planned"
    assert '"planned_runs": 3' in capsys.readouterr().out


def test_existing_r1_collision_preserves_raw_and_metadata_provenance(tmp_path):
    root = tmp_path / "benchmark"
    add_trial(root, "r1", created_at="2026-09-30")
    add_trial(root, "legacy")
    plan_path, plan = save_plan(root, tmp_path)
    before = snapshot(root)
    result = migration.apply_plan(root, plan_path)
    assert result["status"] == "completed"
    assert not list((root / "traces_glm").glob(".normalize*"))
    for row in plan["runs"]:
        for name in row["original_sha256"]:
            if name != "trace_meta.json":
                assert (root / row["destination"] / name).read_bytes() == before[row["source"] + "/" + name]
        meta = json.loads((root / row["destination"] / "trace_meta.json").read_text())
        assert meta["trial"] == row["trial"]
        assert meta["original_trial"] == Path(row["source"]).name
        assert meta["original_trace_path"] == row["source"]
        assert meta["selection_sort_key"][1] == row["source"]
        assert meta["model"] == "historical-model"
        assert json.loads(row["original_metadata_text"])["trial"] == meta["original_trial"]
    after = snapshot(root)
    migration.apply_plan(root, plan_path)
    assert snapshot(root) == after
    repeat = migration.build_plan(root)
    assert repeat["renamed_runs"] == repeat["metadata_updates"] == 0
    with pytest.raises(ValueError, match="completed audit"):
        migration.main(["--benchmark-dir", str(root), "--plan", str(plan_path)])


def test_equal_dates_keep_original_order_even_across_r10(tmp_path):
    root = tmp_path / "benchmark"
    for i in range(12):
        add_trial(root, f"old_{i:02d}")
    path, plan = save_plan(root, tmp_path)
    migration.apply_plan(root, path)
    meta = [json.loads((root / r["destination"] / "trace_meta.json").read_text()) for r in plan["runs"]]
    assert [m["trial"] for m in sorted(meta, key=lambda m: m["selection_sort_key"])] == [f"r{i}" for i in range(1, 13)]
    assert [r["trial"] for r in migration.build_plan(root)["runs"]] == [f"r{i}" for i in range(1, 13)]


def test_original_metadata_bytes_with_crlf_are_kept_in_manifest(tmp_path):
    root = tmp_path / "benchmark"
    trial = add_trial(root, "legacy")
    meta_path = trial / "trace_meta.json"
    original = meta_path.read_bytes().replace(b"\n", b"\r\n")
    meta_path.write_bytes(original)
    path, plan = save_plan(root, tmp_path)
    assert plan["runs"][0]["original_metadata_text"].encode() == original
    migration.apply_plan(root, path)


@pytest.mark.parametrize("change", ["payload", "metadata", "extra_trial"])
def test_stale_plan_is_rejected_before_any_move(tmp_path, change):
    root = tmp_path / "benchmark"
    trial = add_trial(root, "legacy")
    path, _ = save_plan(root, tmp_path)
    if change == "extra_trial":
        add_trial(root, "new")
    else:
        (trial / ("run.json" if change == "payload" else "trace_meta.json")).write_text("{}")
    before = snapshot(root)
    with pytest.raises(ValueError, match="changed"):
        migration.apply_plan(root, path)
    assert snapshot(root) == before
    assert json.loads(path.read_text())["status"] == "planned"


@pytest.mark.parametrize("fail_at", [2, 4])
def test_interrupted_rename_resumes_without_collision_or_raw_changes(tmp_path, monkeypatch, fail_at):
    root = tmp_path / "benchmark"
    add_trial(root, "r1", created_at="2026-09-30")
    add_trial(root, "legacy")
    path, plan = save_plan(root, tmp_path)
    move = migration._move
    calls = 0

    def interrupted(source, dest):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise OSError("simulated interrupted migration")
        move(source, dest)

    monkeypatch.setattr(migration, "_move", interrupted)
    with pytest.raises(OSError, match="simulated"):
        migration.apply_plan(root, path)
    assert json.loads(path.read_text())["status"] == "in_progress"
    with pytest.raises(ValueError, match="resumed"):
        migration.main(["--benchmark-dir", str(root), "--plan", str(path)])
    monkeypatch.setattr(migration, "_move", move)
    migration.apply_plan(root, path)
    for row in plan["runs"]:
        actual = migration.file_hashes(root / row["destination"])
        assert {k: v for k, v in actual.items() if k != "trace_meta.json"} == {
            k: v for k, v in row["original_sha256"].items() if k != "trace_meta.json"}


def test_refuses_symlink_payload_and_plan_path_escape(tmp_path):
    root = tmp_path / "benchmark"
    trial = add_trial(root, "legacy")
    target = tmp_path / "outside"
    target.write_text("unrelated")
    (trial / "alias").symlink_to(target)
    with pytest.raises(ValueError, match="Symlink"):
        migration.build_plan(root)
    (trial / "alias").unlink()
    path, plan = save_plan(root, tmp_path)
    plan["runs"][0]["destination"] = "../../outside"
    migration._write_plan(path, plan)
    before = snapshot(root)
    with pytest.raises(ValueError, match="Invalid manifest"):
        migration.apply_plan(root, path)
    assert snapshot(root) == before


def test_interrupted_metadata_write_resumes_and_keeps_original_audit(tmp_path, monkeypatch):
    root = tmp_path / "benchmark"
    add_trial(root, "legacy")
    add_trial(root, "new", created_at="2026-09-30")
    path, plan = save_plan(root, tmp_path)
    replace = Path.replace

    def interrupted(source, dest):
        if source.name == ".metadata_000001.tmp":
            raise OSError("simulated metadata interruption")
        return replace(source, dest)

    monkeypatch.setattr(Path, "replace", interrupted)
    with pytest.raises(OSError, match="metadata interruption"):
        migration.apply_plan(root, path)
    assert json.loads(path.read_text())["phase"] == "metadata"
    monkeypatch.setattr(Path, "replace", replace)
    completed = migration.apply_plan(root, path)
    assert completed["status"] == "completed"
    assert [r["original_metadata_text"] for r in completed["runs"]] == [r["original_metadata_text"] for r in plan["runs"]]
