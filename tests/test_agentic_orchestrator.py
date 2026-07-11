from __future__ import annotations

import json
from pathlib import Path

from verifier.agentic.orchestrator import AgenticOrchestrator, build_context_response
from verifier.agentic.state import Role
from verifier.agentic_run import main as agentic_main


def test_orchestrator_applies_dry_run_agent_response_and_persists(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_artifact(dataset_root)
    run_dir = tmp_path / "run"

    orchestrator = AgenticOrchestrator(dataset_dir=dataset_root, run_dir=run_dir)
    outputs = orchestrator.apply_agent_response(
        role=Role.ORCHESTRATOR,
        response=build_context_response("toy"),
    )
    persisted = orchestrator.persist()

    assert [item["tool"] for item in outputs] == [
        "load_artifact",
        "inspect_problem",
        "inspect_kernel_source",
    ]
    assert orchestrator.state.entry == "toy"
    assert len(orchestrator.state.history) == 1
    assert len(orchestrator.state.tool_events) == 3
    assert persisted.run_json.exists()
    assert persisted.tool_events_jsonl.exists()
    assert persisted.claims_json.exists()

    run_data = json.loads(persisted.run_json.read_text())
    assert run_data["entry"] == "toy"
    assert run_data["history"][0]["tool_calls"][0]["tool"] == "load_artifact"
    assert "adversarial-precision.md" in run_data["skills"]
    assert "metric-selection.md" in run_data["skills"]
    assert "scope-policy.md" in run_data["skills"]


def test_agentic_run_dry_run_cli_writes_run_json(tmp_path, capsys) -> None:
    dataset_root = tmp_path / "dataset"
    _write_artifact(dataset_root)
    run_dir = tmp_path / "cli-run"

    exit_code = agentic_main([
        "toy",
        "--dataset-dir",
        str(dataset_root),
        "--run-dir",
        str(run_dir),
        "--dry-run",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "tools_executed: 3" in captured.out
    assert (run_dir / "run.json").exists()
    assert (run_dir / "tool_events.jsonl").exists()


def _write_artifact(dataset_root: Path) -> None:
    entry_dir = dataset_root / "toy"
    entry_dir.mkdir(parents=True)
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": "toy", "passed": True, "status": "passed", "rounds": 1})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (entry_dir / "test.py").write_text("def test():\n    pass\n")
