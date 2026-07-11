from __future__ import annotations

import json
from pathlib import Path

from verifier.agentic.state import RunState, ToolStatus
from verifier.agentic.tools.registry import ToolContext, build_core_registry


def test_run_python_probe_captures_stdout_json_and_artifacts(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    context = ToolContext(state=state, run_dir=tmp_path / "run")

    result = registry.call(
        "run_python_probe",
        {
            "code": "import json\nprint('probe started')\nprint(json.dumps({'verdict': 'match', 'value': 3}))\n",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )

    assert result["event_id"] == "t1"
    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert result["json_result"] == {"verdict": "match", "value": 3}
    assert result["json_parse_error"] is None
    assert state.tool_events[0].status == ToolStatus.OK
    assert {artifact["kind"] for artifact in result["artifacts"]} == {
        "probe_code",
        "stdout",
        "stderr",
        "json_result",
    }
    assert (tmp_path / "run" / "probes" / "t1_probe.py").exists()
    assert (tmp_path / "run" / "probes" / "t1_stdout.txt").read_text().startswith("probe started")


def test_run_python_probe_can_import_loaded_artifact_kernel(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")

    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state, dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")

    registry.call("load_artifact", {"entry": "toy"}, context=context)
    result = registry.call(
        "run_python_probe",
        {
            "code": "import json\nimport kernel\nprint(json.dumps({'value': kernel.kernel(4)}))\n",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )

    assert result["event_id"] == "t2"
    assert result["json_result"] == {"value": 5}
    assert Path(result["cwd"]).name == "toy"
    assert [event.tool for event in state.tool_events] == ["load_artifact", "run_python_probe"]


def test_run_python_probe_timeout_returns_structured_result(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()

    result = registry.call(
        "run_python_probe",
        {
            "code": "import time\nprint('before sleep', flush=True)\ntime.sleep(5)\n",
            "timeout_s": 1,
            "use_gpu": False,
        },
        context=ToolContext(state=state, run_dir=tmp_path / "run"),
    )

    assert result["event_id"] == "t1"
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert "before sleep" in result["stdout"]
    assert state.tool_events[0].status == ToolStatus.OK


def test_run_claim_probe_returns_claim_bound_evidence_draft(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    context = ToolContext(state=state, run_dir=tmp_path / "run")
    claim = registry.call(
        "record_claim",
        {"statement": "Probe should observe value 3.", "rationale": "The hypothesis needs runtime evidence."},
        context=context,
    )

    result = registry.call(
        "run_claim_probe",
        {
            "claim_id": claim["id"],
            "code": "import json\nprint(json.dumps({'observed': 3}))\n",
            "expected_signal": "observed should equal 3",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )

    assert result["event_id"] == "t2"
    assert result["claim_id"] == "c1"
    assert result["evidence_draft"]["tool_event_id"] == "t2"
    assert result["evidence_draft"]["supports"] == "needs_interpretation"
    assert result["evidence_draft"]["data"]["json_result"] == {"observed": 3}
    assert [event.tool for event in state.tool_events] == ["record_claim", "run_claim_probe"]


def test_finalize_probe_evidence_consumes_claim_probe_and_updates_claim(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    context = ToolContext(state=state, run_dir=tmp_path / "run")
    claim = registry.call(
        "record_claim",
        {"statement": "Probe should observe value 3.", "rationale": "The hypothesis needs runtime evidence."},
        context=context,
    )
    probe = registry.call(
        "run_claim_probe",
        {
            "claim_id": claim["id"],
            "code": "import json\nprint(json.dumps({'observed': 3}))\n",
            "expected_signal": "observed should equal 3",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )

    result = registry.call(
        "finalize_probe_evidence",
        {
            "event_id": probe["event_id"],
            "supports": "confirmed",
            "summary": "The probe observed the expected value 3.",
        },
        context=context,
    )

    assert result["claim"]["status"] == "confirmed"
    assert result["evidence"]["tool_event_id"] == "t2"
    assert result["evidence"]["kind"] == "runtime_probe"
    assert result["evidence"]["data"]["json_result"] == {"observed": 3}
    assert [event.tool for event in state.tool_events] == [
        "record_claim",
        "run_claim_probe",
        "finalize_probe_evidence",
    ]


def _write_artifact(dataset_root: Path) -> None:
    entry_dir = dataset_root / "toy"
    entry_dir.mkdir(parents=True)
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": "toy", "passed": True, "status": "passed", "rounds": 1})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (entry_dir / "test.py").write_text("def test():\n    pass\n")
