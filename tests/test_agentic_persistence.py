from __future__ import annotations

import json

from verifier.agentic.orchestrator import AgenticOrchestrator, build_context_response
from verifier.agentic.protocol import parse_agent_response
from verifier.agentic.persistence import load_run_state
from verifier.agentic.state import Role, RunState
from verifier.agentic.tools.registry import ToolContext, build_core_registry


def test_load_run_state_round_trip(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    orchestrator = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")
    orchestrator.apply_agent_response(role=Role.ORCHESTRATOR, response=build_context_response("toy"))
    orchestrator.apply_agent_response(
        role=Role.SKEPTIC,
        response=parse_agent_response(
            json.dumps(
                {
                    "message": "Record a claim.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Boundary sizes may be mishandled.",
                                "rationale": "No boundary evidence is present yet.",
                            },
                        }
                    ],
                }
            )
        ),
    )
    orchestrator.state.convergence = {
        "request": "more_debate",
        "reason": "Need one more critique round.",
        "focus_claims": ["c1"],
        "created_at": "2026-01-01T00:00:00Z",
    }
    orchestrator.state.skeptic_review = {
        "decision": "no_new_claims",
        "reason": "No additional in-scope claims remain.",
        "reviewed_claims": ["c1"],
        "reviewed_tool_event_count": 4,
        "turn": 2,
        "created_at": "2026-01-01T00:00:01Z",
    }
    persisted = orchestrator.persist(stop_reason="unit_test_stop")
    transcript = persisted.transcript_md.read_text()

    loaded = load_run_state(persisted.run_json)

    assert loaded.entry == "toy"
    assert len(loaded.history) == 2
    assert loaded.history[1].role == "skeptic"
    assert loaded.claims[0].id == "c1"
    assert loaded.tool_events[0].tool == "load_artifact"
    assert loaded.convergence["request"] == "more_debate"
    assert loaded.convergence["focus_claims"] == ["c1"]
    assert loaded.skeptic_review["decision"] == "no_new_claims"
    assert loaded.skeptic_review["reviewed_claims"] == ["c1"]
    assert persisted.transcript_md.exists()
    assert "# Agentic Verification Transcript" in transcript
    assert "Stop reason: `unit_test_stop`" in transcript
    assert "Skeptic review: `no_new_claims`" in transcript
    assert "## Timeline" in transcript
    assert "Turn 2 - `skeptic`" in transcript
    assert "## Claims" in transcript
    assert "c1 - `open`" in transcript
    assert "## Tool Events" in transcript


def test_description_model_persists_and_renders_in_transcript(tmp_path) -> None:
    run_dir = tmp_path / "run"
    registry = build_core_registry()
    state = RunState(entry="toy")
    context = ToolContext(state=state, run_dir=run_dir)

    request = registry.call(
        "request_description",
        {
            "reason_kind": "source_interpretation",
            "question": "Does the kernel hard-code the feature dimension?",
            "source_refs": ["kernel.py:1-4"],
        },
        context=ToolContext(state=state, run_dir=run_dir, current_role=Role.SKEPTIC.value, current_turn=1),
    )
    registry.call(
        "record_description_update",
        {
            "summary": "The source is too small to prove hard-coding, but the visible model is recorded.",
            "task_id": request["id"],
            "contract_model": ["The toy problem requires adding one to each element."],
            "kernel_model": ["The kernel returns x + 1."],
            "risk_map": ["Boundary behavior should be checked only if benchmark-covered."],
            "scope_notes": ["Use test.py/get_inputs before marking a nearby case in scope."],
        },
        context=ToolContext(state=state, run_dir=run_dir, current_role=Role.DESCRIBER.value, current_turn=2),
    )

    persisted = AgenticOrchestrator(state=state, run_dir=run_dir).persist(stop_reason="description_test")
    loaded = load_run_state(persisted.run_json)
    transcript = persisted.transcript_md.read_text()

    assert loaded.description_model.kernel_model == ["The kernel returns x + 1."]
    assert loaded.description_tasks[0].status == "resolved"
    assert loaded.description_updates[0].task_ids == ["d1"]
    assert "## Description Model" in transcript
    assert "The kernel returns x + 1" in transcript


def test_retrieve_experiment_history_reads_persisted_probe_events(tmp_path) -> None:
    run_dir = tmp_path / "run"
    registry = build_core_registry()
    state = RunState(entry="toy")
    context = ToolContext(state=state, run_dir=run_dir)

    registry.call(
        "run_python_probe",
        {
            "code": "import json\nprint(json.dumps({'verdict': 'match'}))\n",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )
    claim = registry.call(
        "record_claim",
        {"statement": "Claim-bound probe should be persisted.", "rationale": "History must include claim probes."},
        context=context,
    )
    registry.call(
        "run_claim_probe",
        {
            "claim_id": claim["id"],
            "code": "import json\nprint(json.dumps({'claim_probe': True}))\n",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=context,
    )
    AgenticOrchestrator(state=state, run_dir=run_dir).persist()

    new_state = RunState(entry="toy")
    history = registry.call(
        "retrieve_experiment_history",
        {"limit": 5},
        context=ToolContext(state=new_state, run_dir=run_dir),
    )

    assert history["exists"] is True
    assert [event["tool"] for event in history["events"]] == ["run_python_probe", "run_claim_probe"]
    assert history["events"][0]["output"]["json_result"] == {"verdict": "match"}
    assert history["events"][1]["output"]["json_result"] == {"claim_probe": True}


def _write_artifact(dataset_root) -> None:
    entry_dir = dataset_root / "toy"
    entry_dir.mkdir(parents=True)
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": "toy", "passed": True, "status": "passed", "rounds": 1})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (entry_dir / "test.py").write_text("def test():\n    pass\n")
