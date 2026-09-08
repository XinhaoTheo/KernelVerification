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
    context = ToolContext(
        state=state,
        run_dir=run_dir,
        current_role=Role.EXPERIMENTER.value,
    )

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
    context = ToolContext(
        state=state,
        run_dir=run_dir,
        current_role=Role.EXPERIMENTER.value,
    )

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
        context=ToolContext(state=state, run_dir=run_dir, current_role=Role.SKEPTIC.value),
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
        context=ToolContext(
            state=new_state,
            run_dir=run_dir,
            current_role=Role.EXPERIMENTER.value,
        ),
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


def test_every_eval_runner_writes_a_trace() -> None:
    """No runner may finish a run without leaving its complete record.

    Fifteen-odd evaluation runs were made before traces existed, and not one left
    a complete record: the Modal runners kept the last 20,000 characters of
    transcript.md and discarded run.json, tool_events.jsonl, claims.json and
    every probe. Three defects that changed conclusions -- a Skeptic re-reading
    material already in its prompt, a tool rejecting the Describer's first call
    in every run, an agent reaching the right verdict for three wrong reasons --
    were invisible until full traces existed. This test fails if a runner ever
    stops writing one.
    """
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parent.parent / "benchmark_fn_fp" / "eval"
    runners = [
        "baseline2_single_llm.py",
        "run_agentic_modal.py",
    ]
    for name in runners:
        path = eval_dir / name
        assert path.exists(), f"missing runner {name}"
        source = path.read_text()
        assert "write_trace" in source or "TRACES_DIR" in source, (
            f"{name} does not write a trace; a run whose record is thrown away "
            f"cannot be diagnosed afterwards"
        )


def test_every_modal_runner_sets_max_tokens() -> None:
    """A runner that leaves max_tokens at the default silently produces nothing.

    Adaptive thinking is billed against max_tokens, so at the 4096 default a turn
    can spend its entire budget inside the thinking block and return no text and
    no tool call. Two of three near-identical runners passed 16384 and the third
    did not, and nothing caught it: a full 32-case debate run produced three
    cases with zero claims and zero probes, one Skeptic returning nothing nine
    turns in a row, and a Judge that wrote "the debate produced no claims and no
    evidence" and recorded a verdict anyway. The whole batch was discarded. The
    three runners are now one, which removes the way that bug was possible, and
    this test guards the remaining copy.
    """
    from pathlib import Path

    eval_dir = Path(__file__).resolve().parent.parent / "benchmark_fn_fp" / "eval"
    for name in ("run_agentic_modal.py",):
        source = (eval_dir / name).read_text()
        assert '"--max-tokens"' in source, (
            f"{name} does not pass --max-tokens; at the 4096 default whole turns "
            f"return no text and no tool call"
        )
