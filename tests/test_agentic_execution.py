from __future__ import annotations

import json
from pathlib import Path

from verifier.agentic.state import Role, RunState, ToolStatus
from verifier.agentic.tools.registry import ToolContext, build_core_registry


def test_run_python_probe_captures_stdout_json_and_artifacts(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    context = ToolContext(
        state=state,
        run_dir=tmp_path / "run",
        current_role=Role.EXPERIMENTER.value,
    )

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
    context = ToolContext(
        state=state,
        dataset_dir=tmp_path / "dataset",
        run_dir=tmp_path / "run",
        current_role=Role.ORCHESTRATOR.value,
    )

    registry.call("load_artifact", {"entry": "toy"}, context=context)
    result = registry.call(
        "run_python_probe",
        {
            "code": "import json\nimport kernel\nprint(json.dumps({'value': kernel.kernel(4)}))\n",
            "timeout_s": 5,
            "use_gpu": False,
        },
        context=ToolContext(
            state=state,
            dataset_dir=tmp_path / "dataset",
            run_dir=tmp_path / "run",
            current_role=Role.EXPERIMENTER.value,
        ),
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
        context=ToolContext(
            state=state,
            run_dir=tmp_path / "run",
            current_role=Role.EXPERIMENTER.value,
        ),
    )

    assert result["event_id"] == "t1"
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert "before sleep" in result["stdout"]
    assert state.tool_events[0].status == ToolStatus.OK


def test_run_claim_probe_returns_claim_bound_evidence_draft(tmp_path) -> None:
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    context = ToolContext(
        state=state,
        run_dir=tmp_path / "run",
        current_role=Role.EXPERIMENTER.value,
    )
    claim = registry.call(
        "record_claim",
        {"statement": "Probe should observe value 3.", "rationale": "The hypothesis needs runtime evidence."},
        context=ToolContext(
            state=state,
            run_dir=tmp_path / "run",
            current_role=Role.SKEPTIC.value,
        ),
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
    context = ToolContext(
        state=state,
        run_dir=tmp_path / "run",
        current_role=Role.EXPERIMENTER.value,
    )
    claim = registry.call(
        "record_claim",
        {"statement": "Probe should observe value 3.", "rationale": "The hypothesis needs runtime evidence."},
        context=ToolContext(
            state=state,
            run_dir=tmp_path / "run",
            current_role=Role.SKEPTIC.value,
        ),
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
        context=ToolContext(
            state=state,
            run_dir=tmp_path / "run",
            current_role=Role.EXPERIMENTER.value,
        ),
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


def test_finalize_requires_named_data_when_the_probe_printed_no_json(tmp_path) -> None:
    """A claim settled on unstructured output alone must name its numbers.

    stdout and json_result are carried into evidence `data` automatically, and
    the claim ledger never scrolls, so measurements normally reach the Judge on
    their own. A probe that printed no JSON is the exception: raw stdout is
    trimmed to a budget it can exceed, and prose summaries are trimmed harder,
    so a number that exists only there can be gone by the time the verdict is
    written. Settling a claim on that is not allowed.
    """
    state = RunState(entry="adhoc")
    registry = build_core_registry()
    experimenter = ToolContext(state=state, run_dir=tmp_path / "run",
                               current_role=Role.EXPERIMENTER.value)
    claim = registry.call(
        "record_claim",
        {"statement": "The tail block is mishandled.", "rationale": "No masking is visible."},
        context=ToolContext(state=state, run_dir=tmp_path / "run",
                            current_role=Role.SKEPTIC.value),
    )
    probe = registry.call(
        "run_claim_probe",
        {"claim_id": claim["id"], "code": "print('no json here, just prose')",
         "timeout_s": 5, "use_gpu": False},
        context=experimenter,
    )
    assert probe["json_result"] is None

    refused = registry.call(
        "finalize_probe_evidence",
        {"event_id": probe["event_id"], "supports": "confirmed",
         "summary": "It looked wrong to me."},
        context=experimenter,
    )
    assert refused["ok"] is False
    assert "must name its decisive measurements" in refused["message"]

    # Naming the measurement is all it takes.
    ok = registry.call(
        "finalize_probe_evidence",
        {"event_id": probe["event_id"], "supports": "confirmed",
         "summary": "Measured the tail-block difference.", "data": {"max_abs_err": 60.57}},
        context=experimenter,
    )
    assert ok["claim"]["status"] == "confirmed"
    assert ok["evidence"]["data"]["max_abs_err"] == 60.57


def _probe_ctx():
    """The solo role, which may both raise claims and probe them."""
    from verifier.agentic.state import RunState, Role
    from verifier.agentic.tools.registry import ToolContext, build_core_registry

    state = RunState()
    return build_core_registry(), ToolContext(
        state=state, current_role=Role.SOLO.value, current_turn=1
    ), state


def _open_claim(registry, context):
    result = registry.call(
        "record_claim",
        {"statement": "The kernel drops the trailing partial group when K is not an exact "
                      "multiple of group_size, so those columns use the wrong scale row.",
         "rationale": "floor division on the group count"},
        context=context,
    )
    assert result.get("ok") is not False, result
    return result["id"]


def test_a_claim_with_an_unconsumed_successful_probe_cannot_be_reprobed() -> None:
    """A good result that has not been interpreted must be spent, not repeated.

    Probes for independent claims go out together and are finalized next turn.
    When one claim's probe code keeps failing the batch keeps being re-issued,
    and a claim that already succeeded is re-probed alongside it every turn --
    it is still open, so the launch rule still names it. A measured run probed
    one claim four times for three identical results while a sibling failed four
    times running, and the wasted turns came out of the budget the stuck claim
    needed.
    """
    registry, context, _ = _probe_ctx()
    claim_id = _open_claim(registry, context)

    first = registry.call(
        "run_claim_probe",
        {"claim_id": claim_id, "code": "print(\'{\"ok\": 1}\')", "use_gpu": False},
        context=context,
    )
    assert first["exit_code"] == 0, first

    again = registry.call(
        "run_claim_probe",
        {"claim_id": claim_id, "code": "print(\'{\"ok\": 2}\')", "use_gpu": False},
        context=context,
    )
    assert again["ok"] is False
    assert "finalize_probe_evidence" in again["message"]

    # Spending it unblocks the claim for a genuinely different experiment.
    spent = registry.call(
        "finalize_probe_evidence",
        {"event_id": first["event_id"], "supports": "confirmed",
         "summary": "measured", "data": {"max_abs_err": 1.0}},
        context=context,
    )
    assert spent.get("ok") is not False, spent
    third = registry.call(
        "run_claim_probe",
        {"claim_id": claim_id, "code": "print(\'{\"ok\": 3}\')", "use_gpu": False},
        context=context,
    )
    assert third.get("ok") is not False, third


def test_a_failed_probe_may_always_be_rewritten() -> None:
    """Rewriting broken probe code is the loop working, not waste.

    One measured run needed four attempts to get its int4 bit-packing right.
    Blocking that would break the recovery path the guard above depends on.
    """
    registry, context, _ = _probe_ctx()
    claim_id = _open_claim(registry, context)

    for _ in range(3):
        result = registry.call(
            "run_claim_probe",
            {"claim_id": claim_id, "code": "import sys; sys.exit(1)", "use_gpu": False},
            context=context,
        )
        assert result.get("ok") is not False, result
        assert result["exit_code"] != 0
