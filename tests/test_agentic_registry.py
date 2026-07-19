from __future__ import annotations

import json

import pytest

from verifier.agentic.state import Role, RunState, ToolStatus
from verifier.agentic.tools.registry import ToolContext, ToolRegistryError, build_core_registry


def test_core_registry_records_claim() -> None:
    state = RunState()
    registry = build_core_registry()

    result = registry.call(
        "record_claim",
        {
            "statement": "non-contiguous input may be read as contiguous",
            "rationale": "the source appears to use contiguous pointer arithmetic",
        },
        context=ToolContext(state=state),
    )

    assert result["id"] == "c1"
    assert state.claims[0].statement == "non-contiguous input may be read as contiguous"
    assert state.tool_events[0].id == "t1"
    assert state.tool_events[0].status == ToolStatus.OK


def test_skeptic_record_claim_is_limited_to_three_per_turn() -> None:
    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state, current_role=Role.SKEPTIC.value, current_turn=1)

    for index in range(3):
        registry.call(
            "record_claim",
            {
                "statement": f"claim {index}",
                "rationale": "test rationale",
            },
            context=context,
        )

    with pytest.raises(ValueError, match="at most 3 claims per turn"):
        registry.call(
            "record_claim",
            {
                "statement": "claim 4",
                "rationale": "test rationale",
            },
            context=context,
        )

    assert len(state.claims) == 3
    assert [event.status for event in state.tool_events] == [
        ToolStatus.OK,
        ToolStatus.OK,
        ToolStatus.OK,
        ToolStatus.ERROR,
    ]

    next_turn = ToolContext(state=state, current_role=Role.SKEPTIC.value, current_turn=2)
    result = registry.call(
        "record_claim",
        {
            "statement": "new turn claim",
            "rationale": "test rationale",
        },
        context=next_turn,
    )

    assert result["id"] == "c4"
    assert len(state.claims) == 4


def test_skeptic_can_record_no_new_claims_review() -> None:
    state = RunState()
    registry = build_core_registry()

    result = registry.call(
        "record_no_new_claims",
        {
            "reason": "All current claims were reviewed and no further in-scope hypotheses remain.",
            "reviewed_claims": ["c1", "c2"],
        },
        context=ToolContext(state=state, current_role=Role.SKEPTIC.value, current_turn=3),
    )

    assert result["decision"] == "no_new_claims"
    assert result["reviewed_claims"] == ["c1", "c2"]
    assert result["reviewed_tool_event_count"] == 0
    assert result["turn"] == 3
    assert state.skeptic_review == result
    assert state.tool_events[0].tool == "record_no_new_claims"


def test_record_no_new_claims_is_skeptic_only() -> None:
    state = RunState()
    registry = build_core_registry()

    with pytest.raises(ValueError, match="only be called by the skeptic"):
        registry.call(
            "record_no_new_claims",
            {
                "reason": "Judge cannot assert skeptic review.",
                "reviewed_claims": [],
            },
            context=ToolContext(state=state, current_role=Role.JUDGE.value, current_turn=1),
        )


def test_registry_validates_required_args() -> None:
    registry = build_core_registry()

    with pytest.raises(ToolRegistryError, match="missing required arg"):
        registry.call("record_claim", {"statement": "missing rationale"}, context=ToolContext(RunState()))


def test_load_artifact_tool_loads_dataset_entry(tmp_path) -> None:
    _write_artifact(tmp_path)

    state = RunState()
    registry = build_core_registry()
    result = registry.call(
        "load_artifact",
        {"entry": "toy"},
        context=ToolContext(state=state, dataset_dir=tmp_path),
    )

    assert result["entry"] == "toy"
    assert result["status"] == "failed"
    assert "def kernel" in result["kernel_code"]
    assert state.entry == "toy"


def test_artifact_tools_inspect_source_problem_and_files(tmp_path) -> None:
    _write_artifact(tmp_path)

    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state, dataset_dir=tmp_path)

    files = registry.call("list_artifact_files", {"entry": "toy"}, context=context)
    source = registry.call(
        "inspect_kernel_source",
        {"entry": "toy", "start_line": 2, "end_line": 3},
        context=context,
    )
    problem = registry.call("inspect_problem", {"entry": "toy"}, context=context)
    seed = registry.call("read_artifact_file", {"entry": "toy", "path": "seed_0.py"}, context=context)

    assert {item["path"] for item in files["files"]} >= {"kernel.py", "problem.txt", "seed_0.py"}
    assert source["content"] == "2:     y = x + 1\n3:     return y"
    assert problem["exists"] is True
    assert "Add one" in problem["content"]
    assert seed["kind"] == "seed_source"
    assert len(state.tool_events) == 4
    assert [event.id for event in state.tool_events] == ["t1", "t2", "t3", "t4"]


def test_read_artifact_file_rejects_path_escape_and_records_error(tmp_path) -> None:
    _write_artifact(tmp_path)

    state = RunState()
    registry = build_core_registry()

    with pytest.raises(ValueError, match="escapes entry directory"):
        registry.call(
            "read_artifact_file",
            {"entry": "toy", "path": "../outside.txt"},
            context=ToolContext(state=state, dataset_dir=tmp_path),
        )

    assert state.tool_events[0].tool == "read_artifact_file"
    assert state.tool_events[0].status == ToolStatus.ERROR
    assert state.tool_events[0].output["error_type"] == "ValueError"


def test_claim_tools_append_evidence_update_status_and_read_ledger() -> None:
    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state)

    claim = registry.call(
        "record_claim",
        {
            "statement": "kernel ignores stride",
            "rationale": "source line uses x + offsets without stride",
        },
        context=context,
    )
    evidence = registry.call(
        "append_evidence",
        {
            "claim_id": claim["id"],
            "kind": "source_inspection",
            "summary": "Line 8 indexes x + offsets directly.",
            "supports": "confirmed",
            "tool_event_id": "t1",
            "data": {"path": "kernel.py", "line": 8},
        },
        context=context,
    )
    updated = registry.call(
        "update_claim_status",
        {"claim_id": claim["id"], "status": "confirmed"},
        context=context,
    )
    ledger = registry.call("read_claim_ledger", {}, context=context)

    assert evidence["id"] == "c1.e1"
    assert updated["status"] == "confirmed"
    assert ledger["claims"][0]["evidence"][0]["tool_event_id"] == "t1"
    assert [event.tool for event in state.tool_events] == [
        "record_claim",
        "append_evidence",
        "update_claim_status",
        "read_claim_ledger",
    ]


def test_description_tools_record_model_and_resolve_task() -> None:
    state = RunState()
    registry = build_core_registry()

    request = registry.call(
        "request_description",
        {
            "reason_kind": "contract_scope",
            "question": "Does the benchmark require feature size 0?",
            "related_claims": ["c1"],
            "source_refs": ["test.py"],
        },
        context=ToolContext(state=state, current_role=Role.JUDGE.value, current_turn=1),
    )
    update = registry.call(
        "record_description_update",
        {
            "summary": "The benchmark fixes features to 64, so feature size 0 is not directly covered.",
            "task_id": request["id"],
            "contract_model": ["test.py fixes features=64."],
            "kernel_model": ["The kernel specializes the feature dimension."],
            "risk_map": ["Shape specialization is the relevant risk."],
            "scope_notes": ["features==0 needs explicit scope evidence before reject."],
            "open_questions": ["Are nearby feature sizes benchmark-covered?"],
            "impact_on_claims": ["c1"],
        },
        context=ToolContext(state=state, current_role=Role.DESCRIBER.value, current_turn=2),
    )

    assert request["id"] == "d1"
    assert state.description_tasks[0].status == "resolved"
    assert state.description_tasks[0].response_summary.startswith("The benchmark fixes")
    assert state.description_model.contract_model == ["test.py fixes features=64."]
    assert update["update"]["task_ids"] == ["d1"]
    assert [event.tool for event in state.tool_events] == ["request_description", "record_description_update"]


def test_record_description_update_is_describer_only() -> None:
    state = RunState()
    registry = build_core_registry()

    with pytest.raises(ValueError, match="only be called by the describer"):
        registry.call(
            "record_description_update",
            {"summary": "Skeptic cannot mutate the description model."},
            context=ToolContext(state=state, current_role=Role.SKEPTIC.value, current_turn=1),
        )


def test_request_more_debate_tool_records_judge_request() -> None:
    state = RunState()
    registry = build_core_registry()

    result = registry.call(
        "request_more_debate",
        {
            "reason": "Skeptic should review inconclusive evidence.",
            "focus_claims": ["c1"],
        },
        context=ToolContext(state=state),
    )

    assert result["request"] == "more_debate"
    assert result["focus_claims"] == ["c1"]
    assert state.convergence == result
    assert state.tool_events[0].tool == "request_more_debate"


def test_record_claim_requires_scope_evidence_for_in_scope_claim() -> None:
    state = RunState()
    registry = build_core_registry()

    with pytest.raises(ValueError, match="scope_evidence"):
        registry.call(
            "record_claim",
            {
                "statement": "in-scope claim without evidence",
                "rationale": "test rationale",
                "scope": "in_scope",
                "scope_rationale": "This claims to be in scope but cites no source.",
            },
            context=ToolContext(state=state),
        )


def test_record_claim_allows_out_of_scope_without_scope_evidence() -> None:
    state = RunState()
    registry = build_core_registry()

    claim = registry.call(
        "record_claim",
        {
            "statement": "out-of-scope edge case",
            "rationale": "test rationale",
            "scope": "out_of_scope",
            "scope_rationale": "This is outside the contract.",
        },
        context=ToolContext(state=state),
    )

    assert claim["scope"] == "out_of_scope"
    assert claim["scope_evidence"] == []


def test_reject_verdict_requires_in_scope_confirmed_decisive_claim() -> None:
    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state)
    claim = registry.call(
        "record_claim",
        {"statement": "unknown scope bug", "rationale": "test rationale"},
        context=context,
    )
    registry.call(
        "append_evidence",
        {
            "claim_id": claim["id"],
            "kind": "runtime_probe",
            "summary": "Confirmed by probe.",
            "supports": "confirmed",
        },
        context=context,
    )
    registry.call("update_claim_status", {"claim_id": claim["id"], "status": "confirmed"}, context=context)

    with pytest.raises(ValueError, match="scope_evidence"):
        registry.call(
            "record_verdict",
            {
                "verdict": "reject",
                "confidence": 0.9,
                "decisive_claims": [claim["id"]],
                "reason": "Unknown-scope claim should not reject.",
            },
            context=context,
        )


def test_reject_verdict_rejects_problem_only_scope_evidence() -> None:
    state = RunState()
    registry = build_core_registry()
    context = ToolContext(state=state)
    claim = registry.call(
        "record_claim",
        {
            "statement": "problem-only in-scope bug",
            "rationale": "test rationale",
            "scope": "in_scope",
            "scope_rationale": "The case is inferred only from the broad problem text.",
            "scope_evidence": [
                {"source": "problem.txt", "summary": "The broad operator description implies this case."}
            ],
        },
        context=context,
    )
    registry.call(
        "append_evidence",
        {
            "claim_id": claim["id"],
            "kind": "runtime_probe",
            "summary": "Confirmed by probe.",
            "supports": "confirmed",
        },
        context=context,
    )
    registry.call("update_claim_status", {"claim_id": claim["id"], "status": "confirmed"}, context=context)

    with pytest.raises(ValueError, match="benchmark/test-domain scope_evidence"):
        registry.call(
            "record_verdict",
            {
                "verdict": "reject",
                "confidence": 0.9,
                "decisive_claims": [claim["id"]],
                "reason": "Problem-only scope evidence should not reject.",
            },
            context=context,
        )


def test_reject_verdict_accepts_test_domain_confirmed_decisive_claim(tmp_path) -> None:
    _write_artifact(tmp_path)
    state = RunState(entry="toy")
    registry = build_core_registry()
    context = ToolContext(state=state, dataset_dir=tmp_path)
    claim = registry.call(
        "record_claim",
        {
            "statement": "in-scope bug",
            "rationale": "test rationale",
            "scope": "in_scope",
            "scope_rationale": "The case is required by test.py/get_inputs.",
            "scope_evidence": [
                {"source": "test.py::get_inputs", "summary": "The benchmark input generator creates this case."}
            ],
        },
        context=context,
    )
    registry.call(
        "append_evidence",
        {
            "claim_id": claim["id"],
            "kind": "runtime_probe",
            "summary": "Confirmed by probe.",
            "supports": "confirmed",
        },
        context=context,
    )
    registry.call("update_claim_status", {"claim_id": claim["id"], "status": "confirmed"}, context=context)

    verdict = registry.call(
        "record_verdict",
        {
            "verdict": "reject",
            "confidence": 0.9,
            "decisive_claims": [claim["id"]],
            "reason": "Benchmark-domain confirmed claim can reject.",
        },
        context=context,
    )

    assert verdict["verdict"] == "reject"


def _write_artifact(dataset_root) -> None:
    entry_dir = dataset_root / "toy"
    entry_dir.mkdir()
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": "toy", "passed": False, "status": "failed", "rounds": 2})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    y = x + 1\n    return y\n")
    (entry_dir / "test.py").write_text("features = 64\ndef test():\n    pass\n")
    (entry_dir / "seed_0.py").write_text("def seed(x):\n    return x\n")
