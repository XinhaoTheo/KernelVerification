from __future__ import annotations

import json
from pathlib import Path

from verifier.agentic.agents import build_describer_agent, build_experimenter_agent, build_judge_agent, build_skeptic_agent
from verifier.agentic.llm import default_model
from verifier.agentic.orchestrator import AgenticOrchestrator, build_context_response
from verifier.agentic.protocol import parse_agent_response
from verifier.agentic.state import Role
from verifier.agentic_run import main as agentic_main


class FakeLLMClient:
    def __init__(self, text: str | list[str]):
        self.responses = [text] if isinstance(text, str) else list(text)
        self.calls = []

    def call(self, *, system: str, user: str, max_tokens: int = 4096) -> str:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if not self.responses:
            raise AssertionError("FakeLLMClient has no responses left")
        return self.responses.pop(0)


def test_describer_agent_uses_json_protocol_without_recording_claim(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "The kernel appears to add one to the input and return the result.",
                "tool_calls": [],
            }
        )
    )

    orchestrator = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")
    orchestrator.apply_agent_response(role=Role.ORCHESTRATOR, response=build_context_response("toy"))
    outputs = orchestrator.run_agent_once(build_describer_agent(fake))

    assert outputs == []
    assert len(orchestrator.state.history) == 2
    assert orchestrator.state.history[-1].role == Role.DESCRIBER
    assert orchestrator.state.history[-1].text.startswith("The kernel appears")
    assert orchestrator.state.claims == []
    assert "Do not record claims" in fake.calls[0]["system"]


def test_skeptic_agent_uses_json_protocol_and_records_claim(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "The source suggests a stride-specific hypothesis worth recording.",
                "tool_calls": [
                    {
                        "tool": "record_claim",
                        "args": {
                            "statement": "Non-contiguous inputs may be treated as contiguous.",
                            "rationale": "The visible kernel source does not mention stride handling.",
                            "raised_by": "skeptic",
                        },
                    }
                ],
            }
        )
    )

    orchestrator = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")
    orchestrator.apply_agent_response(role=Role.ORCHESTRATOR, response=build_context_response("toy"))
    outputs = orchestrator.run_agent_once(build_skeptic_agent(fake))

    assert outputs[0]["tool"] == "record_claim"
    assert orchestrator.state.claims[0].statement == "Non-contiguous inputs may be treated as contiguous."
    assert fake.calls
    assert "You must respond with exactly one JSON object" in fake.calls[0]["system"]
    assert "Available Tools" in fake.calls[0]["user"]


def test_experimenter_agent_updates_evidence_and_status(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "The current claim needs evidence; mark it inconclusive until a targeted probe exists.",
                "tool_calls": [
                    {"tool": "read_claim_ledger", "args": {}},
                    {
                        "tool": "append_evidence",
                        "args": {
                            "claim_id": "c1",
                            "kind": "agent_analysis",
                            "summary": "No runtime probe has been run yet, so the claim is not decided.",
                            "supports": "inconclusive",
                            "tool_event_id": "t5",
                            "data": {"reason": "needs targeted runtime probe"},
                        },
                    },
                    {
                        "tool": "update_claim_status",
                        "args": {"claim_id": "c1", "status": "inconclusive"},
                    },
                ],
            }
        )
    )

    orchestrator = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")
    orchestrator.apply_agent_response(role=Role.ORCHESTRATOR, response=build_context_response("toy"))
    orchestrator.apply_agent_response(
        role=Role.SKEPTIC,
        response=parse_agent_response(
            json.dumps(
                {
                    "message": "Record a claim for experimenter.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Boundary sizes may be mishandled.",
                                "rationale": "The implementation context does not yet show boundary evidence.",
                            },
                        }
                    ],
                }
            )
        ),
    )
    outputs = orchestrator.run_agent_once(build_experimenter_agent(fake))

    assert [item["tool"] for item in outputs] == [
        "read_claim_ledger",
        "append_evidence",
        "update_claim_status",
    ]
    assert orchestrator.state.claims[0].status == "inconclusive"
    assert orchestrator.state.claims[0].evidence[0].supports == "inconclusive"
    assert "run_python_probe" in fake.calls[0]["system"]
    assert "claim_coverage" in fake.calls[0]["user"]


def test_judge_agent_records_verdict(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "The ledger has only inconclusive evidence, so more evidence is needed.",
                "tool_calls": [
                    {
                        "tool": "record_verdict",
                        "args": {
                            "verdict": "needs_more_evidence",
                            "confidence": 0.4,
                            "decisive_claims": ["c1"],
                            "reason": "The only claim is inconclusive and no runtime probe decides it.",
                        },
                    }
                ],
            }
        )
    )

    orchestrator = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run")
    orchestrator.apply_agent_response(role=Role.ORCHESTRATOR, response=build_context_response("toy"))
    orchestrator.apply_agent_response(
        role=Role.SKEPTIC,
        response=parse_agent_response(
            json.dumps(
                {
                    "message": "Record a claim for judge.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Boundary sizes may be mishandled.",
                                "rationale": "No boundary evidence is present yet.",
                                "scope": "in_scope",
                                "scope_rationale": "Boundary sizes are part of the toy problem input contract.",
                                "scope_evidence": [
                                    {"source": "test.py::get_inputs", "summary": "The benchmark input generator includes this boundary case."}
                                ],
                            },
                        }
                    ],
                }
            )
        ),
    )
    outputs = orchestrator.run_agent_once(build_judge_agent(fake))

    assert outputs[0]["tool"] == "record_verdict"
    assert orchestrator.state.verdict["verdict"] == "needs_more_evidence"
    assert orchestrator.persist().verdict_json.exists()
    assert "Use record_verdict exactly once" in fake.calls[0]["system"]


def test_agentic_cli_single_skeptic_uses_fake_client(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "Record one concrete claim.",
                "tool_calls": [
                    {
                        "tool": "record_claim",
                        "args": {
                            "statement": "Boundary sizes may be mishandled.",
                            "rationale": "The prompt context is not enough to rule out boundary assumptions.",
                        },
                    }
                ],
            }
        )
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agent",
            "skeptic",
            "--max-rounds",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode: skeptic" in captured.out
    assert "claims: 1" in captured.out
    assert (tmp_path / "run" / "run.json").exists()


def test_agentic_cli_runs_describer_then_skeptic_with_fake_client(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps(
                {
                    "message": "The kernel appears to add one to x.",
                    "tool_calls": [],
                }
            ),
            json.dumps(
                {
                    "message": "Record one concrete claim after reading the description.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "The kernel may only handle scalar-like inputs.",
                                "rationale": "The current context does not show shape-general indexing.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic",
            "--max-debate-rounds",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode: describer+skeptic" in captured.out
    assert "turns: 3" in captured.out
    assert "claims: 1" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == ["orchestrator", "describer", "skeptic"]


def test_agentic_cli_runs_describer_skeptic_experimenter_with_fake_client(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "The kernel appears to add one to x.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Record a concrete claim.",
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
            ),
            json.dumps(
                {
                    "message": "Attach inconclusive evidence until a probe is run.",
                    "tool_calls": [
                        {"tool": "read_claim_ledger", "args": {}},
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "agent_analysis",
                                "summary": "A runtime probe has not been executed yet.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {
                            "tool": "update_claim_status",
                            "args": {"claim_id": "c1", "status": "inconclusive"},
                        },
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter",
            "--max-debate-rounds",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode: describer+skeptic+experimenter" in captured.out
    assert "turns: 4" in captured.out
    assert "claims: 1" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
    ]
    assert run_data["claims"][0]["status"] == "inconclusive"



def test_agentic_cli_runs_full_agent_chain_with_judge(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "The kernel appears to add one to x.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Record a concrete claim.",
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
            ),
            json.dumps(
                {
                    "message": "Attach inconclusive evidence until a probe is run.",
                    "tool_calls": [
                        {"tool": "read_claim_ledger", "args": {}},
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "agent_analysis",
                                "summary": "A runtime probe has not been executed yet.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {
                            "tool": "update_claim_status",
                            "args": {"claim_id": "c1", "status": "inconclusive"},
                        },
                    ],
                }
            ),
            json.dumps({"message": "Round 2 description sees evidence.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "No new claims after reviewing c1 evidence.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "The only recorded claim has evidence and no further in-scope hypotheses remain.",
                                "reviewed_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Final verdict needs more evidence.",
                    "tool_calls": [
                        {
                            "tool": "record_verdict",
                            "args": {
                                "verdict": "needs_more_evidence",
                                "confidence": 0.5,
                                "decisive_claims": ["c1"],
                                "reason": "The claim remains inconclusive.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter,judge",
            "--max-debate-rounds",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "mode: describer+skeptic+experimenter+judge" in captured.out
    assert "stop_reason: verdict_recorded" in captured.out
    assert "verdict_json:" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
        "describer",
        "skeptic",
        "judge",
    ]
    assert run_data["skeptic_review"]["decision"] == "no_new_claims"
    assert run_data["verdict"]["verdict"] == "needs_more_evidence"


def test_agentic_cli_workflow_runs_claim_coverage_loop_before_judge(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "Describe implementation.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Record two claims.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Boundary sizes may be mishandled.",
                                "rationale": "No boundary evidence is present yet.",
                            },
                        },
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Non-contiguous inputs may be mishandled.",
                                "rationale": "No stride evidence is present yet.",
                            },
                        },
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Cover c1 first; c2 remains uncovered.",
                    "tool_calls": [
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "agent_analysis",
                                "summary": "Boundary claim still needs a runtime probe.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {"tool": "update_claim_status", "args": {"claim_id": "c1", "status": "inconclusive"}},
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Cover c2 next.",
                    "tool_calls": [
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c2",
                                "kind": "agent_analysis",
                                "summary": "Stride claim still needs a runtime probe.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {"tool": "update_claim_status", "args": {"claim_id": "c2", "status": "inconclusive"}},
                    ],
                }
            ),
            json.dumps({"message": "Round 2 describe after evidence.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "No new claims after reviewing both covered claims.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "Both claims are covered and no further in-scope hypotheses remain.",
                                "reviewed_claims": ["c1", "c2"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Both claims have evidence coverage, so Judge can finalize.",
                    "tool_calls": [
                        {
                            "tool": "record_verdict",
                            "args": {
                                "verdict": "needs_more_evidence",
                                "confidence": 0.35,
                                "decisive_claims": ["c1", "c2"],
                                "reason": "Both claims are covered but inconclusive.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter,judge",
            "--max-debate-rounds",
            "2",
            "--max-claim-rounds",
            "1",
            "--max-claim-rounds-per-claim",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stop_reason: verdict_recorded" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
        "experimenter",
        "describer",
        "skeptic",
        "judge",
    ]
    assert [claim["status"] for claim in run_data["claims"]] == ["inconclusive", "inconclusive"]
    assert run_data["skeptic_review"]["reviewed_claims"] == ["c1", "c2"]
    assert run_data["verdict"]["verdict"] == "needs_more_evidence"


def test_agentic_cli_workflow_allows_probe_result_to_be_consumed_next_turn(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "Describe implementation.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Record one claim.",
                    "tool_calls": [
                        {
                            "tool": "record_claim",
                            "args": {
                                "statement": "Boundary sizes may be mishandled.",
                                "rationale": "No boundary evidence is present yet.",
                                "scope": "in_scope",
                                "scope_rationale": "Boundary sizes are part of the toy problem input contract.",
                                "scope_evidence": [
                                    {"source": "test.py::get_inputs", "summary": "The benchmark input generator includes this boundary case."}
                                ],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Run a probe first; evidence will be attached after tool output is visible.",
                    "tool_calls": [
                        {
                            "tool": "run_python_probe",
                            "args": {
                                "code": "import json\nprint(json.dumps({'claim_id':'c1','passed':False}))\n",
                                "timeout_s": 5,
                                "use_gpu": False,
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Consume the previous probe result as evidence.",
                    "tool_calls": [
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "runtime_probe",
                                "summary": "Previous probe reported passed=false for c1.",
                                "supports": "confirmed",
                                "tool_event_id": "t6",
                                "data": {"passed": False},
                            },
                        },
                        {"tool": "update_claim_status", "args": {"claim_id": "c1", "status": "confirmed"}},
                    ],
                }
            ),
            json.dumps({"message": "Round 2 describe after confirmed evidence.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "No new claims after reviewing confirmed c1.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "The confirmed claim is decisive and no further in-scope hypotheses remain.",
                                "reviewed_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Confirmed correctness-impact claim supports reject.",
                    "tool_calls": [
                        {
                            "tool": "record_verdict",
                            "args": {
                                "verdict": "reject",
                                "confidence": 0.8,
                                "decisive_claims": ["c1"],
                                "reason": "Runtime evidence confirmed the claim.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter,judge",
            "--max-debate-rounds",
            "2",
            "--max-claim-rounds",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stop_reason: verdict_recorded" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
        "experimenter",
        "describer",
        "skeptic",
        "judge",
    ]
    assert [event["tool"] for event in run_data["tool_events"]][-5:] == [
        "run_python_probe",
        "append_evidence",
        "update_claim_status",
        "record_no_new_claims",
        "record_verdict",
    ]
    assert run_data["claims"][0]["status"] == "confirmed"
    assert run_data["verdict"]["verdict"] == "reject"


def test_agentic_cli_respects_min_debate_rounds_before_judge(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "Round 1 describe.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Round 1 claim.",
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
            ),
            json.dumps(
                {
                    "message": "Cover the claim before debate round 2.",
                    "tool_calls": [
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "agent_analysis",
                                "summary": "Evidence is inconclusive but claim is covered.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {"tool": "update_claim_status", "args": {"claim_id": "c1", "status": "inconclusive"}},
                    ],
                }
            ),
            json.dumps({"message": "Round 2 describe with evidence visible.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Round 2 no new claims.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "c1 has been reviewed and no new in-scope claim remains.",
                                "reviewed_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Now Judge can finalize.",
                    "tool_calls": [
                        {
                            "tool": "record_verdict",
                            "args": {
                                "verdict": "needs_more_evidence",
                                "confidence": 0.45,
                                "decisive_claims": ["c1"],
                                "reason": "The claim was covered but remains inconclusive.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter,judge",
            "--max-debate-rounds",
            "2",
            "--min-debate-rounds-before-judge",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stop_reason: verdict_recorded" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
        "describer",
        "skeptic",
        "judge",
    ]
    assert run_data["skeptic_review"]["decision"] == "no_new_claims"
    assert run_data["verdict"]["verdict"] == "needs_more_evidence"


def test_agentic_cli_judge_can_request_another_debate_round(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        [
            json.dumps({"message": "Round 1 describe.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Round 1 claim.",
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
            ),
            json.dumps(
                {
                    "message": "Cover the claim.",
                    "tool_calls": [
                        {
                            "tool": "append_evidence",
                            "args": {
                                "claim_id": "c1",
                                "kind": "agent_analysis",
                                "summary": "Evidence is inconclusive but claim is covered.",
                                "supports": "inconclusive",
                                "data": {},
                            },
                        },
                        {"tool": "update_claim_status", "args": {"claim_id": "c1", "status": "inconclusive"}},
                    ],
                }
            ),
            json.dumps({"message": "Round 2 describe.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Round 2 no follow-up claims.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "c1 was reviewed and no additional in-scope claim remains.",
                                "reviewed_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Judge wants another debate round.",
                    "tool_calls": [
                        {
                            "tool": "request_more_debate",
                            "args": {
                                "reason": "Skeptic should critique the inconclusive evidence once more.",
                                "focus_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps({"message": "Round 3 describe.", "tool_calls": []}),
            json.dumps(
                {
                    "message": "Round 3 still no follow-up claims.",
                    "tool_calls": [
                        {
                            "tool": "record_no_new_claims",
                            "args": {
                                "reason": "After Judge requested another round, c1 was reviewed again and no new claim remains.",
                                "reviewed_claims": ["c1"],
                            },
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "message": "Final verdict after requested debate.",
                    "tool_calls": [
                        {
                            "tool": "record_verdict",
                            "args": {
                                "verdict": "needs_more_evidence",
                                "confidence": 0.4,
                                "decisive_claims": ["c1"],
                                "reason": "The extra debate produced no resolving evidence.",
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "describer,skeptic,experimenter,judge",
            "--max-debate-rounds",
            "3",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stop_reason: verdict_recorded" in captured.out
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == [
        "orchestrator",
        "describer",
        "skeptic",
        "experimenter",
        "describer",
        "skeptic",
        "judge",
        "describer",
        "skeptic",
        "judge",
    ]
    assert [event["tool"] for event in run_data["tool_events"]][-3:] == [
        "request_more_debate",
        "record_no_new_claims",
        "record_verdict",
    ]
    assert run_data["verdict"]["verdict"] == "needs_more_evidence"

def test_agentic_cli_replays_existing_run_and_continues_with_judge(tmp_path, monkeypatch) -> None:
    _write_artifact(tmp_path / "dataset")
    run_dir = tmp_path / "run"
    first_fake = FakeLLMClient(
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
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: first_fake)
    agentic_main(["toy", "--dataset-dir", str(tmp_path / "dataset"), "--run-dir", str(run_dir), "--agent", "skeptic"])

    second_fake = FakeLLMClient(
        json.dumps(
            {
                "message": "Judge replayed ledger.",
                "tool_calls": [
                    {
                        "tool": "record_verdict",
                        "args": {
                            "verdict": "needs_more_evidence",
                            "confidence": 0.3,
                            "decisive_claims": ["c1"],
                            "reason": "The replayed claim has no decisive evidence.",
                        },
                    }
                ],
            }
        )
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: second_fake)
    agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(run_dir),
            "--replay-run",
            str(run_dir / "run.json"),
            "--agent",
            "judge",
            "--no-require-claim-coverage",
        ]
    )

    run_data = json.loads((run_dir / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == ["orchestrator", "skeptic", "judge"]
    assert run_data["verdict"]["verdict"] == "needs_more_evidence"


def test_agentic_cli_skips_judge_when_open_claim_lacks_evidence(tmp_path, monkeypatch, capsys) -> None:
    _write_artifact(tmp_path / "dataset")
    fake = FakeLLMClient(
        json.dumps(
            {
                "message": "Record an uncovered claim.",
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
    )
    monkeypatch.setattr("verifier.agentic_run.build_llm_client", lambda provider=None, model=None: fake)

    exit_code = agentic_main(
        [
            "toy",
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--run-dir",
            str(tmp_path / "run"),
            "--agents",
            "skeptic,judge",
            "--max-debate-rounds",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "stop_reason: claim_coverage_required" in captured.out
    assert "verdict_json:" not in captured.out
    assert len(fake.calls) == 1
    run_data = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [turn["role"] for turn in run_data["history"]] == ["orchestrator", "skeptic"]
    assert run_data["verdict"] is None
    assert run_data["claims"][0]["status"] == "open"
    assert run_data["claims"][0]["evidence"] == []



def test_core_agents_load_adversarial_precision_skill(tmp_path) -> None:
    _write_artifact(tmp_path / "dataset")
    fakes = [
        FakeLLMClient(json.dumps({"message": "describe", "tool_calls": []})),
        FakeLLMClient(json.dumps({"message": "skeptic", "tool_calls": []})),
        FakeLLMClient(json.dumps({"message": "experiment", "tool_calls": []})),
        FakeLLMClient(json.dumps({"message": "judge", "tool_calls": []})),
    ]
    builders = [build_describer_agent, build_skeptic_agent, build_experimenter_agent, build_judge_agent]
    state = AgenticOrchestrator(dataset_dir=tmp_path / "dataset", run_dir=tmp_path / "run").state

    for builder, fake in zip(builders, fakes):
        builder(fake).act(state=state, tools=[])
        assert "Adversarial Precision Verification" in fake.calls[0]["system"]
        if builder is not build_describer_agent:
            assert "Metric Selection" in fake.calls[0]["system"]
            assert "Scope Policy" in fake.calls[0]["system"]


def test_default_model_prefers_agentic_model(monkeypatch) -> None:
    monkeypatch.setenv("AGENTIC_MODEL", "agentic-test-model")

    assert default_model() == "agentic-test-model"


def test_default_model_uses_openai_provider_default(monkeypatch) -> None:
    monkeypatch.delenv("AGENTIC_MODEL", raising=False)
    monkeypatch.setenv("AGENTIC_PROVIDER", "openai")

    assert default_model() == "gpt-5"


def _write_artifact(dataset_root: Path) -> None:
    entry_dir = dataset_root / "toy"
    entry_dir.mkdir(parents=True)
    (entry_dir / "meta.json").write_text(
        json.dumps({"name": "toy", "passed": True, "status": "passed", "rounds": 1})
    )
    (entry_dir / "problem.txt").write_text("Add one to every element.\n")
    (entry_dir / "kernel.py").write_text("def kernel(x):\n    return x + 1\n")
    (entry_dir / "test.py").write_text("features = 64\ndef test():\n    pass\n")
