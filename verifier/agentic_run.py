"""CLI entry point for the agentic verifier."""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from verifier import dataset
from verifier.agentic.agents import build_describer_agent, build_experimenter_agent, build_judge_agent, build_skeptic_agent
from verifier.agentic.llm import build_llm_client, default_provider
from verifier.agentic.persistence import load_run_state
from verifier.agentic.orchestrator import AgenticOrchestrator, build_context_response
from verifier.agentic.state import Role

_print = partial(print, flush=True)

_AGENT_BUILDERS = {
    "describer": build_describer_agent,
    "skeptic": build_skeptic_agent,
    "experimenter": build_experimenter_agent,
    "judge": build_judge_agent,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic kernel verification orchestrator.")
    parser.add_argument("entry", nargs="?", help="dataset entry name")
    parser.add_argument("--all", action="store_true", help="run all dataset entries")
    parser.add_argument("--dataset-dir", default="dataset", help="dataset root directory")
    parser.add_argument("--run-dir", default=None, help="directory for run artifacts")
    parser.add_argument("--replay-run", default=None, help="path to an existing run.json to load before continuing")
    parser.add_argument("--dry-run", action="store_true", help="execute deterministic protocol/tool plumbing")
    parser.add_argument("--agent", choices=sorted(_AGENT_BUILDERS), default=None, help="run one real LLM agent role")
    parser.add_argument(
        "--agents",
        default=None,
        help="comma-separated real LLM agent roles to run in order, e.g. describer,skeptic,experimenter,judge",
    )
    parser.add_argument("--provider", choices=["anthropic", "openai", "chatgpt"], default=None, help="LLM provider")
    parser.add_argument(
        "--max-debate-rounds",
        "--max-rounds",
        dest="max_debate_rounds",
        type=int,
        default=1,
        help="max outer debate rounds; --max-rounds is a deprecated alias",
    )
    parser.add_argument("--max-claim-rounds", type=int, default=3, help="minimum Experimenter rounds per debate round for claim coverage")
    parser.add_argument(
        "--min-debate-rounds-before-judge",
        type=int,
        default=1,
        help="minimum outer debate rounds that must complete before Judge can run",
    )
    parser.add_argument(
        "--max-claim-rounds-per-claim",
        type=int,
        default=3,
        help="dynamic Experimenter rounds per uncovered claim",
    )
    parser.add_argument("--model", default=None, help="override AGENTIC_MODEL for this run")
    parser.add_argument("--max-tokens", type=int, default=4096, help="max tokens per LLM agent call")
    parser.add_argument("--tool-budget", type=int, default=None, help="max tool calls during the LLM agent loop")
    parser.add_argument("--stop-when-no-open-claims", action="store_true", help="stop loop after a round resolves all claims")
    parser.add_argument(
        "--no-require-claim-coverage",
        action="store_true",
        help="allow Judge to run even when open claims have no evidence",
    )
    args = parser.parse_args(argv)

    if args.max_debate_rounds < 1:
        parser.error("--max-debate-rounds must be >= 1")
    if args.max_claim_rounds < 1:
        parser.error("--max-claim-rounds must be >= 1")
    if args.min_debate_rounds_before_judge < 1:
        parser.error("--min-debate-rounds-before-judge must be >= 1")
    if args.max_claim_rounds_per_claim < 1:
        parser.error("--max-claim-rounds-per-claim must be >= 1")
    if args.dry_run and (args.agent or args.agents):
        parser.error("--dry-run cannot be combined with --agent or --agents")
    if args.all and args.entry:
        parser.error("provide either an entry or --all, not both")
    if not args.all and not args.entry:
        parser.error("provide an entry or --all")
    if args.replay_run and args.all:
        parser.error("--replay-run is only supported for a single entry")

    agent_names = _parse_agent_names(args.agent, args.agents, parser)
    if not args.dry_run and not agent_names:
        parser.error("use --dry-run, --agent, or --agents")

    entries = list(dataset.iter_entries(dataset_dir=Path(args.dataset_dir))) if args.all else [args.entry]
    if not entries:
        parser.error("no dataset entries found")

    failures = 0
    for entry in entries:
        try:
            _run_one_entry(entry, args, agent_names)
        except Exception as exc:
            failures += 1
            _print(f"entry: {entry}")
            _print(f"error: {type(exc).__name__}: {exc}")
            if not args.all:
                raise
    return 1 if failures else 0


def _run_one_entry(entry: str, args, agent_names: list[str]) -> None:
    mode = "dry_run" if args.dry_run else "+".join(agent_names)
    run_dir = _run_dir_for_entry(entry, args, mode)
    state = load_run_state(Path(args.replay_run)) if args.replay_run else None
    orchestrator = (
        AgenticOrchestrator(state=state, dataset_dir=Path(args.dataset_dir), run_dir=run_dir)
        if state
        else AgenticOrchestrator(dataset_dir=Path(args.dataset_dir), run_dir=run_dir)
    )

    if args.dry_run:
        outputs = orchestrator.apply_agent_response(
            role=Role.ORCHESTRATOR,
            response=build_context_response(entry),
        )
        loop_result = None
    else:
        outputs = []
        if not args.replay_run:
            outputs = orchestrator.apply_agent_response(
                role=Role.ORCHESTRATOR,
                response=build_context_response(entry),
            )
        agents = _build_agents(
            agent_names,
            provider=args.provider,
            model=args.model,
            max_tokens=args.max_tokens,
        )
        if _uses_workflow(agent_names):
            loop_result = orchestrator.run_verification_workflow(
                agents,
                max_debate_rounds=args.max_debate_rounds,
                max_claim_rounds=args.max_claim_rounds,
                max_claim_rounds_per_claim=args.max_claim_rounds_per_claim,
                min_debate_rounds_before_judge=args.min_debate_rounds_before_judge,
                tool_budget=args.tool_budget,
                stop_when_no_open_claims=args.stop_when_no_open_claims,
                require_claim_coverage=not args.no_require_claim_coverage,
            )
        else:
            loop_result = orchestrator.run_agents_sequential(
                agents,
                max_rounds=args.max_debate_rounds,
                tool_budget=args.tool_budget,
                stop_when_no_open_claims=args.stop_when_no_open_claims,
                require_claim_coverage=not args.no_require_claim_coverage,
            )
        outputs.extend(loop_result.outputs)

    persisted = orchestrator.persist(stop_reason=loop_result.stop_reason if loop_result is not None else None)

    _print(f"entry: {entry}")
    _print(f"mode: {mode}")
    if not args.dry_run:
        _print(f"provider: {args.provider or default_provider()}")
    _print(f"tools_executed: {len(outputs)}")
    _print(f"turns: {len(orchestrator.state.history)}")
    _print(f"claims: {len(orchestrator.state.claims)}")
    if loop_result is not None:
        _print(f"stop_reason: {loop_result.stop_reason}")
    _print(f"run_dir: {persisted.run_dir}")
    _print(f"run_json: {persisted.run_json}")
    _print(f"transcript_md: {persisted.transcript_md}")
    if persisted.verdict_json is not None:
        _print(f"verdict_json: {persisted.verdict_json}")


def _run_dir_for_entry(entry: str, args, mode: str) -> Path:
    if args.run_dir:
        base = Path(args.run_dir)
        return base / entry if args.all else base
    return Path(args.dataset_dir) / entry / "agentic_runs" / str(mode)


def _parse_agent_names(agent: str | None, agents: str | None, parser: argparse.ArgumentParser) -> list[str]:
    if agent and agents:
        parser.error("use either --agent or --agents, not both")
    if agent:
        return [agent]
    if not agents:
        return []

    names = [name.strip() for name in agents.split(",") if name.strip()]
    if not names:
        parser.error("--agents must contain at least one agent name")
    invalid = [name for name in names if name not in _AGENT_BUILDERS]
    if invalid:
        parser.error(f"unsupported agent(s): {', '.join(invalid)}")
    return names


def _uses_workflow(agent_names: list[str]) -> bool:
    return "experimenter" in agent_names and "judge" in agent_names


def _build_agents(names: list[str], *, provider: str | None, model: str | None, max_tokens: int):
    llm_client = build_llm_client(provider=provider, model=model)
    return [_AGENT_BUILDERS[name](llm_client, max_tokens=max_tokens) for name in names]


if __name__ == "__main__":
    raise SystemExit(main())
