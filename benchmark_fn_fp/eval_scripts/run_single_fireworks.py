"""Fireworks source-only baseline using the shared, lossless trace layout."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline2_single_llm import SYSTEM_PROMPT, USER_TEMPLATE, VERDICT_SCHEMA
from models import profile_for, profile_for_trace, pricing_snapshot
from datasets import (DATASETS, case_dataset, case_names, cases_dir as dataset_cases_dir,
                      checked_case_hashes, ensure_active_dataset)
from traces import (reserve_trace, write_trace, next_trial_id, single_call_readable_files,
                    trace_path, update_trace_metadata, experiment_trial_id)
from verifier.agentic.llm_trace import LLMCallTrace
from verifier.agentic.provenance import file_sha256, runtime_fingerprint

MODEL = "accounts/fireworks/models/glm-5p3"


def run_one(name, *, dataset, trial, max_tokens, timeout_s=1800, reasoning_effort="low", original_trial=None):
    dataset = case_dataset(REPO, dataset, name, require_active=True)
    if reasoning_effort not in {"default", "low", "medium", "high"}:
        raise ValueError("Unsupported reasoning effort")
    from openai import OpenAI
    rates = pricing_snapshot(MODEL)
    profile = profile_for_trace(MODEL, {"pricing_snapshot": rates})
    cases_dir = dataset_cases_dir(REPO, dataset)
    hashes = checked_case_hashes(REPO, dataset, name)
    code = (cases_dir/name/"kernel.py").read_text()
    problem = (cases_dir/name/"problem.txt").read_text()
    system = SYSTEM_PROMPT + "\nOutput schema:\n" + json.dumps(VERDICT_SCHEMA)
    prompt = USER_TEMPLATE.format(problem=problem,kernel=code)
    request = {"model":MODEL,"max_tokens":max_tokens,
        "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
        "response_format":{"type":"json_object"}}
    if reasoning_effort != "default":
        request["reasoning_effort"] = reasoning_effort
    runtime = runtime_fingerprint()
    provenance = {"verifier_sha256": runtime["verifier_sha256"],
                  "runner_sha256": file_sha256(__file__),
                  "public_input_files": ["kernel.py", "problem.txt"]}
    dest=reserve_trace(name,"single_call",traces_dir=profile.traces_dir,trial=trial,metadata={
        "model":MODEL,"provider":"fireworks","dataset":dataset,"max_tokens":max_tokens,
        "reasoning_effort":reasoning_effort,"timeout_s":timeout_s,
        "raw_api_capture":True,"original_trial":original_trial or experiment_trial_id(trial),**hashes,**provenance})
    write_trace(name,"single_call",traces_dir=profile.traces_dir,trial=trial,files={
        "request.json":json.dumps(request,indent=2),"system_prompt.txt":system,"user_prompt.txt":prompt,
        "runtime.json":json.dumps(runtime,indent=2)})
    client=OpenAI(api_key=os.environ["FIREWORKS_API_KEY"],base_url="https://api.fireworks.ai/inference/v1",
                  timeout=timeout_s,max_retries=0)
    started=time.monotonic();call=None
    print(f"{name}/single_call started; trace={dest}",flush=True)
    try:
        call=LLMCallTrace(str(dest/"llm_calls"),request)
        response=client.chat.completions.create(**request)
        call.record_response(response)
        call.finish()
        raw_response=response.model_dump(mode="json")
        choice=raw_response["choices"][0]; message=choice["message"]
        raw=message.get("content") or ""
        try:
            answer=json.loads(raw)
            if not isinstance(answer,dict) or answer.get("verdict") not in VERDICT_SCHEMA["properties"]["verdict"]["enum"]:
                raise ValueError("Invalid verdict")
        except ValueError:
            answer={"verdict":"no_verdict","reason":"Missing or invalid final JSON verdict"}
        usage=response.usage
        meta={"model":MODEL,"provider":"fireworks","response":answer,
            "reasoning_effort":reasoning_effort,"timeout_s":timeout_s,
            "usage":{"input_tokens":usage.prompt_tokens,"output_tokens":usage.completion_tokens},
            "stop_reason":choice["finish_reason"],"max_tokens":max_tokens,"elapsed_s":time.monotonic()-started,
            **hashes,"prompt_variant":"original","estimated_usd":
            (usage.prompt_tokens*profile.price_in+usage.completion_tokens*profile.price_out)/1e6,
            "pricing_snapshot":rates,
            "pricing":"dated list-price estimate; not invoice; excludes GPU and unreported HTTP usage"}
        files={"raw_response.json":json.dumps(raw_response,indent=2),"response_text.json":raw,
               "usage.json":json.dumps(meta,indent=2),"runner_error.txt":""}
        files.update(single_call_readable_files(system=system,user=prompt,response=raw,
                     thinking=message.get("reasoning_content") or "",usage=meta))
        write_trace(name,"single_call",traces_dir=profile.traces_dir,trial=trial,files=files,
                    metadata={"status":"completed","stop_reason":choice["finish_reason"]})
        print(f"{name}/single_call completed: {answer.get('verdict')} finish={choice['finish_reason']} "
              f"output_tokens={usage.completion_tokens} API_estimate=${meta['estimated_usd']:.6f}",flush=True)
        return meta
    except Exception as exc:
        if call is not None:call.finish(exc)
        write_trace(name,"single_call",traces_dir=profile.traces_dir,trial=trial,
                    files={"runner_error.txt":f"{type(exc).__name__}: see llm_calls/*/error.json for sanitized details\n"},
                    metadata={"status":"error"})
        raise


def main(argv=None):
    from dotenv import load_dotenv
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases',default='')
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--dataset',choices=DATASETS,default='benchmark_fn_fp')
    parser.add_argument('--trial',default='')
    parser.add_argument('--max-tokens',type=int,default=32768)
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--reasoning-effort',choices=('default','low','medium','high'),default='low',
                        help="default omits the API field and uses the provider's default")
    parser.add_argument('--concurrency',type=int,default=2)
    args=parser.parse_args(argv)
    ensure_active_dataset(args.dataset)
    load_dotenv(REPO/'.env')
    if not os.getenv('FIREWORKS_API_KEY'):raise RuntimeError('FIREWORKS_API_KEY is not set')
    if args.max_tokens<1 or args.concurrency<1 or args.timeout<1:parser.error('Budgets and timeout must be positive')
    names=[n.strip() for n in args.cases.split(',') if n.strip()]
    if not names and args.all:names=case_names(REPO,args.dataset)
    if not names:parser.error('pass --cases or --all')
    if len(names)!=len(set(names)):parser.error('duplicate cases')
    trial=args.trial or next_trial_id(names, ['single_call'], traces_dir=profile_for(MODEL).traces_dir)
    original_trial=experiment_trial_id(args.trial)
    for name in names:
        case_dataset(REPO,args.dataset,name,require_active=True)
        checked_case_hashes(REPO,args.dataset,name)
        dest=trace_path(name,'single_call',traces_dir=profile_for(MODEL).traces_dir,trial=trial)
        if dest.exists():raise FileExistsError(dest)
    failures=[]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures={pool.submit(run_one,n,dataset=args.dataset,trial=trial,max_tokens=args.max_tokens,
                             timeout_s=args.timeout,reasoning_effort=args.reasoning_effort,
                             original_trial=original_trial):n for n in names}
        for future in as_completed(futures):
            try:future.result()
            except Exception as exc:
                failures.append(futures[future]);print(f"{futures[future]} failed: {type(exc).__name__}; trace retained",flush=True)
    if failures:raise SystemExit(f"Failed trials: {failures}")

if __name__=='__main__':main()
