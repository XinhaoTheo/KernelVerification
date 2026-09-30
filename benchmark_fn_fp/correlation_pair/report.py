"""Derive a local scoreboard from frozen GPU labels and all trial traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"eval_scripts"))
from audit_traces import audit
from models import profile_for
from summarize_traces import read_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fireworks",action="store_true")
    args=parser.parse_args()
    trace_root="traces_fireworks" if args.fireworks else "traces"
    model="accounts/fireworks/models/glm-5p3" if args.fireworks else "claude-opus-5"
    suffix="_fireworks" if args.fireworks else ""
    labels=json.loads((ROOT/"validation_gpu.json").read_text())
    rows=[]
    for case,truth in labels["cases"].items():
        base=ROOT/trace_root/case
        for path in sorted(base.glob("single_call/*/usage.json")):
            d=json.loads(path.read_text())
            verdict=d["response"].get("verdict")
            rows.append({"case":case,"arm":"single_call","trial":path.parent.name,
                "truth":truth["ground_truth"],"verdict":verdict,"correct":verdict==truth["ground_truth"],
                "confidence":d["response"].get("confidence"),"stop_reason":d["stop_reason"],
                "prompt_variant":d.get("prompt_variant","original"),
                "reason":d["response"].get("reason"),"usd":d["estimated_usd"],
                "api_calls":1,"input_tokens":d["usage"]["input_tokens"],
                "llm_elapsed_s":d.get("elapsed_s"),
                "output_tokens":d["usage"]["output_tokens"],"path":str(path.relative_to(ROOT))})
        for arm in ("solo","debate"):
            for path in sorted(base.glob(f"{arm}/*/run.json")):
                d=read_run(path,profile_for(model))
                run=json.loads(path.read_text())
                profile=profile_for(model)
                d["usd"]=sum(((u.get("input_tokens",0) or 0)*profile.price_in
                    +(u.get("output_tokens",0) or 0)*profile.price_out
                    +(u.get("cache_creation_input_tokens",0) or 0)*profile.price_in*profile.cache_write
                    +(u.get("cache_read_input_tokens",0) or 0)*profile.price_in*profile.cache_read)/1e6
                    for u in (h.get("usage") or {} for h in run.get("history",[])))
                for field,hash_field in (("kernel_code","kernel_sha256"),("problem_text","problem_sha256")):
                    assert hashlib.sha256(run["artifact"][field].encode()).hexdigest()==truth[hash_field], path
                tokens={k:sum((h.get("usage") or {}).get(k,0) or 0 for h in run.get("history",[]))
                        for k in ("input_tokens","output_tokens")}
                rows.append({"case":case,"arm":arm,"trial":path.parent.name,
                    "truth":truth["ground_truth"],"correct":d["verdict"]==truth["ground_truth"],
                    **d,**tokens,
                    "api_calls":sum(bool(h.get("usage")) for h in run.get("history",[])),
                    "llm_elapsed_s":sum(h.get("duration_s") or 0 for h in run.get("history",[])),
                    "reason":(run.get("verdict") or {}).get("reason"),
                    "audit":audit(path.parent),"path":str(path.relative_to(ROOT))})
    for row in rows:
        if row.get("stop_reason") in ("length","max_tokens"):
            row["outcome"]="token_limit"
        elif row["verdict"]=="needs_more_evidence":
            row["outcome"]="abstention"
        elif row["verdict"] not in ("trust","reject"):
            row["outcome"]="no_verdict"
        else:
            row["outcome"]="correct" if row["correct"] else "wrong_verdict"
    attempts=[]
    for path in sorted((ROOT/trace_root).glob("*/*/*/request.json")):
        dest=path.parent
        if not (dest/"usage.json").exists() and not (dest/"run.json").exists():
            error=dest/"runner_error.txt"
            attempts.append({"path":str(dest.relative_to(ROOT)),
                             "status":"error" if error.exists() and error.read_text() else "pending",
                             "error":error.read_text() if error.exists() else None})
    summary={"model":model,"rows":rows,"incomplete_attempts":attempts,
             "estimated_api_usd":round(sum(r["usd"] for r in rows),4),
             "gpu_cost":"not measured; excluded", "labels":labels}
    summary["arms"]={}
    for arm in ("single_call","solo","debate"):
        arm_rows=[r for r in rows if r["arm"]==arm]
        summary["arms"][arm]={"completed_trials":len(arm_rows),
            "outcomes":{outcome:sum(r["outcome"]==outcome for r in arm_rows)
                        for outcome in ("correct","wrong_verdict","abstention","token_limit","no_verdict")},
            "estimated_usd":round(sum(r["usd"] for r in arm_rows),4),
            "api_calls":sum(r["api_calls"] for r in arm_rows),
            "probes":sum(r.get("probes",0) for r in arm_rows),
            "llm_elapsed_s":round(sum(r.get("llm_elapsed_s") or 0 for r in arm_rows),3)}
    (ROOT/f"scoreboard{suffix}.json").write_text(json.dumps(summary,indent=2))
    lines=["# Correlation-pair experiment", "",
        "Frozen before model evaluation; every trial below is retained. CPU construction and GPU verification "
        "are separate from the answer-free artifacts given to the models.","",
        "## GPU labels", "", "| Case | Relative L2 error | Budget | Label |",
        "| --- | ---: | ---: | --- |"]
    for case,d in labels["cases"].items():
        lines.append(f"| {case} | {d['errors'][0]:.10f} | 0.1 | {d['ground_truth']} |")
    lines += ["", "Each case ran ten times on T4; input hashes match CPU construction and two FP64 reference "
              "implementations agree. GPU environment: " + json.dumps(labels["environment"]), "",
              "## All trials", "", "| Case | Arm | Trial | Verdict | Outcome | Confidence | API estimate |",
              "| --- | --- | --- | --- | --- | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['arm']} | {r['trial']} | {r['verdict']} | {r['outcome']} | "
                     f"{r.get('confidence')} | ${r['usd']:.4f} |")
    lines += ["",f"Recorded API estimate: **${summary['estimated_api_usd']:.4f}**. Uses the existing "
              "project model profile, not a billing invoice; Modal charges are excluded.","",
              "## Trial evidence", ""]
    for r in rows:
        lines += [f"### {r['case']} / {r['arm']} / {r['trial']}","",r.get("reason") or "No final explanation.",""]
        lines += [f"API calls with recorded usage: {r['api_calls']}; input tokens: {r['input_tokens']}; "
                  f"output tokens: {r['output_tokens']}; probes: {r.get('probes',0)}.",""]
        if "audit" in r:lines += ["Automated trace audit: "+json.dumps(r["audit"]),""]
        else:lines += [f"Stop reason: {r['stop_reason']}; output tokens: {r['output_tokens']}.",""]
        if r.get("prompt_variant")=="neutral_contract":
            lines += ["Prompt ablation: removed only the explanatory sentence allowing either branch to exceed "
                      "0.1 individually; final-output contract, threshold, source and inputs are unchanged.",""]
    lines += ["## Interpretation limits", "",
        "This is a selected synthetic pair testing finite-workload numerical compliance. It is not a proof "
        "over all valid inputs, not a representative dataset, and not a claim that the kernel is an upstream bug. "
        "With the correct reference and metric supplied, a fixed numerical script solves it. A tool advantage "
        "does not imply a debate advantage over a solo agent with tools. A source-only abstention is legitimate "
        "uncertainty; a token-cap exhaustion is not a wrong semantic judgment. Repeated calls on two frozen "
        "cases measure repeatability, not performance on independent held-out cases.",""]
    lines += ["Model: "+model,"", "Incomplete attempts: "+json.dumps(attempts),""]
    (ROOT/("REPORT_FIREWORKS.md" if args.fireworks else "REPORT.md")).write_text("\n".join(lines))
    print(json.dumps({"trials":len(rows),"api_estimate":summary["estimated_api_usd"]}))


if __name__=="__main__":main()
