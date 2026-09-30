"""Isolated source-only trials; same original prompt/schema; all attempts saved."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"eval_scripts"))
from baseline2_single_llm import SYSTEM_PROMPT, USER_TEMPLATE, VERDICT_SCHEMA


def main():
    from anthropic import Anthropic
    from dotenv import load_dotenv
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",default="case_36,case_37")
    parser.add_argument("--trial",default="r1")
    parser.add_argument("--max-tokens",type=int,default=32768)
    parser.add_argument("--neutral-contract",action="store_true",
                        help="Ablation: remove the nonbinding branch-error explanation, retaining the exact final-output contract")
    args=parser.parse_args()
    load_dotenv(ROOT.parents[1]/".env")
    client=Anthropic(max_retries=0,timeout=900,default_headers={"accept-encoding":"gzip, deflate"})
    validation=json.loads((ROOT/"validation_gpu.json").read_text())
    for name in args.cases.split(","):
        folder=ROOT/"eval_cases"/name
        code=(folder/"kernel.py").read_text()
        problem=(folder/"problem.txt").read_text()
        v=validation["cases"][name]
        for text,field in [(code,"kernel_sha256"),(problem,"problem_sha256")]:
            assert hashlib.sha256(text.encode()).hexdigest()==v[field]
        if args.neutral_contract:
            note="Either branch\nmay individually exceed 0.1 without violating the contract.\n"
            assert note in problem
            problem=problem.replace(note,"")
        dest=ROOT/"traces"/name/"single_call"/args.trial
        if dest.exists():raise RuntimeError(f"Already attempted: {dest}")
        dest.mkdir(parents=True)
        prompt=USER_TEMPLATE.format(problem=problem,kernel=code)
        (dest/"user_prompt.txt").write_text(prompt)
        (dest/"system_prompt.txt").write_text(SYSTEM_PROMPT)
        start=time.monotonic()
        print(f"{name} {args.trial} starting max_tokens={args.max_tokens}",flush=True)
        with client.messages.stream(model="claude-opus-5",max_tokens=args.max_tokens,
                system=SYSTEM_PROMPT,
                output_config={"format":{"type":"json_schema","schema":VERDICT_SCHEMA}},
                messages=[{"role":"user","content":prompt}]) as stream:
            response=stream.get_final_message()
        raw="".join(b.text for b in response.content if getattr(b,"type",None)=="text")
        (dest/"response_text.json").write_text(raw)
        usage=response.usage.model_dump()
        meta={"model":"claude-opus-5","usage":usage,"stop_reason":response.stop_reason,
              "max_tokens":args.max_tokens,"elapsed_s":time.monotonic()-start,
              "kernel_sha256":v["kernel_sha256"],
              "source_problem_sha256":v["problem_sha256"],
              "problem_sha256":hashlib.sha256(problem.encode()).hexdigest(),
              "prompt_variant":"neutral_contract" if args.neutral_contract else "original",
              "estimated_usd":(usage["input_tokens"]*5+usage["output_tokens"]*25)/1e6,
              "pricing":"existing project profile estimate; GPU charges excluded"}
        try:meta["response"]=json.loads(raw)
        except ValueError:meta["response"]={"verdict":"no_verdict"}
        (dest/"usage.json").write_text(json.dumps(meta,indent=2))
        print(name,args.trial,json.dumps(meta),flush=True)


if __name__=="__main__":main()
