"""Update only README's METHODS V2/V3 RESULTS block; preserve the pilot and raw traces."""
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark_fn_fp.eval_scripts.solo_vs_debate_challenges.report import derive, render, replace_results

ROOT = Path(__file__).resolve().parents[2] / "solo_vs_debate_challenges"
CASES = tuple(f"case_{i + 61:02d}" for i in range(5, 11))
TRIALS = ("ea_methods_v2_r1", "ea_methods_v2_r2")


def main(*, version="v2", cases=CASES, trials=TRIALS, export_json=False):
    if version not in {"v2", "v3"}:
        raise ValueError("Unknown report version")
    result = derive(cases=cases, trials=trials)
    replace_results(ROOT, f"METHODS {version.upper()} RESULTS", render(result))
    if export_json:
        output = ROOT / "private_data" / "reports" / f"methods_{version}_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": result["groups"], "gate": result["gate"],
                      "api_estimate_usd": result["api_estimate_usd"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
