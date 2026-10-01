"""Derive case_50–case_61 results; update only README's OZ RESULTS block, without model calls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "single_call_vs_tools_challenges"
sys.path.insert(0, str(ROOT.parents[1]))
from benchmark_fn_fp.eval_scripts.single_call_vs_tools_challenges.report_extension import ARMS, derive, render, replace_results

CASES = tuple(f"case_{i:02d}" for i in range(50, 62))
LOW_TRIALS = ("oz_low32_r1", "oz_low32_r2", "oz_low32_r3")
PLANNED = {
    LOW_TRIALS[0]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ARMS},
    LOW_TRIALS[1]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ARMS},
    LOW_TRIALS[2]: {"reasoning_effort": "low", "max_tokens": 32768, "arms": ("single_call",)},
    "oz_default64_r1": {"reasoning_effort": "default", "max_tokens": 65536,
                        "arms": ("single_call",)},
}


def main(*, export_json=False):
    result = derive(ROOT, cases=CASES, low_trials=LOW_TRIALS, planned=PLANNED,
                    required_families=3, case_range="case_50–case_61")
    replace_results(ROOT, "OZ RESULTS", render(result))
    if export_json:
        output = ROOT / "private_data" / "reports" / "oz_scoreboard.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Recorded attempts: {len(result['rows'])}; qualifying cases: "
          f"{result['replication_gate']['qualifying_cases']}")
    print(f"Recorded API estimate: {result['recorded_api_estimate_usd']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    main(export_json=parser.parse_args().json)
