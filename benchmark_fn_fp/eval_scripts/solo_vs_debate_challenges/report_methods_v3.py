"""Separate case_72–case_75 cohort; reuse the same trace validity and pairing rules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from benchmark_fn_fp.eval_scripts.solo_vs_debate_challenges import report_methods_v2 as renderer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Also export JSON under private_data/reports/")
    args = parser.parse_args()
    renderer.main(export_json=args.json, version="v3", cases=tuple(f"case_{i + 61:02d}" for i in range(11, 15)),
                  trials=("ea_methods_v3_r1", "ea_methods_v3_r2"))
