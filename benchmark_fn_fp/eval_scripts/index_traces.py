"""Generate registry-wide GLM coverage and the complete historical trial index."""
from collections import defaultdict
from pathlib import Path
from summarize_traces import build_report, BENCHMARK
from case_registry import case_collection, case_sort_key, is_active_case, load_registry
from index_cases import DATASET_LABELS, curation_lines
from traces import iter_trace_records, trace_selection_key, trial_sort_key

ARMS = ("single_call", "solo", "debate")
FINAL_VERDICTS = {"trust", "reject", "needs_more_evidence"}
LENGTH_STOPS = {"length", "max_tokens", "max_output_tokens"}


def completed(row, metadata):
    """A wrong verdict or abstention is complete; an interrupted run is not."""
    return (
        row.get("verdict") in FINAL_VERDICTS
        and row.get("outcome") not in {"token_limit", "no_verdict", "runner_error", "error", "running"}
        and not row.get("runner_error")
        and metadata.get("status") in {None, "completed", "historical"}
        and row.get("stop_reason") not in LENGTH_STOPS
        and metadata.get("stop_reason") not in LENGTH_STOPS
    )


def build_coverage(rows, details, metadata_by_path):
    """Choose the first completed attempt per registry case/arm, never the best answer."""
    details = {case: detail for case, detail in details.items() if is_active_case(detail)}
    grouped = defaultdict(list)
    for row in rows:
        if row["case"] in details and row["arm"] in ARMS:
            grouped[(row["case"], row["arm"])].append(row)
    totals = {arm: {"completed": 0, "missing": 0, "failed_or_unfinished": 0, "running": 0}
              for arm in ARMS}
    cases = []
    for case, detail in sorted(details.items(), key=lambda item: case_sort_key(item[0])):
        arms = {}
        for arm in ARMS:
            attempts = sorted(grouped[(case, arm)], key=lambda row: trace_selection_key(
                row, metadata_by_path.get(row["path"], {})))
            selected = next((row for row in attempts
                             if completed(row, metadata_by_path.get(row["path"], {}))), None)
            status = ("completed" if selected else "missing" if not attempts else
                      "running" if any(metadata_by_path.get(row["path"], {}).get("status") == "running"
                                       for row in attempts) else "failed_or_unfinished")
            arms[arm] = {"status": status, "selected": selected, "attempts": len(attempts)}
            totals[arm][status] += 1
        cases.append({"case": case, "dataset": detail["dataset"],
                      "collection": case_collection(detail), "arms": arms})
    return {"cases": cases, "totals": totals}


def _target(row, benchmark):
    path = Path(row["path"]).relative_to("traces_glm")
    filename = "transcript.md" if (benchmark / row["path"] / "transcript.md").exists() else "trace_meta.json"
    return (path / filename).as_posix()


def _raw_capture_complete(row, benchmark):
    """Confirm saved payloads; raw_api_capture only says capture was enabled."""
    status = row.get("usage_coverage", {}).get("status")
    if status not in {None, "complete", "legacy_history"}:
        return False
    directory = benchmark / row["path"]
    calls = [path for path in (directory / "llm_calls").glob("*") if path.is_dir()]
    if calls:
        return status == "complete" and all(
            (call / "request.json").is_file() and (call / "response.json").is_file()
            for call in calls)
    # Original single-call runs predate llm_calls but saved the full API pair.
    return row["arm"] == "single_call" and all(
        (directory / filename).is_file() for filename in ("request.json", "raw_response.json"))


def _coverage_lines(coverage, metadata_by_path, benchmark):
    cases = coverage["cases"]
    lines = ["## 全题库三组覆盖", "",
        f"范围：case_map.json 中全部 {len(cases)} 个活跃案例，共 {len(cases) * len(ARMS)} 个 case-arm 槽位。",
        "完成包含正确、错误和 needs_more_evidence（弃答）；截断、无最终判断、运行错误和运行中均不算完成。"
        "错误答案和有效弃答保留，不因结果不理想而补跑。", "",
        "| 实验组 | 已完成 | 从未运行 | 失败/未完成 | 运行中 | 未完成合计 |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        count = coverage["totals"][arm]
        remaining = len(cases) - count["completed"]
        lines.append(f"| {arm} | {count['completed']} | {count['missing']} | "
                     f"{count['failed_or_unfinished']} | {count['running']} | {remaining} |")
    selected = [slot["selected"] for case in cases for slot in case["arms"].values() if slot["selected"]]
    uncaptured = sum(not _raw_capture_complete(row, benchmark) for row in selected)
    lines += ["", "完成记录按最早的 created_at 选取；缺少日期的旧记录优先，并以迁移前路径稳定排序。"
        "不按正确性或置信度筛选；下方索引保留当前题库的全部尝试。",
        "本页是覆盖清单，不是统一协议的准确率对照：历史记录包含 OpenRouter 的 `z-ai/glm-5.3-flash` "
        "与 Fireworks 的 `accounts/fireworks/models/glm-5p3`，模型、推理配置、预算和运行器版本可能不同。",
        f"选用记录中 {uncaptured} 条没有可确认的完整原始 API capture；历史缺失保留披露，不为补日志重跑。"
        "补跑前审计有 16 条已完成旧记录属于这种情况。", "",
        "| Case | 题库 | single_call | solo | debate |", "|---|---|---|---|---|"]
    labels = {"missing": "缺失", "failed_or_unfinished": "失败/未完成", "running": "运行中"}
    outcomes = {"correct": "正确", "wrong_verdict": "错误", "abstention": "弃答"}
    for case in cases:
        cells = []
        for arm in ARMS:
            slot = case["arms"][arm]
            row = slot["selected"]
            if row:
                cells.append(f"[{row['trial']}]({_target(row, benchmark)}) · {row['verdict']} / "
                             f"{outcomes.get(row['outcome'], row['outcome'])}")
            else:
                cells.append(labels[slot["status"]])
        lines.append(f"| {case['case']} | {case['collection']} | " + " | ".join(cells) + " |")
    return lines


def main():
    records = list(iter_trace_records(benchmark_dir=BENCHMARK))
    report=build_report(records=records, benchmark_dir=BENCHMARK)
    rows=[row for group in report['arms'].values() for row in group['per_case'].values()
          if row['path'].startswith('traces_glm/')]
    metadata_by_path = {str(record['path'].relative_to(BENCHMARK)): record.get('metadata') or {}
                        for record in records}
    details = load_registry(BENCHMARK).get('case_details', {})
    if not details:
        raise ValueError('case_map.json has no complete case_details registry')
    rows = [row for row in rows if is_active_case(details.get(row['case'], {}))]
    coverage = build_coverage(rows, details, metadata_by_path)
    lines=['# GLM trace index','',
        'Current benchmark GLM runs are listed together by case, arm and trial. API provenance is retained in trace metadata. '
        'Historical tool traces do not have raw API payloads; new runs include `llm_calls/`.', '',
        'See [案例总索引](../CASE_INDEX.md) for numeric ranges, original IDs and source kernels. '
        'Historical trace payloads keep their original IDs; the current directory and registry determine the case.', '']
    lines += ['每个 case/arm 下按历史时间顺序编号为 `r1`、`r2`……；不同案例的同名 rN 不代表同一实验批次。'
              '原始批次保存在 `trace_meta.json` 的 `original_trial`，统计继续按批次、模型和预算配置分组。', '']
    archived = [case for case, detail in details.items() if not is_active_case(detail)]
    if archived:
        archive = BENCHMARK / 'archive/numerical_pilot/traces_glm'
        archived_trials = sum(1 for case in archived for _ in (archive / case).glob('*/*/trace_meta.json'))
        lines += [f'早期 pilot 的 {len(archived)} 题、{archived_trials} 条 GLM trials 已移入'
                  '[归档](../archive/numerical_pilot/README.md)，不计入下方当前题库覆盖。'
                  '历史记录保留；归档不表示重跑或改判。', '']
    lines += curation_lines(details, trace_prefix="")
    lines += _coverage_lines(coverage, metadata_by_path, BENCHMARK)
    lines += ['', '## 全部历史 trials', '']
    current_collection = None
    for r in sorted(rows,key=lambda x:(case_sort_key(x['case']),x['arm'],trial_sort_key(x['trial']))):
        collection = case_collection(details.get(r['case'], {'dataset': r['dataset']}))
        if collection != current_collection:
            current_collection = collection
            lines += ['', '### ' + DATASET_LABELS.get(collection, collection), '',
                      '| Dataset | Case | Arm | Trial | Model / provider | Status | Verdict / outcome | Trace |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- |']
        status = metadata_by_path.get(r['path'], {}).get('status', 'legacy')
        lines.append(f"| {r['dataset']} | {r['case']} | {r['arm']} | {r['trial']} | "
                     f"{r.get('model')} / {r.get('provider')} | {status} | "
                     f"{r.get('verdict')} / {r['outcome']} | [open]({_target(r, BENCHMARK)}) |")
    lines+=['',f'{len(rows)} recorded GLM trials. See [trace format](../TRACES.md) for provenance and capture limits.','']
    (BENCHMARK/'traces_glm/INDEX.md').write_text('\n'.join(lines))
    print(f'Indexed {len(rows)} GLM trials')

if __name__=='__main__':main()
