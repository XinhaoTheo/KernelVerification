"""Derive pilot and explicit corrective cases from source, labels and traces.

Default: replace only REAL KERNEL RESULTS in the collection README.
--json: print the same report, without writing any file. No API/GPU calls.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] / "real_kernel_challenges"
sys.path.insert(0, str(ROOT.parent / "eval_scripts"))
from models import profile_for_trace
from case_registry import load_registry, remap_validation_cases
from summarize_traces import classify_outcome, read_run, read_single_call
from traces import experiment_trial, iter_trace_records, trial_sort_key

COHORTS = {"pilot": tuple(f"case_{i:02d}" for i in range(33, 39)),
           "boundary": tuple(f"case_{i:02d}" for i in range(39, 43)),
           "splitk_moe": tuple(f"case_{i:02d}" for i in range(43, 47)),
           "int4_pipeline": ("case_47", "case_48"),
           "stencil": ("case_49", "case_50"),
           "requant": ("case_51", "case_52"),
           "erf": ("case_53", "case_54"),
           "gelu": ("case_55", "case_56"),
           "silu": ("case_57", "case_58"),
           "sine": ("case_59", "case_60"),
           "hypot": ("case_61", "case_62"),
           "softplus": ("case_63", "case_64"),
           "compensated": ("case_65", "case_66"),
           "fastdiv": ("case_67", "case_68"),
           "butterfly": ("case_69", "case_70"),
           "packed_argmax": ("case_71", "case_72"),
           "atan": ("case_73", "case_74"),
           "rounding": ("case_75", "case_76"),
           "ntt": ("case_77", "case_78"),
           "crc": ("case_79", "case_80"),
           "complex": ("case_81", "case_82"),
           "requant_compose": ("case_83", "case_84"),
           "affine_scan": ("case_85", "case_86"),
           "winograd": ("case_87", "case_88"),
           "realfft": ("case_89", "case_90"),
           "exp2": ("case_91", "case_92"),
           "log1p": ("case_93", "case_94"),
           "aes": ("case_95", "case_96"),
           "oct20": tuple(f"case_{i}" for i in range(91, 97)),
           "expm1": ("case_97", "case_98"),
           "asinh": ("case_99", "case_100"),
           "cbrt": ("case_101", "case_102"),
           "oct20_followup": tuple(f"case_{i}" for i in range(97, 103))}
SCOPED_TOOL_CASES = frozenset(case for cohort, cases in COHORTS.items()
                             if cohort != "pilot" for case in cases)
CASES = COHORTS["pilot"]
ARMS = ("single_call", "solo", "debate")
MODEL = "accounts/fireworks/models/glm-5p3"
BEGIN = "<!-- BEGIN REAL KERNEL RESULTS -->"
END = "<!-- END REAL KERNEL RESULTS -->"


def _markers(cohort):
    suffix = {"pilot": "", "boundary": " BOUNDARY", "splitk_moe": " SPLITK MOE",
              "int4_pipeline": " INT4 PIPELINE", "stencil": " STENCIL",
              "requant": " REQUANT", "erf": " ERF", "gelu": " GELU", "silu": " SILU", "sine": " SINE",
              "hypot": " HYPOT", "softplus": " SOFTPLUS", "compensated": " COMPENSATED",
              "fastdiv": " FASTDIV", "butterfly": " BUTTERFLY", "packed_argmax": " PACKED ARGMAX",
              "atan": " ATAN", "rounding": " ROUNDING", "ntt": " NTT",
              "crc": " CRC", "complex": " COMPLEX", "requant_compose": " REQUANT COMPOSE",
              "affine_scan": " AFFINE SCAN", "winograd": " WINOGRAD", "realfft": " REALFFT",
              "exp2": " EXP2", "log1p": " LOG1P", "aes": " AES",
              "oct20": " OCTOBER20", "expm1": " EXPM1", "asinh": " ASINH",
              "cbrt": " CBRT", "oct20_followup": " OCTOBER20 FOLLOWUP"}[cohort]
    return (f"<!-- BEGIN REAL KERNEL{suffix} RESULTS -->",
            f"<!-- END REAL KERNEL{suffix} RESULTS -->")


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _frozen(root, cases=CASES):
    labels = remap_validation_cases(
        read(root / "private_data/validation_gpu.json").get("cases", {}),
        "real_kernel_challenges", load_registry(root.parent))
    public = {}
    for case in cases:
        if labels.get(case, {}).get("ground_truth") not in {"trust", "reject"}:
            raise ValueError(f"Missing independently frozen label: {case}")
        public[case] = {}
        for key, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            path = root.parent / "triton_eval_cases" / case / filename
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != labels[case].get(f"{key}_sha256"):
                raise ValueError(f"Frozen public file changed: {path}")
            public[case][key] = content.decode("utf-8")
    return {case: labels[case] for case in cases}, public


def _check_task(request, public, arm):
    messages = request.get("messages") or []
    if arm == "single_call":
        return any(isinstance(m.get("content"), str) and public["kernel"] in m["content"]
                   and public["problem"] in m["content"] for m in messages)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or "=== Current Run State ===\n" not in content:
            continue
        try:
            state, _ = json.JSONDecoder().raw_decode(content.split("=== Current Run State ===\n", 1)[1])
            artifact = state["artifact"]
            numbered = artifact["kernel_code"].splitlines()
            recovered = []
            for number, line in enumerate(numbered, 1):
                prefix, value = line.split(": ", 1)
                if prefix != str(number):
                    return False
                recovered.append(value)
            return recovered == public["kernel"].splitlines() and artifact["problem_text"] == public["problem"]
        except (ValueError, KeyError, TypeError):
            return False
    return False


def _attempt(record, labels, public, benchmark):
    path = Path(record["path"])
    meta = record["metadata"]
    case, arm = record["case"], record["arm"]
    for field in ("kernel_sha256", "problem_sha256"):
        if meta.get(field) != labels[case][field]:
            raise ValueError(f"Trace source hash mismatch: {path}/{field}")
    usage_record = read(path / "usage.json")
    pricing = meta.get("pricing_snapshot") or usage_record.get("pricing_snapshot")
    profile = profile_for_trace(record.get("model") or "unknown", {**meta, **({"pricing_snapshot": pricing} if pricing else {})})
    row = (read_single_call(path, profile) if arm == "single_call"
           else read_run(path / "run.json", profile))
    run = read(path / "run.json")
    if run:
        row["probes"] = sum(event.get("tool") in {"run_claim_probe", "run_python_probe"}
                            for event in run.get("tool_events", []))
    if run.get("artifact"):
        for field, expected in (("kernel_code", public[case]["kernel"]),
                                ("problem_text", public[case]["problem"])):
            if run["artifact"].get(field) != expected:
                raise ValueError(f"Trace artifact changed: {path}/{field}")
    issues = []
    if record.get("dataset") != "real_kernel_challenges":
        issues.append("trace dataset identity differs from real_kernel_challenges")
    if record.get("model") != MODEL or record.get("provider") != "fireworks":
        issues.append("model/provider differs from Fireworks GLM-5p3")
    if meta.get("reasoning_effort") != "low":
        issues.append("reasoning_effort is not low")
    budget = meta.get("max_tokens") if arm == "single_call" else meta.get("total_output_token_budget")
    if budget != 32768:
        issues.append("total output budget differs from 32768")
    if case in SCOPED_TOOL_CASES and arm != "single_call":
        if meta.get("max_tokens") != 8192:
            issues.append("boundary tool turn limit differs from 8192")
        if meta.get("require_claim_scope_fields") is not True:
            issues.append("boundary claim schema setting missing")
        if bool(meta.get("debate_budget_closeout")) != (arm == "debate"):
            issues.append("boundary debate closeout setting mismatch")
        visibility = read(path / "artifact_visibility.json")
        visible_cases = visibility.get("visible_cases")
        aliases = set(record.get("case_aliases", (case,)))
        if (meta.get("case_visibility") != "current_case_only"
                or not isinstance(visible_cases, list) or len(visible_cases) != 1
                or visible_cases[0] not in aliases
                or visibility.get("previous_trace_root_removed") is not True):
            issues.append("current-case-only worker isolation missing or mismatched")
        for kind, filename in (("kernel", "kernel.py"), ("problem", "problem.txt")):
            if visibility.get("public_files_sha256", {}).get(filename) != labels[case][f"{kind}_sha256"]:
                issues.append("isolated worker public source fingerprint mismatch")
    if row.get("output_tokens", 0) > 32768:
        issues.append("recorded output exceeds the total 32768 token budget")
    if meta.get("raw_api_capture") is not True:
        issues.append("raw_api_capture is not enabled")
    calls = sorted(p for p in (path / "llm_calls").glob("*") if p.is_dir())
    captures = []
    for call in calls:
        request, response, detail = (read(call / name) for name in ("request.json", "response.json", "metadata.json"))
        usage = response.get("usage") or {}
        # A saved provider response can be followed by a local parsing error.
        # Capture completeness describes the evidence, not execution success;
        # retain the call status and classify the failed run separately below.
        complete = bool(request and response
                        and all(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens")))
        captures.append({"call": call.name, "status": detail.get("status"), "complete": complete})
        if request:
            if request.get("model") != MODEL or request.get("reasoning_effort") != "low":
                issues.append(f"request config mismatch: {call.name}")
            if not _check_task(request, public[case], arm):
                raise ValueError(f"Incomplete or changed public task in API request: {call}")
            if not isinstance(request.get("max_tokens"), int) or not 0 < request["max_tokens"] <= 32768:
                issues.append(f"request output limit mismatch: {call.name}")
            elif case in SCOPED_TOOL_CASES and arm != "single_call" and request["max_tokens"] > 8192:
                issues.append(f"boundary request exceeds per-turn 8192 cap: {call.name}")
    if not calls or not all(call["complete"] for call in captures):
        issues.append("raw API capture incomplete")
    runtime = read(path / "runtime.json")
    fingerprint = runtime.get("verifier_sha256")
    if not fingerprint or fingerprint != meta.get("verifier_sha256"):
        issues.append("runtime verifier fingerprint missing or mismatched")
    error_path = path / "runner_error.txt"
    runner_error = bool(error_path.exists() and error_path.read_text().strip())
    status = meta.get("status", "unknown")
    # A reserved local folder has no visibility into an in-flight remote SDK
    # call. Missing returned usage is unknown cost, not zero expenditure.
    if not any(call["complete"] for call in captures) and status != "completed":
        infrastructure = read(path / "infrastructure_failure.json")
        if infrastructure.get("api_calls") == 0 and infrastructure.get("api_cost_usd") == 0:
            row["usd"] = 0.0
        else:
            row["usd"] = None
            row["usd_is_partial"] = True
    outcome = classify_outcome(row, labels[case]["ground_truth"])
    if runner_error or status in {"error", "failed", "aborted", "cancelled", "timeout"}:
        # Keep the token_limit category for a length-limited response without
        # a verdict, even if unused total/closeout budget remains after an error.
        if outcome != "token_limit":
            outcome = "error"
    elif status != "completed":
        outcome = "pending"
    row.update(case=case, arm=arm, trial=record["trial"], batch=experiment_trial(record),
               path=str(path.relative_to(benchmark)), truth=labels[case]["ground_truth"],
               status=status, outcome=outcome, audit_issues=sorted(set(issues)),
               verifier_sha256=fingerprint, captures=captures, runtime=runtime,
               pricing_source="trace_snapshot" if pricing else "model_profile_fallback",
               elapsed_s=usage_record.get("elapsed_s"),
               link=str((path / ("transcript.md" if (path / "transcript.md").exists() else "trace_meta.json")).relative_to(benchmark)),
               reason=(row.get("reason") or (run.get("verdict") or {}).get("reason")),
               protocol={k: meta.get(k) for k in ("max_tokens", "total_output_token_budget",
                         "reasoning_effort", "max_rounds", "debate_budget_closeout",
                         "require_claim_scope_fields", "normal_turn_output_cap",
                         "closeout_reserve", "closeout_stage_caps")})
    return row


def _counts(rows):
    counts = Counter(row["outcome"] for row in rows)
    priced = [row["usd"] for row in rows if row.get("usd") is not None]
    return {"attempts": len(rows), "outcomes": dict(counts),
            "audited_correct": sum(row["outcome"] == "correct" and not row["audit_issues"] for row in rows),
            "audit_issue_attempts": sum(bool(row["audit_issues"]) for row in rows),
            "recorded_api_estimate_usd": round(sum(priced), 6) if priced else None,
            "unknown_cost_attempts": sum(row.get("usd") is None for row in rows),
            "partial_cost_attempts": sum(bool(row.get("usd_is_partial")) for row in rows),
            "probes": sum(row.get("probes", 0) for row in rows)}


def derive(root=ROOT, *, batch=None, cohort="pilot"):
    cases = COHORTS[cohort]
    root = Path(root)
    labels, public = _frozen(root, cases)
    records = [r for r in iter_trace_records(benchmark_dir=root.parent)
               if r["case"] in cases and r["arm"] in ARMS
               and Path(r["path"]).is_relative_to(root.parent / "traces_glm")
               and (batch is None or experiment_trial(r) == batch)]
    rows = [_attempt(record, labels, public, root.parent) for record in records]
    groups = defaultdict(list)
    for row in rows:
        # rN is the local replicate number for this pilot and its explicit repairs.
        # The two CLIs can generate separate original_trial submission ids even
        # when all three arms belong to the same r1 experiment.
        groups[row["trial"]].append(row)
    for members in groups.values():
        fingerprints = {row["verifier_sha256"] for row in members if row["verifier_sha256"]}
        if len(fingerprints) > 1:
            for row in members:
                row["audit_issues"].append("different verifier runtime fingerprints within this rN cohort")
    rows.sort(key=lambda row: (cases.index(row["case"]), ARMS.index(row["arm"]), trial_sort_key(row["trial"])))
    occupied = {(row["case"], row["arm"]) for row in rows}
    missing = [{"case": case, "arm": arm} for case in cases for arm in ARMS if (case, arm) not in occupied]
    model_rows = [row for row in rows if row["captures"]]
    return {"dataset": "real_kernel_challenges", "cases": labels, "case_ids": list(cases), "cohort": cohort, "selected_batch": batch,
            "model": MODEL, "provider": "fireworks", "expected_slots": len(cases) * len(ARMS),
            "occupied_slots": len(occupied), "missing_slots": missing, "attempts": rows,
            "totals": _counts(rows), "arms": {arm: _counts([r for r in rows if r["arm"] == arm]) for arm in ARMS},
            "model_attempts": len(model_rows),
            "model_arms": {arm: _counts([r for r in model_rows if r["arm"] == arm]) for arm in ARMS}}


def _money(value):
    return "费用未知" if value is None else f"${value:.4f}"


def render(report):
    rows = report["attempts"]
    cases = report["case_ids"]
    begin, end = _markers(report["cohort"])
    totals = report["totals"]
    outcomes = totals["outcomes"]
    lines = [begin, "", "以下结果直接由冻结标签和原始 traces 生成；保留所有尝试，不挑选最好结果。",
             "正确实现的标签依据合同、实现分析和独立验证；测试通过不等于证明整个输入域。", "",
             "| Case | 固定标签 | 无工具单次调用 | Solo＋工具 | Debate＋工具 |",
             "|---|---|---|---|---|"]
    names = {"correct": "正确", "wrong_verdict": "错误", "abstention": "弃答", "token_limit": "token 上限",
             "no_verdict": "无最终答案", "error": "运行失败", "pending": "未完成"}
    for case in cases:
        cells = [case, report["cases"][case]["ground_truth"]]
        for arm in ARMS:
            trials = [r for r in rows if r["case"] == case and r["arm"] == arm]
            cells.append("<br>".join(
                f"[{r['trial']}](../{r['link']})：{r.get('verdict') or '—'} / {names.get(r['outcome'], r['outcome'])}"
                f"；probe {r.get('probes', 0)}；{_money(r.get('usd'))}"
                + ("；审计未通过" if r["audit_issues"] else "") for r in trials) or "缺失")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Arm | 尝试数 | 记录完整且判断正确 | 错误判断 | 弃答 | 输出截断／上限 | 其他运行失败 | Probe | API 估算 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        count = report["arms"][arm]
        values = count["outcomes"]
        lines.append(f"| {arm} | {count['attempts']} | {count['audited_correct']} | {values.get('wrong_verdict', 0)} | "
                     f"{values.get('abstention', 0)} | {values.get('token_limit', 0)} | {values.get('error', 0)} | {count['probes']} | {_money(count['recorded_api_estimate_usd'])} |")
    lines += ["", f"目标 **{report['expected_slots']} 个 case×arm 槽位**：已有记录 {report['occupied_slots']}，缺失 {len(report['missing_slots'])}。"
              f"共有 {totals['attempts']} 次尝试；按互斥结果类别计：其他运行失败 {outcomes.get('error', 0)}，弃答 {outcomes.get('abstention', 0)}，"
              f"其他无最终答案 {outcomes.get('no_verdict', 0)}，输出截断／上限 {outcomes.get('token_limit', 0)}，未完成 {outcomes.get('pending', 0)}。",
              f"全部尝试的已记录 API 费用估算：**{_money(totals['recorded_api_estimate_usd'])}**；"
              f"费用未知 {totals['unknown_cost_attempts']} 次、usage 不完整 {totals['partial_cost_attempts']} 次。"
              "优先使用 trace 定价记录，缺失时采用模型配置费率；非实际账单。GPU 账单未知，未计入；未返回 usage 的请求也未计入。", ""]
    lines += [f"实际启动模型的评测有 **{report['model_attempts']} 次**。按每组实际评测次数计正确率（输出截断／上限仍计入分母）：",
              "", "| Arm | 判断正确 / 实际评测 | 正确率 |", "|---|---:|---:|"]
    for arm in ARMS:
        count = report["model_arms"][arm]
        rate = f"{count['audited_correct'] / count['attempts']:.1%}" if count["attempts"] else "—"
        lines.append(f"| {arm} | {count['audited_correct']} / {count['attempts']} | {rate} |")
    if report["cohort"] == "pilot":
        lines += ["", "35/36 的四次 r1 工具尝试在 Modal 启动阶段失败，尚未调用模型；"
                  "其 API 调用数和费用均确认是 0，因此保留故障记录、另用 r2 执行，"
                  "不作为四次模型答错。下列这些 r1 的 API/runtime 缺项来自进程未启动。", ""]
    batches = sorted({row["batch"] for row in rows})
    if batches:
        lines += ["", "原始提交批次：" + "、".join(f"`{batch}`" for batch in batches) + "。按 rN 核验共同 verifier 指纹；保留原始提交身份。", ""]
    issues = [r for r in rows if r["audit_issues"]]
    if issues:
        lines += ["", "记录一致性审计问题：", ""]
        for row in issues:
            lines.append(f"- `{row['case']}/{row['arm']}/{row['trial']}`：" + "; ".join(row["audit_issues"]))
    elif rows:
        lines.append("已核对公开文件与 trace 哈希、每次请求的完整公开材料、模型/预算、原始 API 捕获及同批 verifier 指纹。")
    for case, label in report["cases"].items():
        correction = label.get("label_correction")
        if correction:
            lines += ["", f"`{case}` 真值修正：初始 {correction['previous_ground_truth']} 已撤回，"
                       f"当前固定标签为 {label['ground_truth']}。{correction['reason']} "
                       "原源码、合同与所有模型 traces 均保持原样；初始资格结果作为历史记录保留。"]
    lines += ["", "完整记录不等于所有 probe 都合法或构成充分证据；另见下方人工证据审计。", "",
              f"样本量仅 {len(cases)} 例。" + ("其中 37–38 是补审后新增的修复实现；" if report["cohort"] == "pilot" else "") +
              "单次胜负不能证明普遍工具收益或 debate 优势。", "", end]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--batch", help="Optional exact original_trial; default retains all attempts in the cohort")
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="pilot")
    args = parser.parse_args(argv)
    report = derive(batch=args.batch, cohort=args.cohort)
    begin, end = _markers(args.cohort)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    path = ROOT / "README.md"
    text = path.read_text()
    if text.count(begin) != 1 or text.count(end) != 1 or text.index(begin) > text.index(end):
        raise ValueError("README must contain exactly one ordered REAL KERNEL RESULTS marker pair")
    start, finish = text.index(begin), text.index(end) + len(end)
    path.write_text(text[:start] + render(report) + text[finish:])
    print(f"Refreshed {path}: {report['totals']['attempts']} attempts, {len(report['missing_slots'])} missing slots")


if __name__ == "__main__":
    main()
