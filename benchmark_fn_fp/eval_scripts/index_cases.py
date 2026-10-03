"""Generate the global numeric case index from the private case registry; no API calls."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from case_registry import case_collection, case_sort_key, dataset_root, is_active_case, load_registry
from datasets import cases_dir

ROOT = Path(__file__).resolve().parents[1]
DATASET_LABELS = {
    "benchmark_fn_fp": "原始 FN/FP benchmark",
    "correlation_pair": "无工具单次调用 vs 工具：量化误差配对",
    "numerical_challenges": "数值精度：实际误差是否超标",
    "evidence_challenges": "参考实现与测试覆盖：验证结论是否可靠",
    "numerical_pilot": "早期 24 例数值 pilot（独立旧实验）",
    "single_call_vs_tools_challenges": "无工具单次调用 vs 带工具验证",
    "solo_vs_debate_challenges": "Solo vs 多角色 debate",
    "real_kernel_challenges": "专门构造的真实 kernel 验证挑战（单独统计）",
}
CURATION_LABELS = {
    "tool_gain_observed": "成功案例：已观察到工具流程增益，单独保留",
    "ordinary_candidate": "未体现增益，保留待归普通集合",
}


def curation_lines(details, *, trace_prefix="traces_glm/"):
    groups = {status: sorted((case for case, row in details.items()
                             if row.get("curation_status") == status and is_active_case(row)),
                            key=case_sort_key) for status in CURATION_LABELS}
    if not any(groups.values()):
        return []
    lines = ["## 实验后的保留分组", "",
             "这是后续整理状态，原始题库来源与全部实验统计保留。"
             "单次观察到增益不代表稳定优势；待归普通集合的案例目前仍保留在原来源组。", "",
             "| 分组 | Case / traces |", "|---|---|"]
    for status, cases in groups.items():
        if cases:
            links = "、".join(f"[{case}]({trace_prefix}{case}/)" for case in cases)
            lines.append(f"| {CURATION_LABELS[status]} | {links} |")
    return lines + [""]


def _relative(value):
    value = str(value)
    return value.removeprefix("benchmark_fn_fp/")


def _cell(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


def main():
    registry = load_registry(ROOT)
    details = registry.get("case_details", {})
    if not details:
        raise ValueError("case_map.json has no canonical case_details registry")
    groups = defaultdict(list)
    archived = []
    for case, row in sorted(details.items(), key=lambda item: case_sort_key(item[0])):
        if is_active_case(row):
            groups[case_collection(row)].append(case)
        else:
            archived.append(case)
    lines = ["# Benchmark 案例总索引", "",
        "所有题库和 traces 共用 `case_数字` 编号。同一编号始终指向同一道题；来源记录在 case_map.json，部分题库共用统一公开目录。",
        "本页是说明与答案侧索引，不作为模型输入。编号描述实验来源，不保证某种方法一定获胜。", "",
        f"当前 {len(details) - len(archived)} 题；另有 {len(archived)} 题已归档。归档编号保留，不复用、不进入默认实验。", "",
        "| 编号范围 | 数量 | 测什么 | 说明与题目 |", "|---|---:|---|---|"]
    for dataset, cases in groups.items():
        path = cases_dir(ROOT.parent, dataset).relative_to(ROOT).as_posix()
        guide = dataset_root(ROOT, dataset).relative_to(ROOT).as_posix()
        guide = "README.md" if dataset == "benchmark_fn_fp" else f"{guide}/README.md"
        lines.append(f"| `{cases[0]}`–`{cases[-1]}` | {len(cases)} | "
                     f"{DATASET_LABELS.get(dataset, dataset)} | [说明]({guide}) · [题目]({path}/) |")
    if archived:
        lines += ["", f"已归档：`{archived[0]}`–`{archived[-1]}`，共 {len(archived)} 题。"
                  "[早期数值 pilot](archive/numerical_pilot/README.md) 保留题目、真值、旧 Opus 结果和 72 条 GLM 实验。"]
    lines += [""] + curation_lines(details)
    retired = ", ".join(f"`{name}`" for name in registry.get("retired_ids", [])) or "无"
    lines += ["", f"已退休且不复用的编号：{retired}。下一可用编号从 {registry.get('next_case_number', '注册表')} 开始。",
        "`real_kernel_challenges/` 保存真实源码衍生的专门构造挑战；标签针对固定实现与合同，结果单独统计。", "",
        "每道题的完整对应关系如下。旧编号仅用于查阅历史记录，今后运行请使用新编号。", "",
        "| 统一编号 | 状态 | 旧编号 | 题库 / 来源机制 | 公开题目 | GLM traces |",
        "|---|---|---|---|---|---|"]
    for case, row in sorted(details.items(), key=lambda item: case_sort_key(item[0])):
        public = _relative(row["public_dir"])
        if not (ROOT / public / "kernel.py").is_file():
            raise FileNotFoundError(f"Missing public kernel for {case}: {public}")
        mechanism = row.get("title") or row.get("mechanism") or row.get("source_name") or row.get("category")
        source = f"{case_collection(row)} / {_cell(mechanism)}"
        state = "当前" if is_active_case(row) else "已归档"
        if row.get("curation_status") in CURATION_LABELS:
            state += " · " + CURATION_LABELS[row["curation_status"]]
        trace_root = "traces_glm" if is_active_case(row) else "archive/numerical_pilot/traces_glm"
        traces = f"[记录]({trace_root}/{case}/)" if (ROOT / trace_root / case).is_dir() else "无 GLM 记录"
        lines.append(f"| `{case}` | {state} | `{row.get('previous_id', case)}` | {source} | "
                     f"[kernel]({public}/kernel.py) · [合同]({public}/problem.txt) | {traces} |")
    lines += ["", "## 如何读 traces", "",
        "`traces_glm/<case>/<arm>/<trial>/` 中，`single_call` 是无工具单次调用，`solo` 是单 agent 加工具，"
        "`debate` 是多角色加工具。各组批次目录统一为 `r1`、`r2` 等运行序号；原批次名保留在 "
        "`trace_meta.json` 的 `original_trial` 中。同名 `r1` 不代表模型、预算或实验配置相同。",
        "[GLM 逐次运行索引](traces_glm/INDEX.md) 保留不同配置、复测与失败尝试；题库编号相同不代表三组都已跑齐。",
        "目录已统一编号；历史请求、回复和 probe 代码保留原始字节；目录和 metadata 的身份、迁移字段按索引更新。"
        "原始内容中的旧编号不是另一道题，读取程序根据当前目录与本注册表核对身份。",
        "case_36–case_37 已并入单次调用与工具比较组；其 Opus 和 GLM 原始记录仍在 traces_opus5 与 traces_glm，历史来源字段不改写。",
        "早期数值 pilot（case_82–case_105）和它的 72 条 GLM 记录移入 archive/numerical_pilot，已从当前 GLM 索引与默认评分移出；"
        "过去 104 题的历史报告保留原结论。", "",
        "## 文件清理说明", "",
        "两组 challenges 各保留一份 README 和 `private_data/`。后者保存答案、搜索过程和 GPU 真值校验；这些是实验可复现性依据。",
        "当前题目统一在 `triton_eval_cases/`，构造、校验与报告程序统一在 `eval_scripts/` 的对应目录；pilot 的公开题目留在归档中。",
        "零散实验说明已合并进各组 README；旧逻辑数据集名只用于兼容历史 traces 和旧命令。可从原始 traces 重建的 scoreboard JSON 已清理；报告脚本默认更新 README 中的结果表，"
        "如需机器可读汇总可显式请求 JSON。原始模型调用与 GPU 记录不属于可重建汇总。", "",
        "本页由 `eval_scripts/index_cases.py` 根据 `case_map.json` 生成。", ""]
    (ROOT / "CASE_INDEX.md").write_text("\n".join(lines))
    print(f"Indexed {len(details)} globally numbered cases")


if __name__ == "__main__":
    main()
