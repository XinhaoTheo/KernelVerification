"""Generate the global numeric case index from the private case registry; no API calls."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from case_registry import DATASET_DIRECTORIES, case_sort_key, dataset_root, load_registry
from datasets import cases_dir

ROOT = Path(__file__).resolve().parents[1]
DATASET_LABELS = {
    "benchmark_fn_fp": "原始 FN/FP benchmark",
    "correlation_pair": "无工具单次调用 vs 工具：量化误差配对",
    "numerical_challenges": "数值精度：实际误差是否超标",
    "evidence_challenges": "参考实现与测试覆盖：验证结论是否可靠",
    "numerical_pilot": "早期 24 例数值 pilot（独立旧实验）",
}


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
    for case, row in sorted(details.items(), key=lambda item: case_sort_key(item[0])):
        groups[row["dataset"]].append(case)
    lines = ["# Benchmark 案例总索引", "",
        "所有题库和 traces 共用 `case_数字` 编号。同一编号始终指向同一道题；来源记录在 case_map.json，部分题库共用统一公开目录。",
        "本页是说明与答案侧索引，不作为模型输入。编号描述实验来源，不保证某种方法一定获胜。", "",
        "| 编号范围 | 数量 | 测什么 | 说明与题目 |", "|---|---:|---|---|"]
    for dataset, cases in groups.items():
        path = cases_dir(ROOT.parent, dataset).relative_to(ROOT).as_posix()
        guide = dataset_root(ROOT, dataset).relative_to(ROOT).as_posix()
        guide = "README.md" if dataset == "benchmark_fn_fp" else f"{guide}/README.md"
        lines.append(f"| `{cases[0]}`–`{cases[-1]}` | {len(cases)} | "
                     f"{DATASET_LABELS.get(dataset, dataset)} | [说明]({guide}) · [题目]({path}/) |")
    retired = ", ".join(f"`{name}`" for name in registry.get("retired_ids", [])) or "无"
    lines += ["", f"已退休且不复用的编号：{retired}。下一可用编号从 {registry.get('next_case_number', '注册表')} 开始。",
        "`real_kernel_challenges/` 目前是下一轮真实 kernel 的计划，尚无已分配案例。", "",
        "每道题的完整对应关系如下。旧编号仅用于查阅历史记录，今后运行请使用新编号。", "",
        "| 统一编号 | 旧编号 | 题库 / 来源机制 | 公开题目 | GLM traces |",
        "|---|---|---|---|---|"]
    for case, row in sorted(details.items(), key=lambda item: case_sort_key(item[0])):
        public = _relative(row["public_dir"])
        if not (ROOT / public / "kernel.py").is_file():
            raise FileNotFoundError(f"Missing public kernel for {case}: {public}")
        mechanism = row.get("title") or row.get("mechanism") or row.get("source_name") or row.get("category")
        source = f"{DATASET_DIRECTORIES.get(row['dataset'], row['dataset'])} / {_cell(mechanism)}"
        traces = f"[记录](traces_glm/{case}/)" if (ROOT / "traces_glm" / case).is_dir() else "无 GLM 记录"
        lines.append(f"| `{case}` | `{row.get('previous_id', case)}` | {source} | "
                     f"[kernel]({public}/kernel.py) · [合同]({public}/problem.txt) | {traces} |")
    lines += ["", "## 如何读 traces", "",
        "`traces_glm/<case>/<arm>/<trial>/` 中，`single_call` 是无工具单次调用，`solo` 是单 agent 加工具，"
        "`debate` 是多角色加工具。各组批次目录统一为 `r1`、`r2` 等运行序号；原批次名保留在 "
        "`trace_meta.json` 的 `original_trial` 中。同名 `r1` 不代表模型、预算或实验配置相同。",
        "[GLM 逐次运行索引](traces_glm/INDEX.md) 保留不同配置、复测与失败尝试；题库编号相同不代表三组都已跑齐。",
        "目录已统一编号；历史请求、回复和 probe 代码保留原始字节；目录和 metadata 的身份、迁移字段按索引更新。"
        "原始内容中的旧编号不是另一道题，读取程序根据当前目录与本注册表核对身份。",
        "Correlation pair（case_36–case_37）的 Opus 和 GLM 原始记录分别统一存入 traces_opus5 与 traces_glm，"
        "题库目录不再重复保存 traces。",
        "早期数值 pilot（case_82–case_105）的旧 Opus 单次调用结果仍留在 numerical_pilot 中；"
        "新增 GLM 记录统一放入 traces_glm，对应覆盖情况以逐次运行索引为准。", "",
        "## 文件清理说明", "",
        "量化误差配对、数值精度、参考与覆盖三组数据各保留一份 README 和 `private_data/`。后者保存答案、搜索过程和 GPU 真值校验；这些是实验可复现性依据。",
        "这三组的公开题目统一在 `triton_eval_cases/`，构造、校验与报告程序统一在 `eval_scripts/` 的对应目录。",
        "零散实验说明已合并进各组 README；旧逻辑数据集名只用于兼容历史 traces 和旧命令。可从原始 traces 重建的 scoreboard JSON 已清理；报告脚本默认更新 README 中的结果表，"
        "如需机器可读汇总可显式请求 JSON。原始模型调用与 GPU 记录不属于可重建汇总。", "",
        "本页由 `eval_scripts/index_cases.py` 根据 `case_map.json` 生成。", ""]
    (ROOT / "CASE_INDEX.md").write_text("\n".join(lines))
    print(f"Indexed {len(details)} globally numbered cases")


if __name__ == "__main__":
    main()
