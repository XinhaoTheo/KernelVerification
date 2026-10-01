"""Render compact trace summaries and replace one explicitly marked README block.

These helpers never run models or change traces. Full report dictionaries remain
available through the individual report commands' optional JSON exports.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path


def replace_results(root: Path, name: str, content: str) -> None:
    """Preserve every byte outside one unique, ordered generated-results block."""
    path = Path(root) / "README.md"
    text = path.read_bytes().decode("utf-8")
    begin, end = f"<!-- BEGIN GENERATED {name} -->", f"<!-- END GENERATED {name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one {name} marker pair in {path}")
    first, last = text.index(begin) + len(begin), text.index(end)
    if first > last:
        raise ValueError(f"Reversed {name} markers in {path}")
    path.write_bytes((text[:first] + "\n\n" + content.strip() + "\n\n" + text[last:]).encode("utf-8"))


def _cell(value):
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def compact_results(rows: list[dict]) -> str:
    """Keep experiment/model/budget groups separate without copying raw traces."""
    grouped = defaultdict(list)
    for row in rows:
        protocol = dict(row.get("protocol") or {})
        for field in ("reasoning_effort", "max_tokens", "max_rounds", "total_output_token_budget"):
            if protocol.get(field) is None and row.get(field) is not None:
                protocol[field] = row[field]
        key = (row.get("original_trial") or row.get("trial"), row.get("provider"),
               row.get("model"), json.dumps(protocol, sort_keys=True), row.get("arm"))
        grouped[key].append(row)
    lines = [
        "| 批次 | 模型 / 提供方 | 预算 / 推理 / 轮次 | 方式 | 次数 | 正确 | 错误 | 弃答 | 截断 | 其他 | API 估算 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, attempts in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        batch, provider, model, serialized, arm = key
        config = json.loads(serialized)
        budget = (f"每次 {_cell(config.get('max_tokens'))} / 总 {_cell(config.get('total_output_token_budget'))} / "
                  f"推理 {_cell(config.get('reasoning_effort'))} / 轮次 {_cell(config.get('max_rounds'))}")
        outcomes = Counter(row.get("outcome") for row in attempts)
        named = ("correct", "wrong_verdict", "abstention", "token_limit")
        other = len(attempts) - sum(outcomes[name] for name in named)
        cost = sum(row.get("usd") or 0 for row in attempts)
        partial = any(row.get("usd") is None or row.get("usd_is_partial") for row in attempts)
        label = (model or "unknown").rsplit("/", 1)[-1]
        lines.append(f"| {_cell(batch)} | {_cell(label)} / {_cell(provider)} | {budget} | {_cell(arm)} | "
                     f"{len(attempts)} | {outcomes['correct']} | {outcomes['wrong_verdict']} | "
                     f"{outcomes['abstention']} | {outcomes['token_limit']} | {other} | ${cost:.6f}{'（不完整）' if partial else ''} |")
    unknown = sum(row.get("usd") is None for row in rows)
    partial = sum(bool(row.get("usd_is_partial")) for row in rows)
    lines += ["", f"共 {len(rows)} 次记录；API 费用估算 ${sum(row.get('usd') or 0 for row in rows):.6f}。"
              f"费用未知 {unknown} 次，费用记录不完整 {partial} 次。", "",
              "“其他”含未完成、无判定及运行失败，不算明确判断错误。费用不含 Modal GPU 和未返回的用量。"
              "按原实验批次、模型及完整配置分别统计；本地 rN 只是目录编号。unknown 表示原记录未声明该项。", "",
              "逐次记录及原始证据见 [GLM traces 索引](../traces_glm/INDEX.md) "
              "和 [Opus traces 目录](../traces_opus5/)。运行本报告的 `--json` 可将完整审核结果导出至 `private_data/reports/`。"]
    return "\n".join(lines) + "\n"
