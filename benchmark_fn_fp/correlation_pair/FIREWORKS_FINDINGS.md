# Fireworks case_36/case_37 三组对照：四角色没有增加正确答案，但增加了机制解释

使用 `FIREWORKS_API_KEY`，同一模型 `accounts/fireworks/models/glm-5p3`，共六次完整试验（每格一次）。2026-09-22 晚启动，2026-09-23 凌晨结束。原 Opus 记录保留在 `traces/`，本次独立保存在 `traces_fireworks/`。未修改输入、源码、合同、阈值或标签，也未进行语义重试挑选结果。

| 配置 | case_36：误差 22.4116%，应 reject | case_37：误差 3.3647%，应 trust | API 估算合计 | 模型调用数 | GPU probe 数 |
| --- | --- | --- | ---: | ---: | ---: |
| 无工具单次调用 | reject，正确 | 32K 输出预算耗尽，无最终 verdict | $0.0504 | 2 | 0 |
| solo + tools | reject，正确 | trust，正确 | $0.0954 | 8 | 3 |
| 四角色 debate + tools | reject，正确 | trust，正确 | $0.3224 | 13 | 5 |

API 估算总计 **$0.4682**，根据项目价格配置和返回的 token usage 计算，不是实际账单，且不含 Modal GPU 费用或未返回用量的 HTTP 重试。四角色费用约为 solo 的 **3.38 倍**，但两者正确答案数都是 **2/2**。

## 是否发现四角色的额外价值

**没有观察到额外的正确率收益或对 solo 错误的修复。** Solo 自己就测量了两例的真实误差，并在 case_37 上做了第二套 FP64 reference、输入未修改检查、重复执行和两路误差分解；它不需要另一个角色提醒，就发现了误差抵消。

**四角色在 case_37 上确实多产生了机制证据。** 第三个 probe t13 独立重建输入，找回了构造规律：按误差反向排序得到的排列，与给定 PERMUTATION 逐行匹配 **64/64**。它还测试 identity pairing 和 200 个随机排列：identity 误差约 16.717%，随机排列平均约 16.095%，200 个都不满足 10% 要求。指定排列的误差则为 3.365%。这是比仅报告负相关更深入的解释。

但这些额外实验没有改变最终标签；solo 已有充分证据正确判断本合同中的固定输入。不能把“这一次多做了一层解释”直接归因于多角色结构：总 token、调用次数和角色提示都不同，也没有给 solo 同等总预算做对照。

## 无工具结果不能包装成两例误判

case_36 的无工具调用正确拒绝，虽然理由仍采用独立误差估算，估 E 约 17%，并未算出真实的 22.4116%。因此 case_36 不是跨模型稳定误判例。

case_37 原始响应 `finish_reason=length`，输出 token 恰为 32768，`message.content` 为空。**这不是一个错误 verdict，也不是主动弃答，而是预算耗尽。** 未完成的 reasoning 中虽然出现“应 reject”，不能把其中某句话抽取出来冒充最终答案。应报告：无工具两例中一例正确、一例未完成、零个已提交的错误 verdict。

## 流程质量与预算

四个工具流程都有与冻结真值一致的实际 T4 证据。三个自动审计无标记；case_37 debate 有一条恢复过的工具错误：t11 漏了 `finalize_probe_evidence.supports`，t12 随后补齐。该错误和后续恢复均保留。case_37 debate 的一个 Experimenter 回合输出恰为 32768 tokens，随后仍继续运行；共享适配器没有保留该回合的 provider finish reason，因此不能确定其截断原因。无工具流程只有一次调用，无法像工具流程一样继续下一回合。

逐项人工证据审查，包括 case_36 claim 文本中的小数点笔误、case_37 的错误恢复与 evidence 引用格式，见 [FIREWORKS_AUDIT.md](FIREWORKS_AUDIT.md)。没有将自动审计无标记视为自然语言解释必然正确。

各组 SDK 调用累计墙钟时间约：无工具 10.26 分钟，solo 10.57 分钟，debate 33.82 分钟。它们包含服务端/网络等待和 SDK 重试，不是纯生成时间，不能直接用来推断架构的因果性能差异，也不是并行任务的总历时。

## 能支持的结论范围

这是同一模型、两个选定的合成固定 workload、每格一次的探索实验。每次调用上限同为 32768，但各组总调用/总 token 预算不匹配。原始六格结果如上；没有追加更大预算的无工具重试，因此不能断言它在充足预算下也无法完成。

这些结果支持：带工具流程可以实际验证这对输入，solo 已能获得正确答案；尚未证明四角色有额外的正确率优势。机制解释的深度值得单独衡量，但本轮不够证明它只有多角色才能获得。

这组数据与 `generation/benchmark_design_generalized.md` 的原始真实 kernel 家族 FN/FP 评测边界不同，是隔离的合成机制试验；未并入当时的 32-case scoreboard（原始题库目前已有 34 例）。给定正确 reference 和 metric，一个固定数值脚本也能解决它。

## 可复核文件

- [FIREWORKS_PROTOCOL.md](FIREWORKS_PROTOCOL.md)：预先固定的配置、调度说明与复现命令。
- [REPORT_FIREWORKS.md](REPORT_FIREWORKS.md)、`scoreboard_fireworks.json`：从全部六条 traces 派生的结果、token、费用和审计标记。
- `traces_fireworks/case_36/`、`traces_fireworks/case_37/`：原始请求、无工具完整 provider 响应、完整工具运行和 probe 文件。
- `validation_gpu.json`：先前冻结的独立 GPU oracle。四个工具运行载入的源码及合同哈希均与它匹配。

最初无工具脚本的缺依赖错误发生在 API 调用前，已保留；并行启动 debate case_37 后，原串行驱动因重复目录检查退出，未重复发送 case_37 请求。最后的备份读取恰逢容器正常结束，完整返回的归档已包含全部证据。以上均不计为新的模型试验。

后续无工具 64K 补测已完成，见 [FIREWORKS_SINGLE_64K.md](FIREWORKS_SINGLE_64K.md)：case_36 正确，case_37 正常结束但误拒绝（confidence 0.85）。这两次独立补测未并入上面的原始六格结果。
