# 无工具单次调用补测：64K

用户要求再次测试无工具调用。预先固定：case_36/case_37 各一次，trial=r2_64k，模型仍为 Fireworks accounts/fireworks/models/glm-5p3，FIREWORKS_API_KEY；仅将 max_tokens 从 32768 增至 65536。源码、输入、阈值、system/user prompt 和 JSON-object 格式不变。两个案例并行请求，各自是独立单次调用，无工具、无答案反馈、无语义重试。所有响应及 finish_reason 保留。

这是看到 r1 后追加的预算补测，同时也有模型随机采样差异；不并入原始六格比较。原 REPORT_FIREWORKS.md 和 scoreboard_fireworks.json 保持原样。按现有项目价格配置，两次用满输出预算的 API 估算约 $0.145，实际按返回 usage 记录。

## 已完成结果

case_36、case_37 并行，总等待约 5 分 14 秒。两次均以 `finish_reason=stop` 正常结束，无预算耗尽，无工具调用。

| 案例 | 独立 GPU 真值 | 本次 verdict | 结果 | confidence | 输出 tokens | 耗时 | API 估算 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| case_36 | E=22.4116%，reject | reject | 正确 | 0.90 | 25,492 | 305 秒 | $0.02838 |
| case_37 | E=3.3647%，trust | reject | **误拒绝** | 0.85 | 22,122 | 314 秒 | $0.024673 |

新增 API 估算合计 **$0.053053**（约 $0.053），使用项目配置价格，不是账单；这轮没有 GPU 任务。

## 错误发生在哪里

case_37 的最终回答估计两路量化误差合成后约为 16%–17%，认为明显超过 10% 阈值。该估计把单路误差按独立噪声合成，没有计算给定行排列后的误差相关性。实际排列使误差强烈抵消，真实误差只有 3.3647%。上一轮 solo 和 debate 都通过执行验证正确接受 case_37。

因此本次 case_37 是一次真实提交的错误 verdict，不能再解释为“仅因没有输出最终答案而失败”。此前 r1 的预算耗尽仍单独保留，不能追溯改记成误判。

case_36 虽然标签正确，最终解释仍依赖独立误差估算，而且把 64 个输出行误称为 128 行。它没有数值验证真实的 22.4116%。正确标签不意味着解释中的每个假设都成立。

## 解释边界

本次两条实际输出都少于原来的 32768 tokens；不能据此说增加上限导致模型完成，独立采样差异也可能解释变化。本轮是一次看到 r1 后追加的补测，不是独立大样本或稳定错误率证明。它提供了 Fireworks 无工具在 case_37 上正常结束却误判的实例，与原先 Opus 的 case_37 误拒绝方向一致。

这仍未证明四角色比 solo 更好：同一个 case_37，solo 已能独立正确验证。

请求逐字段核对过，除 max_tokens 外与 r1 完全一致；源码及合同哈希匹配冻结 oracle。原始六格 REPORT_FIREWORKS.md 与 scoreboard_fireworks.json 没有重算或混入本轮。

- 结构化对照：`scoreboard_fireworks_single_64k.json`。
- 原始 provider 响应、精确请求及用量：`traces_fireworks/case_36/single_call/r2_64k/`、`traces_fireworks/case_37/single_call/r2_64k/`。
- 无语义重试；case_36、case_37 各一次。
