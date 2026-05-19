# Roadmap: Kernel Verification via Debate

给claude的 roadmap

## 1. 项目初始化
目前以 meta 的 [KernelAgent](https://github.com/meta-pytorch/KernelAgent) 作为生成 kernel / 数据集的工具。
已 clone 到 [KernelAgent/](KernelAgent/)。

安装：
```bash
cd KernelAgent && pip install -e .
```

环境变量（在项目根写 `.env`）：
```bash
ANTHROPIC_API_KEY=sk-ant-...        # 或 OPENAI_API_KEY
OPENAI_MODEL=claude-sonnet-4-20250514   # 默认模型，也可改成 gpt-5
NUM_KERNEL_SEEDS=4                  # 并行 worker 数
MAX_REFINEMENT_ROUNDS=10            # 每个 worker 最多重试轮数
```

运行时还需要一张支持 CUDA 的 GPU（KernelAgent 会真的把 kernel 编译跑起来做正确性验证）。

## 2. KernelAgent 入口（已核对）

真实入口在 [KernelAgent/triton_kernel_agent/agent.py:462](KernelAgent/triton_kernel_agent/agent.py#L462)：

```python
def generate_kernel(
    self,
    problem_description: str,             # 必填：natural language + 可选 PyTorch 参考实现的字符串
    test_code: str | None = None,         # 选填：额外正确性测试源码
    generate_default_test: bool = True,   # 是否让 LLM 自动生成一个默认测试
) -> dict[str, Any]
```

**输入 `problem_description` 是一个字符串**，没有强 schema。但 KernelAgent 的示例（参见 [examples/triton_01_element_add.py](KernelAgent/examples/triton_01_element_add.py)）以及 KernelBench 题目都遵循同一种"半结构化"写法：

```python
import torch
class Model(torch.nn.Module):
    def forward(self, a, b):
        return a + b

vector_size = 1024
dtype = torch.float32

def get_inputs():
    a = torch.randn(vector_size, dtype=dtype, device='cuda')
    b = torch.randn(vector_size, dtype=dtype, device='cuda')
    return [a, b]

def get_init_inputs():
    return []
```

即：**reference `Model` + `get_inputs()` + `get_init_inputs()` 三件套**，整体当成字符串传进去。LLM 既靠它理解语义，也靠它生成测试。

题目来源有三条路：
1. **手写字符串**（MVP 最快）
2. **KernelBench 题目文件**（推荐做数据集用，克隆到 `../KernelBench/`，KernelAgent 默认从那里找）
3. **JSON subgraph**（Fuser pipeline 从真实模型里抽，进阶用法）

**输出**（成功时）：
```python
{
    "success": True,
    "kernel_code": str,        # Triton kernel 源码字符串
    "worker_id": int,
    "rounds": int,
    "session_dir": str,        # 会话目录，里面有 problem.txt / test_*.py / seed_*.py / final_kernel.py / result.json
}
```
失败时：`{"success": False, "message": str, "session_dir": str}`。

注意：**返回值里没有 `test_code` 字段，也没有显式的 `passed`**。
- `success == True` 已经隐含"至少一个测试 pass 了"（KernelAgent 内部跑过验证才会返 True）。
- 测试源码要从 `session_dir/test_0.py` 自己读。

## 3. 最小 generator 包装

写 [generator.py](generator.py)，把上游字段归一化成我们 debate 阶段想要的形状：

```python
def generate_kernel(problem_description: str) -> dict:
    """
    Returns:
        {
            "kernel_code": str,    # Triton kernel 源码
            "test_code":   str,    # session_dir/test_0.py 读出来
            "passed":      bool,   # 对应上游的 success
            "session_dir": str,    # 留着给 debate agents 翻日志/seed/profile
        }
    """
```

内部：
1. `agent = TritonKernelAgent()`，调 `agent.generate_kernel(problem_description)`
2. `passed = result["success"]`
3. 从 `result["session_dir"] / "test_0.py"` 读出 `test_code`
4. `kernel_code = result.get("kernel_code", "")`（失败分支没有这个字段，置空串或 None）
5. 透传 `session_dir` —— debate 的 skeptic 可能要去翻历史 seed / worker 日志找证据

## 4. Review Agents

在 `agents/` 下写三个文件，每个文件 = 一段 system prompt + 一个 `respond(history, artifact, tools=None) -> Turn` 函数：

- `agents/author.py` —— **证人**，不是辩护者。读 final kernel + `session_dir/seed_*.py`，描述"kernel 在做什么 + 是怎么改出来的"。prompt 明确禁止评价好坏。
- `agents/skeptic.py` —— 找问题：测试过拟合、伪 Triton、数值漂移、race condition、author 描述与代码不符等。
- `agents/judge.py` —— 中立裁判，判 skeptic 主张是否被 author 用可观察证据回应。

**为什么不是 advocate？** advocate 角色在 MVP 阶段没有工具，所谓"辩护"全是 hallucinate；author 改成"被审计的证人"——只描述可观察事实，skeptic 可以对它的描述做事实核查，judge 有了"被声称行为 vs 实际质疑"的对照基线。Phase 2 加 verifier（带工具的执行体）来替代 author 的部分职能 / 直接验证 skeptic 主张。

**信息不对称（有意为之）**：
- author 看：final kernel + test code + `session_dir/seed_*.py`（演化历史）
- skeptic 看：final kernel + test code + 完整 history（包括 author 发言），**看不到** seed 文件
- judge 看：final kernel + test code + 完整 history

这样 author 是"演化轨迹的专业读者"，不会和 skeptic 完全重叠；skeptic 想核查 author 关于 seed 的声明，只能通过质问让 author 引用具体行号。

### 接口约定（为 Phase 2 反 hack 预留扩展点）

```python
# agents/types.py
Turn = {
    "by": "author" | "skeptic" | "judge",
    "round": int,
    "text": str,                      # 自由文本发言（MVP 阶段够用）
    "claims": list[Claim] | None,     # Phase 2 启用：结构化主张
    "tool_calls": list[dict] | None,  # Phase 2 启用：本轮调用过的工具及结果
}

Claim = {
    "type": str,           # e.g. "test_overfit" / "numerical_drift" / "fake_triton" / "race_condition"
    "statement": str,
    "evidence": list[dict],  # {"kind": "kernel_lines" | "tool_call", "ref": ...}
    "status": "open" | "rebutted" | "confirmed",
}

def respond(history: list[Turn], artifact: dict, tools=None) -> Turn: ...
```

MVP 阶段 `tools=None`、`claims=None`、`tool_calls=None`，只用 `text` 字段；Phase 2 再把后两个填起来。这样改接口不破坏已有调用方。

### LLM client 设计要点

所有 agent 共用 `verifier/llm_client.py`（claude-sonnet-4-6 起步）。**API 本身是无状态的**——message list 由我们维护，不是 Anthropic / OpenAI 替我们记。`build_messages` 强制按下面顺序拼接，让 [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)（5 分钟 TTL）能在同一 agent 跨轮命中：

1. **可缓存前缀**：`artifact["kernel_code"]` + `artifact["test_code"]`（每轮不变，所有 agent 共享格式）
2. **可缓存前缀**（可选）：`extra_context`，author 用它塞 `seed_*.py` 和 `final_kernel.py`；skeptic / judge 不用
3. **system prompt**：该 agent 的角色定义（每轮不变，但每个 agent 不同 → 不同 agent 之间无法共享缓存）
4. **history 各轮发言**：append-only，最新一轮在最后

不同 agent 之间不要试图共享 cache，先把单 agent 跨轮命中做好。

## 5. Review 主循环

写 `verifier/debate.py`：

```python
def run_debate(artifact, max_rounds=4, tools=None):
    history: list[Turn] = []
    for round_idx in range(max_rounds):
        auth = author.respond(history, artifact, tools)   # 证人陈述
        history.append(auth)
        skp = skeptic.respond(history, artifact, tools)    # 质询
        history.append(skp)

        # 轻量裁断：judge 判断本轮双方是否都没引入新信息
        if judge.no_new_arguments(history, artifact, round_idx):
            break

    verdict = judge.final_verdict(history, artifact)
    return verdict, history
```

收敛条件（MVP）：**author 和 skeptic 最新一轮都未引入新信息 / 新质疑**（由 judge 在循环内做轻量裁断）。
Phase 2 收敛条件升级为：**所有 `claims` 都进入 `confirmed` 或 `rebutted` 状态**，judge 不再判"谁更能说"而是判"哪些 claim 的 evidence 站得住脚"。

## 6. 离线生成 + 在线 debate 拆分

verification 流程**只该读数据集**，不该跟生成耦合。每次跑 debate 都现场调 KernelAgent 是浪费——又烧 LLM 钱又抢 GPU，还让 agent prompt 的迭代变贵。所以拆成两个阶段：

```
[ 离线（贵、少跑）]                  [ 在线（便宜、反复跑）]

kv-build / build_dataset.py          kv-run / run.py
   |                                     |
   v                                     v
generator.py ──> dataset.save_entry ──> dataset/<name>/
                                          ├── problem.txt
                                          ├── kernel.py
                                          ├── test.py
                                          ├── seed_*.py
                                          └── meta.json
                                              |
                                              v
                                          dataset.load_entry ──> debate.run_debate
```

**dataset entry 自包含**（拷贝，不引用原 session_dir）。理由：
- 删 `triton_kernel_logs/` 不会让数据集失效
- 能 git commit 当 fixture，能搬到别的机器
- **手写注入恶意 kernel** 时（`dataset/_bad_attn/` 里塞个故意 hack 的实现），跟生成出来的 entry 同结构，零额外代码

### `verifier/dataset.py` 接口

```python
save_entry(name, artifact, *, dataset_dir=None) -> Path
load_entry(name, *, dataset_dir=None) -> artifact  # 跟 generator 返回值同形状
iter_entries(*, dataset_dir=None) -> Iterator[str]
```

`load_entry` 把 `session_dir` 设成 entry 自己的目录——author 的 `seed_*.py` glob 直接命中，**不用改 author/skeptic/judge 任何一行**。

### CLI

```
uv run kv-build                          # 把 PROBLEMS 全部生成进 dataset/
uv run kv-build --problem softmax        # 单独建一个
uv run kv-run                            # 在 dataset/elem_add 上跑 debate
uv run kv-run --list                     # 列出现有 entries
uv run kv-run softmax                    # 跑指定 entry
```

### 反 hack sanity check（顺便能干的事）

手写 `dataset/_overfit_add/` 等 adversarial entry——故意做错的 kernel + 一个会过的 test。跑 `kv-run _overfit_add` 看 debate 能不能识别。这是验证我们 review 体系**真有用**的关键基线。

## 7. 目录结构（目标）

```
kernel-verification/
├── KernelAgent/            # 已 clone（meta-pytorch/KernelAgent），uv editable 安装
├── KernelBench/            # 可选，做数据集时 clone 到与 KernelAgent 同级
├── verifier/               # 主包
│   ├── __init__.py
│   ├── generator.py        # 包装 TritonKernelAgent，归一化字段
│   ├── llm_client.py       # Anthropic 调用 + prompt cache 友好的 message 排列
│   ├── debate.py           # author → skeptic → judge 主循环
│   ├── dataset.py          # save_entry / load_entry / iter_entries
│   ├── build_dataset.py    # 离线 CLI（kv-build），跑 generator 写入 dataset/
│   ├── run.py              # 在线 CLI（kv-run），从 dataset 读后跑 debate
│   └── gpu_pick.py         # 自动挑可用 GPU 设 CUDA_VISIBLE_DEVICES
├── agents/                 # 三个角色 agent，独立成包
│   ├── __init__.py
│   ├── types.py            # Turn / Claim TypedDict
│   ├── author.py           # 证人：读 seeds + final kernel，描述行为与演化
│   ├── skeptic.py          # 质疑者：找 hack / bug / 边界 / author 描述漏洞
│   └── judge.py            # 裁判：收敛判定 + 最终 verdict
├── dataset/                # 生成的 kernel artifact 集合（gitignored 默认）
│   └── <name>/
│       ├── problem.txt
│       ├── kernel.py
│       ├── test.py
│       ├── seed_*.py
│       └── meta.json
├── tests/                  # 探索/调试脚本
│   └── inspect_generator.py
├── pyproject.toml          # uv workspace；torch 锁 cu128 wheel；KernelAgent path dep
├── .env                    # ANTHROPIC_API_KEY 等（gitignored）
├── .env.example
└── roadmap.md
```

跑法：`uv run kv-build` 离线建数据集 → `uv run kv-run [entry]` 在线跑 debate。

## 8. Phase 2: 反 hack 升级路径

MVP（Phase 1）跑通后，光靠"读代码 + 默认测试通过"无法识别下列典型 hack：

- **测试过拟合**：kernel 里硬编码 `if shape == (1024,): return precomputed_answer`，default test 用的就是这个 shape，必过
- **伪 Triton**：包了 `@triton.jit` 装饰但里面整段 `torch.*` 实现，根本没用 Triton 语义
- **数值精度漂移**：fp16 reduction 顺序差异让结果跟 reference 差 1e-2，default test tolerance 1e-3 刚好擦边过
- **race condition**：小输入 block 数少不暴露，大输入 atomic 顺序乱套

纯文本 review 里 author 只能描述 kernel 在做什么、skeptic 只能凭代码推测，两边都没法实际验证假设。需要给 skeptic（或新引入的 verifier 角色）配"武器"。

### 升级方向 A：给 skeptic 加工具

把 skeptic 从纯 LLM 升级成 LLM + tool use，至少提供：

- `read_kernel(lineno_range)` —— 引用具体行号说话，不是凭印象指控
- `run_kernel(custom_inputs)` —— 自己构造 adversarial input（极端 shape、非对齐、边界数值）跑一遍
- `diff_against_reference(inputs, tolerance)` —— 跟 PyTorch reference 实现做严格数值比对
- `inspect_session_dir()` —— 翻 `seed_*.py` / worker 日志 / `result.json`，看看 KernelAgent 自己改了多少轮、修过哪些 bug

这就是为什么 [generator.py](generator.py) 必须透传 `session_dir`——那是 skeptic 的证据池，已经在 [第 3 节](roadmap.md#L80) 写了。

### 升级方向 B：结构化 claim 替代自由对话

启用 [Turn / Claim 结构](roadmap.md#L106)，每个主张都强制附 `evidence`（kernel 行号 或 工具调用结果），judge 按 claim 粒度裁决而不是按"谁更雄辩"。

### 升级方向 C：差异化测试

KernelAgent 默认测试可信度低（同一个 LLM 既出题又解题）。Phase 2 需要：

- 强制 `generate_default_test=False`，自己提供更严格的测试（KernelBench 自带 verifier 或我们写的 adversarial test suite）
- skeptic 有权在辩论中动态追加测试用例，临时跑一次看 kernel 表现

### 升级方向 D：引入 verifier 第四角色

skeptic 加工具会让单个 agent 的职责变重（既要发现问题又要执行验证），可以拆出一个独立的 verifier：

- skeptic 仍是纯 LLM，提结构化质疑（"我认为 line 42 在 shape=(1023,) 下会越界"）
- verifier 收到 skeptic 的 claim，去 session_dir 真跑这个 case，把结果回填进 claim 的 evidence
- judge 看的是"被 verifier 实测过的 claim"，几乎不用做主观判断

这样 author 描述、skeptic 质疑、verifier 执行、judge 裁决，每个角色单一职责。

### 演进策略

不大改架构，只填上面预留的扩展点：`tools` 参数从 None 变成真实工具集，`claims` 字段从 None 变成结构化列表，judge 的收敛判定从"无新论点"切到"全 claim 闭合"。三个 agent 文件本身不用重写，verifier 是新增 agent。

## 9. Open Questions

- KernelAgent 的中间产物已确认会落到 `session_dir/` 下（`problem.txt` / `seed_*.py` / `test_*.py` / `final_kernel.py` / `result.json` + agent 主日志在 `triton_kernel_logs/agent_*.log`）。下一步要确认的是 worker 的编译/运行日志在哪、是否含 profile 数据，以及 skeptic 怎么消费这些原始 artifact。
- judge 的"无新论点"判定要不要用 embedding 相似度兜底，避免 LLM 自我重复时误判收敛？
- 是否需要在 debate 结束后让 generator 根据 verdict 重新生成 kernel（闭环 vs. 一次性）？KernelAgent 本身已经有 `max_rounds` 的自修复循环，要想清楚我们这层 debate 跟它内部 refinement 的边界。
