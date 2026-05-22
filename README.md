# Kernel Verification via Debate

给 LLM 生成的 Triton kernel 做**可信度验证**的系统:不轻信生成器自己的测试,而是
独立复测正确性 + 多 agent 辩论,重点抓那些"通过了自己测试但其实是错的/作弊的" kernel。

---

## 1. 为什么要做这个

像 meta 的 KernelAgent 这种工具能自动生成 Triton kernel,但它**既出题又答题**——
同一个 LLM 写 kernel,也写测试,还经常把 fp32 偷偷降成 bf16 放宽容差、只测一个 shape。
结果是 kernel "通过了它自己的测试",但其实可能:

- **非连续输入读错内存**(没调 `.contiguous()`)
- **大输入截断**(BLOCK_SIZE 写死,行太长就丢数据)
- **数值不稳**(softmax 不减 max、reduction 累加顺序漂移)
- **作弊**(硬编码 test 的 shape、假装 Triton 实际调 torch、lazy eval 骗过 allclose)

本系统就是那个**不信任生成器、独立审查**的环节。它接收任意来源的 kernel(KernelAgent
只是当前的生成器,可替换),输出一个带证据的可信度判断。

---

## 2. 两阶段架构

```
┌─────────────────────────┐         ┌──────────────────────────────┐
│  离线:生成数据集         │         │  在线:验证                    │
│  (可替换的上游)          │         │  (我们的核心,自给自足)        │
│                          │         │                              │
│  kv-build                │  写入   │  kv-run <entry>              │
│  KernelAgent 生成 kernel │ ──────► │  recheck → debate → verdict  │
│                          │ dataset/│                              │
└─────────────────────────┘         └──────────────────────────────┘
```

- **离线(`kv-build`)**:跑 KernelAgent 把 problem 变成 kernel,存进 `dataset/`。烧钱、慢、要 GPU。跑一次,数据沉淀下来。
- **在线(`kv-run`)**:从 `dataset/` 读 kernel,做我们自己的验证。便宜,可反复迭代 agent prompt 不用重新生成 kernel。

关键设计:**验证完全不依赖生成器有没有给测试**。换个只吐 `kernel.py` 的生成器,
`kv-run` 照样能验证——因为它自己造测试。

---

## 3. 完整流程图

```
══════════════════════ 离线: kv-build ══════════════════════

 problem (PyTorch 参考实现: Model + get_inputs + get_init_inputs)
   │
   ▼
 verifier/generator.py  ── 包装 KernelAgent.TritonKernelAgent
   │   ① LLM 写一份测试
   │   ② LLM 生成 N 个 kernel seed
   │   ③ N 个 worker 并行: 写kernel→subprocess跑测试→喂错误给LLM改, 最多 max_rounds 轮
   │   ④ 任一 worker 通过即成功; 失败也回收"最像样的尝试"+错误
   ▼
 verifier/dataset.py : save_entry()
   │   把 kernel/test/seed/problem 拷进自包含目录
   ▼
 dataset/<name>/
   ├── problem.txt        原始 problem
   ├── kernel.py          最终 kernel(失败时是最佳尝试)
   ├── test.py            KernelAgent 自带测试(仅参考, 不可信)
   ├── seed_*.py          初始 seed
   ├── meta.json          { passed, status, rounds, ... }
   └── error.txt          失败时的 stderr/stdout


══════════════════════ 在线: kv-run <entry> ══════════════════════

 dataset/<entry>/  ──load_entry()──►  artifact { kernel_code, ... }
   │
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 1: RECHECK  (verifier/recheck.py — 我们独立的正确性复测)  │
│                                                               │
│  get_recheck(entry)  (有缓存则复用, --force-recheck 强制重跑)  │
│    │                                                          │
│    ├─ generate_test():  LLM 看 problem+kernel 写一份测试,      │
│    │     里面含一组【写死的刁难输入清单(battery)】:           │
│    │       • standard          (spec 正常输入 → 核心正确性)    │
│    │       • noncontig_stride2 (非连续 [::2] / [:,::2])        │
│    │       • noncontig_transpose (转置 .t())                  │
│    │       • odd_size          (size ±1, 非对齐)              │
│    │       • empty             (空张量)                       │
│    │     每个 case 用固定的 compare_outputs() 判, 打印         │
│    │       "CASE <名>: PASS/FAIL/SKIP"                        │
│    │                                                          │
│    ├─ run_test():  临时目录里放 kernel.py + kverify_compare.py │
│    │     + 测试, subprocess 跑, 退出码只反映 standard          │
│    │                                                          │
│    └─ 解析 CASE 行 →  status   = standard 的结果(核心对错)    │
│                       robustness = 其余刁难 case 的结果        │
│                                                               │
│  分两档(关键设计):                                            │
│    • standard FAIL  → status=failed → 真 bug                  │
│    • 刁难 FAIL       → 只记 robustness, 不自动判死              │
└─────────────────────────────────────────────────────────────┘
   │  把 recheck 结果折进 artifact (passed/status/test_code/error)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 步骤 2: DEBATE  (verifier/debate.py — 开放式语义审查)          │
│                                                               │
│  每轮 (最多 DEBATE_MAX_ROUNDS):                                │
│                                                               │
│    author    ── 证人, 描述 kernel 在做什么 + 怎么演化的        │
│       │         (只描述, 不评好坏; 读 seed_*.py)               │
│       ▼                                                       │
│    skeptic   ── 质疑者, 立结构化 claim(可测的具体断言)        │
│       │         例: {"type":"non_contig", "statement":         │
│       │              "对 x[::2] kernel 会读错"}               │
│       │  ──► 登记进 claims 台账, 状态 open                    │
│       ▼                                                       │
│    verifier  ── 执行者, 逐条 open claim 写探针真跑 GPU,        │
│                 用 compare_outputs() 判, 回填 claim:          │
│                   confirmed (真错) / rebutted (没事) /        │
│                   inconclusive (测不了)                       │
│                                                               │
│    收敛: skeptic 这轮没立新 claim → 停                        │
│       ▼                                                       │
│    judge     ── 裁判, 读整张 claims 台账出最终 verdict:        │
│                   trust / reject / needs_more_evidence        │
│                 + decisive_claims (哪几条定的罪)             │
│                 + 可推翻 verifier 的初判(如认定是预期舍入)    │
└─────────────────────────────────────────────────────────────┘
   │
   ▼
 dataset/<entry>/debate_result.json
   { recheck_status, verdict, claims(台账), history(全发言) }
```

---

## 4. 逐环节详解

### 4.1 `verifier/generator.py` — 包装 KernelAgent

把 `KernelAgent.TritonKernelAgent.generate_kernel()` 的返回归一化成统一 artifact:
`{kernel_code, test_code, passed, status, rounds, error, session_dir, raw}`。

成功时 `kernel_code` 是最终 kernel;失败时是"打得最久的 worker 的最佳尝试"+ 错误信息
(KernelAgent 上游被我们 patch 过,原本失败什么都不留,见 [roadmap.md](roadmap.md))。

### 4.2 `verifier/dataset.py` — 自包含数据集

`save_entry()` 把每次生成的产物**拷贝**进 `dataset/<name>/`(不是记路径),所以删掉
KernelAgent 的运行目录也不影响数据集,能 git commit、能搬机器、能手写注入。
`load_entry()` 读回成 artifact;`session_dir` 指向 entry 目录本身,author 读 seed 就从这里读。

### 4.3 `verifier/recheck.py` — 独立正确性复测(我们的 ground truth)

这是**不信任生成器**的核心。`get_recheck()`:

1. **`generate_test()`**:让 LLM 看 `problem.txt` + `kernel.py`,写一份测试。LLM 看得到
   kernel 源码,所以知道怎么调它(解决了"每个 kernel 调用方式不同"的问题)。prompt 强制
   它跑一组**写死的刁难输入清单(battery)**,每个打印 `CASE <名>: PASS/FAIL/SKIP`。
2. **`run_test()`**:临时目录里放 `kernel.py` + `kverify_compare.py`(固定比对器)+ 测试,
   subprocess 跑。全新进程,规避 CUDA fork 陷阱。
3. **解析 `CASE` 行**,分两档存进 `meta.json["recheck"]`:
   - `status`:只看 `standard`(spec 正常输入)→ 核心对错,这个才驱动"是不是真 bug"
   - `robustness`:其余刁难 case → 只记录,**不自动判死**

**为什么要 battery**:之前只靠 debate 的 skeptic 临场想刁难输入,非确定——同一个有
非连续 bug 的 kernel,这轮想到了就抓到、那轮没想到就漏。battery 写死清单,每次必跑,
机械 bug(非连续/奇怪 size/空)不再漏。

**两档的意义**(scope 契约):kernel 在 spec 正常输入下错 = 铁板钉钉的 bug(reject);
只在非连续这种刁钻输入下错 = 健壮性问题,记一笔但不武断判死(因为这些输入可能超出
kernel 的设计范围)。

### 4.4 `verifier/compare.py` — 统一比对器(单一真相源)

`compare_outputs(out, ref) → (matches, max_diff, detail)`。recheck 测试和 verifier 探针
**都 import 它**(运行时拷进临时目录叫 `kverify_compare.py`),所以"对不对"的判定
逻辑只有一处、固定:

- **容差按 dtype 定**:fp32 用 1e-3,fp16/bf16 用 1e-2/2e-2(不让 LLM 自己拍 `1e-4` 这种死数)
- **`equal_nan=True`**:kernel 和 reference 同位置都 NaN 算匹配(softmax 喂全 inf,PyTorch 自己也吐 NaN,不能算 kernel 的错)
- **判 bug 的标准是"偏离 reference"**,不是"出现了 NaN/大数"

> 这个文件是踩坑后加的:早期让 LLM 在每个探针里自己写比对,它容差乱拍、NaN 不对称处理,
> 造出假阳性。抽出固定比对器后,recheck 和 verifier 用同一把尺。

### 4.5 `verifier/debate.py` + `agents/` — 多 agent 辩论

debate 管 **battery 列不出清单的语义/算法 bug**(如 cumsum 跨块累加错、作弊、微妙数值)。
四个角色:

- **`agents/author.py`(证人)**:读 final kernel + `seed_*.py`,客观描述 kernel 做什么、
  从 seed 到 final 改了什么。prompt 禁止评好坏,只描述。无新可说时喊 `NO_NEW_OBSERVATIONS.`。
- **`agents/skeptic.py`(质疑者)**:找可能的 bug/作弊,但必须立成**结构化 claim**——
  一个能跑代码验证的具体断言(附带要测的精确输入)。无新质疑时喊 `NO_NEW_CONCERNS.` +
  空 claim 列表。
- **`agents/verifier.py`(执行者)**:遍历台账里 `open` 的 claim,**每条写一段探针**构造
  那个输入、跑 kernel + reference、用 `compare_outputs` 判,回填 claim 状态
  (confirmed/rebutted/inconclusive)+ 证据(探针代码 + 实测数字)。
- **`agents/judge.py`(裁判)**:不按轮次发言,**最后读整张台账出一次 verdict**。
  它做"严重性"判断(计数做不到的):一个 confirmed 的 256 误差是真 bug 还是预期 bf16 舍入?
  judge 可以**推翻** verifier 的初判。输出 `{verdict, confidence, decisive_claims, claim_notes, reason}`。

- **`agents/parsing.py`**:从 LLM 回复里稳健抠 JSON(优先 ```json 块,跳过 prose 里的 ```python 代码块)。
- **`agents/types.py`**:`Turn` / `Claim` 的 TypedDict 定义。

**收敛**:每轮 author→skeptic→verifier;当 skeptic 这轮没立新 claim 就停。judge 不参与
轮次,只在循环结束后裁决一次。

### 4.6 `verifier/llm_client.py` — LLM 调用

统一的 Anthropic 调用。两个要点:(1) API 无状态,message 列表由我们维护;
(2) message 排列让 prompt cache 能在同一 agent 跨轮命中(kernel/test 放可缓存前缀)。
`oneshot()` 给 recheck/verifier 写探针用(无历史的一次性调用)。

### 4.7 `verifier/gpu_pick.py` — 自动选 GPU

跑前 `nvidia-smi` 排序 + 真做一次小 allocate 探测(光看 nvidia-smi"空闲"不够,共享机器
上有的卡报空闲但拿不到 context),选第一个能用的设个 `CUDA_VISIBLE_DEVICES`。

---

## 5. 三个信号

一个 kernel 跑完 `kv-run` 后,你拿到三层判断,**别混为一谈**:

| 信号 | 来源 | 含义 |
|---|---|---|
| `recheck.status` | battery 的 `standard` case | **核心正确性**:spec 正常输入下对不对 |
| `recheck.robustness` | battery 的刁难 case | **健壮性**:非连续/奇怪 size/空 等是否处理 |
| `debate.verdict` | author/skeptic/verifier/judge | **语义审查**:列不出清单的算法 bug / 作弊 |

> 注:目前这三个信号是分别产出的,"怎么合成一个最终结论"还是个待定的设计点。

---

## 6. 目录结构

```
kernel_verification/
├── KernelAgent/        # 上游生成器(meta-pytorch/KernelAgent),已 patch,vendored
├── KernelBench/        # 题库(ScalingIntelligence/KernelBench),vendored
├── verifier/           # 主包
│   ├── generator.py    # 包装 KernelAgent
│   ├── dataset.py      # save_entry / load_entry / iter_entries
│   ├── recheck.py      # 独立复测 + 刁难 battery(kv-recheck)
│   ├── compare.py      # 固定比对器 compare_outputs(单一真相源)
│   ├── debate.py       # 辩论主循环 run_debate
│   ├── llm_client.py   # Anthropic 调用 + prompt cache
│   ├── gpu_pick.py     # 自动选可用 GPU
│   ├── build_dataset.py# 离线建数据集(kv-build)
│   └── run.py          # 在线验证入口(kv-run)
├── agents/             # 四个角色 + 工具
│   ├── author.py  skeptic.py  verifier.py  judge.py
│   ├── parsing.py      # 抠 JSON
│   └── types.py        # Turn / Claim
├── dataset/            # 生成的 kernel + 三标签(可 commit)
│   └── <name>/{problem.txt, kernel.py, test.py, seed_*.py,
│               meta.json, recheck_test.py, debate_result.json, ...}
├── tests/              # 探索/调试脚本
├── pyproject.toml      # uv 项目, torch cu128, KernelAgent path dep
└── roadmap.md          # 设计演进记录
```

---

## 7. CLI 速查

```bash
# 离线: 建数据集 (烧钱/慢/要 GPU, 跑一次)
uv run kv-build --curated              # 建 10 个精选 KernelBench 题
uv run kv-build --problem elem_add     # 建单个内置题
uv run kv-build --curated --list       # 干跑, 只看会建哪些

# 独立复测 (只跑 recheck, 不辩论)
uv run kv-recheck                       # 跑所有还没复测的
uv run kv-recheck elem_add              # 强制重跑一个
uv run kv-recheck --list                # 看每个 entry 的复测状态

# 在线: 验证 (recheck → debate → verdict)
uv run kv-run elem_add                  # 跑一个
uv run kv-run elem_add --verbose        # 看全过程: recheck 测试 / 每个 agent 发言 / 探针代码
uv run kv-run elem_add --force-recheck  # 重新生成 recheck 测试再跑
uv run kv-run --list                    # 列出能跑的 entry
```

跑完看 `dataset/<entry>/debate_result.json` 拿完整记录。

---

## 8. 已知局限

- **debate verdict 仍有非确定性**:battery 把机械 bug(非连续等)变确定了,但 debate 里
  skeptic 仍靠 LLM 即兴立 claim,语义类 bug 的覆盖还是 run-to-run 有波动。
- **battery 的 case 仍是 LLM 写的**:清单是写死的,但每个 case 的具体构造仍由 LLM 生成,
  有构造变数。彻底消除要纯 Python 写死的 harness(但撞到"通用调 kernel"的签名问题)。
- **三个信号未合成**:recheck.status / robustness / debate.verdict 目前分别产出,最终
  结论怎么综合还没定。
- **scope 契约未固定**:非连续这类输入"算不算 bug"取决于 kernel 的预期用途,目前用
  "正常输入失败才判死、刁钻失败只记录"的折中。
