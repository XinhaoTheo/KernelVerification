# benchmark_fn_fp — 目录说明

一套对抗性的 GPU kernel 验证 benchmark，外加跑它的评测工具。

**它要回答的问题**：给一份 kernel 源码和一份契约，验证器能不能判断这份 kernel
是否满足契约 —— 既要抓出真缺陷（不漏报），也要放过合法的实现差异（不误报）。

**它的假想敌**：固定容差的 `allclose` 比对。每道题都构造成让这种方法失效：要么
缺陷在常规输入下看不出来（FN 题），要么正确的实现被容差误杀（FP 题）。

---

## 一张图看懂

```
benchmark_fn_fp/
│
├── triton/          32 道题的答案 key   ← 只有打分脚本能看
├── eval_cases/      32 个去答案副本      ← 验证器看到的就是这个
├── case_map.json    两者的对照表         ← 放在 eval_cases 外面
│
├── eval/            评测脚本 + 结果
├── traces/          每次运行的完整记录
│
├── numerical_pilot/ 早期的数值探索实验（已完成，留档）
└── modal_runner.py  通用的 Modal GPU 执行封装
```

---

## `triton/` — 答案 key，32 道题

每道题一个目录，目录名即题名，`fn` 开头是漏报题、`fp` 开头是误报题：

| 文件 | 内容 |
|---|---|
| `kernel.py` | 被验证的 kernel。真实上游代码（AutoGPTQ、Liger-Kernel、vLLM、SGLang、FlashAttention、Mamba 等）改造而来 |
| `problem.txt` | **契约** —— 这个算子必须满足什么。判决的唯一依据 |
| `test.py` | 参考实现 + 触发条件，用来证明这道题确实成立 |
| `meta.json` | **答案** —— 真值、缺陷机制、为什么 `allclose` 会失效 |

`meta.json` 里的关键字段：

```json
{
  "group": "FN",                  // FN=真有缺陷，FP=其实没问题
  "seed_class": "FN3",            // 属于哪一类种子（见下）
  "mechanism": "...",             // 缺陷怎么藏起来的
  "expected": {
    "correct_verdict": "BUGGY",   // 真值：BUGGY=reject, CORRECT=trust
    "naive_allclose_verdict": "PASS on divisible K=4096, FAIL on K=304"
  }
}
```

**这个目录绝不能给验证器看** —— `meta.json` 有答案，`test.py` 有参考实现。

### 两类题、13 种种子

**FN（false negative，漏报）—— 真有缺陷，但常规测试看不出来**

| 种子 | 缺陷怎么藏的 |
|---|---|
| FN1 | 并列时的 tie-break 与契约不符，随机数据极少出现并列 |
| FN2 | 缺一个 clamp/mask，温和输入下那一支走不到 |
| FN3 | 整块处理正确、尾块错，而测试用的尺寸恰好能整除 |
| FN4 | 读错了 head/expert/页表，但每个来源的数据都差不多 |
| FN5 | 量化规则在真实分布的离群值上才崩，高斯随机数触发不了 |
| FN6 | 每步误差极小，只有长序列累积才超出容差 |
| FN7 | 两个公式在正常量级上无差别，只在接近零时分道扬镳 |

**FP（false positive，误报）—— 其实是对的，但会被容差误杀**

| 种子 | 为什么被误杀 |
|---|---|
| FP1 | 契约在并列时允许多个答案，参考选了其中一个 |
| FP2 | 同样的信息，不同的存储布局 |
| FP3 | 同样的加法，不同的求和顺序，浮点舍入不同 |
| FP4 | 低精度格式本来就要舍入，用 fp32 的容差去卡 |
| FP5 | kernel 本身就是随机的（stochastic rounding、采样） |
| FP6 | 接近零时相对误差这个指标本身失效 |

---

## `eval_cases/` — 验证器看到的副本

由 `docs/benchmark-generation/generators/build_eval_cases.py` 从 `triton/` 生成。

```
eval_cases/case_33/
├── kernel.py     （首行的题名注释已删除）
├── problem.txt
└── meta.json     只有 {"name","status","passed":null}，没有答案
```

### 具体差在哪四处

| | 答案 key（`triton/`） | 验证器看到的（`eval_cases/`） |
|---|---|---|
| 目录名 | `fn21_gptq_group_count_floor_division` | `case_33` |
| `meta.json` | 2153 字节，含真值和缺陷机制 | 68 字节，什么都没有 |
| `kernel.py` 首行 | `"""Triton kernel under test: fn21_..."""` | `import torch` |
| `test.py` | 有 | 没有 |
| `problem.txt` | — | **完全一样** |
| `kernel.py` 正文 | — | **完全一样** |

**最要命的是 `meta.json`，不是 `test.py`。** 答案 key 里它写着
`"group": "FN"`（这题有缺陷）、`"mechanism": "...K // group_size instead of
ceil(...)"`（缺陷是什么、在哪）、`"correct_verdict": "BUGGY"`（直接就是答案）。
发出去题目就没了。

**目录名** 也不是小事：它随 `state.entry` 进入**每个 agent 的每一轮提示词**，
`list_artifact_files` 还会再暴露一次。`fn` 前缀说明"有缺陷"，后半截说明缺陷是
什么。所以改成 `case_NN`，而且编号**打乱** —— 否则 FN 排前 FP 排后，编号本身
就是分类。

**`kernel.py` 首行**原本是 `"""Triton kernel under test: <题名>."""`，等于把答案
钉在源码顶上。删掉，正文一字不动。

**`problem.txt` 一个字不删** —— 它是契约，是判决的唯一依据。不给它，验证器没有
任何标准可对照，题目不成立。

`case_03` 是空号：那道题的缺陷藏在不发出去的 `test.py` 里，验证器根本看不到出错
的代码，任何答 reject 的都白得分。该题作废、id 退役不复用，`case_33` 是它的替代。

---

## `case_map.json` — 对照表

```json
{ "cases": { "case_33": "fn21_gptq_group_count_floor_division", ... } }
```

放在 `eval_cases/` **外面**，只有打分脚本读。验证器看到 `case_26`，不知道那是
FP 题。

---

## `eval/` — 评测脚本

### 四个 baseline

| 脚本 | 这一档是什么 | 能不能跑实验 |
|---|---|---|
| `baseline1_allclose_modal.py` | 固定容差 `allclose` | — |
| `baseline2_single_llm.py` | 模型单次调用，不给工具 | ❌ 只能读代码推理 |
| `baseline2_5_solo_modal.py` | **单 agent + 全套工具** | ✅ 能写代码在 GPU 上跑 |
| `baseline3_debate_modal.py` | **四角色 debate** | ✅ 能跑，且互相质疑 |

这四档是一个消融：从第一档到第二档，量的是"会推理"值多少；第二到第三档，量的是
"能执行"值多少；第三到第四档，量的是"辩论结构"值多少。

### 支撑文件

| 文件 | 作用 |
|---|---|
| `common.py` | 载入用例、按 `case_map.json` 打分 |
| `traces.py` | **所有 runner 都必须经它写 trace**，见下 |
| `audit_traces.py` | 自动审计 trace，八项检查 |
| `capture_traces_modal.py` | 单题抓取，用来做前后对比 |
| `results_baseline*.json` | 各次运行的汇总结果 |
| `README.md` | 说明哪些历史结果已作废、为什么 |

**为什么 trace 是强制的**：前十五次运行都只保留了 `transcript.md` 的最后两万字符，
其余全丢在容器里。三个改变结论的缺陷 —— Skeptic 白白重读提示词里已有的材料、
`record_description_update` 每次都拒收第一次调用、agent 判对但理由全错 —— 都是有了
完整 trace 才看见的，而它们一直在每次运行里发生。所以 `tests/` 里有测试盯着：哪个
runner 不写 trace、或者漏传 `--max-tokens`，测试就红。

---

## `traces/` — 运行记录

```
traces/<case_id>/<arm>/
├── transcript.md      整个过程的叙述，按时间顺序 ← 从这里开始读
├── verdict.json       最终判决、置信度、理由
├── claims.json        claim 台账：每条假设 + 证据 + scope + 状态
├── tool_events.jsonl  每次工具调用的完整参数和返回
├── run.json           完整状态，含每轮 token 用量
├── probes/            agent 当场写的实验代码 + 真跑出来的 stdout/stderr
└── runner_stdout.txt  runner 自己的日志
```

`<arm>` 是 `solo` 或 `debate`，同一道题的两种配置并排放。

**怎么读**：只需要 `transcript.md`。它分四段，倒着读最快 ——
`## Verdict` → `## Claims` → 有疑问再回 `## Timeline`。

`probes/` 是单次调用完全没有对应物的部分：`tN_probe.py` 是 agent 自己写、在真 GPU
上跑过的代码，`tN_stdout.txt` 是跑出来的结果。判决里引用的每个数字都能从这里复现。

---

## `numerical_pilot/` — 早期探索（留档）

在造 benchmark 之前做的数值实验：先测清楚各种"合法的实现差异"实际能有多大偏差，
才知道 FP 题的容差该定在哪。`REPORT.md` 是结论，`answer_key.json` 和
`candidate_measurements.json` 是原始测量值。已完成，不再改动。

---

## 怎么跑

```bash
# 单题，抓完整 trace（前后对比用）
modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_33 --arm solo   --max-rounds 10
modal run benchmark_fn_fp/eval/capture_traces_modal.py --cases case_33 --arm debate --max-rounds 4

# 全量
modal run benchmark_fn_fp/eval/baseline2_5_solo_modal.py  --all --max-debate-rounds 10
modal run benchmark_fn_fp/eval/baseline3_debate_modal.py  --all --max-debate-rounds 4

# 跑完先审计，再看分数
python benchmark_fn_fp/eval/audit_traces.py
```

`--max-tokens` 默认 16384，别调低。adaptive thinking 是算在 `max_tokens` 里的，
4096 时一整轮可能全花在思考块里、返回空文本和零个工具调用 —— 有一次全量跑出三道题
零 claim 零探针，Judge 写着"the debate produced no claims and no evidence"照样下了
判决。实测各角色峰值 6.3k–8.5k token，16384 留了约一半余量。

---

## 改动 benchmark 时的规矩

1. **缺陷必须在发给验证器的文件里**。`eval_cases/` 只发 `kernel.py`、
   `problem.txt`、空壳 `meta.json`。缺陷藏在 `test.py` 里的题是无解的，而且答
   reject 就白得分 —— `fn3_gptq_dequant_group_div_coverage` 就是这么废掉的。
   host wrapper 里的缺陷是可以的，**前提是 wrapper 跟着 `kernel.py` 一起发**。

2. **加完题跑一次生成器**：
   `python docs/benchmark-generation/generators/build_eval_cases.py`
   它会分配新 id、重建 `eval_cases/`、跑泄漏检查（禁用词、真名、目录名编码、
   配对题的可区分性）。已有的 id 不会重排 —— 历史结果是按 id 存的。

3. **废题退役、不复用 id**。复用会让旧结果看起来像是在给新题打分。

4. **全量之前先单题冒烟测试**。这条是拿钱换来的：改完 runner 不验证就发全量，
   一次卡了六小时四十三分（import 位置错误），一次烧掉三十多美元（漏传
   `--max-tokens`）。
