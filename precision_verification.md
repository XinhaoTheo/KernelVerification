# 精度验证的失效模式与四类对抗测试

> 本文档的定位：这是当前阶段（Phase 2 的一条主线）的**设计 + 对抗测试构造计划**。
> 它和 [roadmap.md](roadmap.md) §10「防 hack 技巧清单」是**正交**的两条轴——
> §10 防的是「生成器作弊」（fake Triton / 偷答案 / 过拟合），
> 本文档攻的是「**数值误差这个判据本身，在某些算子上不再等价于正确性**」。
> 它直接针对 [README.md](README.md) §3.4（`compare.py` 单一 allclose 容差）和
> §7 limitation #4（「什么算 in-scope」未定）所暴露的根本问题。
> 来源：与 Zhuoming 的讨论。

---

## 0. 一句话结论

现在 verifier 的核心判据是 `compare_outputs(out, ref)` —— 把候选 kernel 输出和高精度 reference 比，看
`|cand − ref|` 是否落在一个按 dtype 分级的容差里（fp32 1e-3 / fp16·bf16 1e-2，见
[verifier/compare.py](verifier/compare.py)）。**这套判据默认「数值误差小 ⇔ 算得对」。**
在 matmul 这类算子上这个等价成立；但在一大类「会把小值压扁、抹掉，或被指数放大误差」的算子上，
**「数值误差通过」和「真的算对」会分叉成两件事**，单一标量容差永远分不开它们。

---

## 1. 主线：数值误差什么时候不再是「正确性」的忠实代理

为什么 matmul 上判据好用：matmul 大致**保数量级、误差线性传播**——输入错一点，输出错对应的一点，
数值误差的大小**忠实反映**了 kernel 错得多严重。「误差小」和「算得对」几乎是同一件事。

一旦换成会**压扁小值**的算子，等价关系断裂，验证朝两个相反方向出错。每一类失效场景都用这两个方向去套：

| 方向 | 含义 | 触发机制 |
|---|---|---|
| **false accept**（放过坏的）| kernel 其实算错了，但被判「过」 | 它错的那部分恰好是被算子**压没了的小值**，最终输出几乎不变，误差显示不出来 |
| **false reject**（误杀好的）| kernel 其实是对的，但被判「不过」 | 算子本身太 lossy（低精度量化 / exp 放大），正确输出和高精度 reference 也差很远，误差超容差 |

**统一原则：找「会把小值压扁/抹掉」或「会把误差放大」的算子。** 光测 matmul 没用——它不压缩小值，bug
藏不住。从第 1 类到第 4 类，「数值误差」和「真的算对」之间的缝越拉越大；第 2 类则在另一根轴上——
它的正确性根本不是数值量。

### 四类速查表

| 类 | 代表算子 | 误差为何失效 | false accept 机制 | false reject 机制 | 该怎么判 |
|---|---|---|---|---|---|
| 1 | softmax / ReLU·GELU·SiLU | softmax 压扁尾部 + exp 放大头部；activation 有死区 | 尾部/死区里的错误被压没 → 输出几乎不动 | 头部小扰动被 exp 放大成大输出误差 | 在 activation **之前**比，或对被压扁区域专门构造输入/加权 |
| 2 | sort / topk | 输出是**离散 index 集合**，`\|cand−ref\|` 无定义 | 激进近似只在 benign 分布验过，换真实分布漏掉重要 key | 量化打平 + 稳定排序 tie-break → 选中值相同但 index 不同 | 按被丢元素相对保留集的 **value-gap 加权**，或直接比**下游输出** |
| 3 | softmax/activation **+ FP8** | 第 1 类的压扁 + FP8 格子太粗（E4M3 每区间仅 8 值） | 尾部概率被 FP8 snap 成同码或 flush 到 0 → **误差严格为 0** | 正确 FP8 kernel 跟 fp32 ref 误差本来就大，逼你开大容差 | 量化感知 reference；按算子类型切判据 |
| 4 | low-bit FP4 / INT4 | 格子粗到离谱（E2M1 每区间 2 值；INT4 共 16 档） | 真实差距 < 量化步长 → round 成 0，bug 逐 bit 隐形 | 正确 INT4 kernel 跟 fp32 ref 差到天上，任何紧容差全 reject | 退化到**下游 / 任务级指标**，逐元素误差失效 |

---

## 2. 四类失效场景（逐类拆开）

### 第 1 类：softmax / activation —— 最基础的双向失效

出现在 attention 的 softmax、MLP 的激活（ReLU / GELU / SiLU）。排第一是因为 softmax **同时**具备两个相反的危险性质。

**压扁尾部 → false accept。** softmax = `exp(x_i) / Σ exp(x_j)`，离 max 越远的 logit 贡献趋近 0。
例子：真实 logits 里 key A = 8.0（主导），key C = 0.5（尾部）。softmax 后 A 权重 ≈ 1，C 极小。
候选 kernel 把 C 算成 3.0（真值 0.5，错得离谱）——但 `exp(3)≈20` 对比 `exp(8)≈2981` 仍是小不点，
C 权重从 ~0.05% 变到 ~0.7%，最终输出 `o` 几乎不动。**尾部错得一塌糊涂，输出误差极小，白送过关。**
activation 同理：ReLU 把负半轴清零，负数区算错多少输出都是 0；GELU/SiLU 在大负值区饱和压向 0——
天然有「死区」，死区里的错误输出上完全看不见。

**指数放大头部 → false reject。** 因为有 `exp()`，大 logit 上一点点绝对误差被指数放大。
同例：kernel 把主导项 A 算成 8.5（真值 8.0，误差 0.5，纯属正常低精度抖动），但
`exp(8.5)/exp(8) = e^0.5 ≈ 1.65`，A 权重涨 1.65 倍，而 A 主导整个输出 → `o` 偏一大截，超容差被判错。
**一个没问题的 kernel 被 exp 放大成「错的」。**

**对 verifier 的意义。** 比 pre-softmax（logits）还是 post-softmax（`o`）、用绝对还是相对误差，
会给出**不同判决，且没有干净选法**：比 `o` 放过尾部/死区的错（放过坏的），头部小误差又被 exp 放大（冤枉好的）。
可行方向是在 activation **之前**比、或针对被压扁区域专门构造输入并加权——这本身就证明
「比最终输出 + 单一容差」在这里不成立。

### 第 2 类：sort / topk —— 正确性根本不是数值量

性质和 softmax 完全不同。softmax 输出连续值，至少还能谈「数值距离」；sort/topk 输出是**一组离散选择**
（被选中元素的下标集合），`|cand − ref|` **根本没有定义**。所以最根本的问题是：「跟 reference 比数值误差、卡容差」
这招在这里压根套不上，你被迫换 surrogate metric，最顺手抓到的是 **recall**（集合 overlap）——但它有坑。

#### radix select 在干什么（讲清机制，否则「打平/换序/分布依赖」只是散乱现象）

top-k 常用 radix select：先把 float 做**单调 bit 变换**（正数翻符号位、负数全翻），让无符号整数比大小
等价于 float 比大小。然后从最高位起每 8 bit 算一个 digit（256 进制位）：对当前 digit 做直方图；
从高桶往低桶累加计数，找到累计跨过 k 的桶 = **边界桶（boundary bucket）**；比它高的桶铁定入选，
比它低的铁定出局；**只对边界桶**带上接下来 8 bit 递归细分。32-bit 数排满 4 轮（32/8）得到精确 top-k。

#### 近似版错在哪：所有错误只发生在边界桶

近似版（flashinfer / vortex 那种 approx top-k）只排 2 轮就停（只定高 16 bit，后 16 bit 不递归），
边界桶里的元素直接糊弄。**关键：边界桶里元素高 16 bit 完全相同 = 值非常接近，近似版分不出谁大谁小，
要么全收要么乱挑。** 所以错误**结构性地只发生在 cutoff 附近那条带**——远高于/远低于阈值的永远判对，
只有「卡在第 k 名附近、值彼此极接近」的那一撮才会乱。

这一句把三条散乱现象收编成同一件事：
- `torch.randn()` 输入下值连续分散，cutoff 附近几乎没有近到这种程度的两个值，**边界带没几个元素，recall ≈ 100%**
  ——看着完美，但这个 kernel 本就不返回精确值，它「过关」不是因为正确，**而是因为 randn 这道题太简单，没考到软肋**。
- **BF16 / FP8 输入下**，低精度把大量本来不同的值**压成相等（打平）**，一堆元素挤进同一个 16-bit 桶，
  **边界带变胖，recall 暴跌。**
- 所以「打平」和「分布依赖」不是两件事，就是**「边界带里塞了多少元素」**这一件事。

#### 为什么这会击穿精度 verifier（两层）

1. **输出是 index 集合**，被迫换 metric，而不同 metric 给不同判决：按选中**值**比，边界带选错一个几乎零误差
   （选了 0.998 而非 0.999）；按 index **recall** 比，同一次就是实打实的 miss。两个都不绝对正确。
2. **边界处 ground truth 本身不唯一**：值打平时多个 top-k 集合一样对，你的 reference（稳定排序）只是钦定了
   一种 tie-break；候选按另一种顺序选了另一组，两组其实一样对（选中值相同），只是 index 不同，
   recall 一比 < 100% → **bit-exact 正确的 kernel 都可能被冤枉。** 这就是本类 false reject 的来源：
   不是 kernel 错，是量化打平 + 指标选择联手把它判错了。

#### 真正该带走的：不是所有 miss 都一样

退化成用 recall 当判据会踩更深的坑：**recall 把所有 miss 等权，但不同 miss 危害天差地别。**
判一个 miss 有没有害，标准**不是被丢元素的绝对大小，而是被丢的值和保留下来的值之间的 gap**：

- **漏掉骑在边界、值≈cutoff 的元素 = 无害**（设计内近似误差，不是 bug）。近似版只能搞错和第 k 名值差距在
  分辨率以内的元素，这撮元素和被选中的 tie-partner 值几乎相等，下游 softmax 权重也几乎一样。
  对照：选中集合最小 logit 是 5.0，丢了 4.998、补进 5.001，`exp` 差千分之几，`o` 纹丝不动。
- **漏掉远高于 cutoff 的大值 = 真 bug，后果灾难性。** 明显大的值高 bit 明显大，第一轮就铁定入选，
  正常永远不会被丢。它要是真被丢，只可能来自实现 bug：bit 变换没处理好符号位/负数/NaN/Inf、
  直方图边界 off-by-one、并行 reduce 的 race、按 fp32 读 bit 但数据其实是 bf16 导致高位读歪……
  例如 logit=12 的 key 被错丢、换进 logit=3 的，`exp(12)` 主导项没了，`o` 整个塌掉。

**重要的不对称**：边界 miss 是近似算法的正常产物；远高于 cutoff 的 miss 它结构上**根本产生不了**，
所以后者一旦出现基本等于有实现 bug——**但这种 bug 产出又大又明显的误差，任何阈值都一抓一个准，
属于 verifier 最不费劲的区域，不是它会被骗的地方。** verifier 真正的痛点在**灰区**：误差小到分不清
「可接受的近似」和「真的算错」。对应 topk，真正危险的 false accept 不是某个 coding bug，而是：
**一个过度激进的近似，只在 randn 这种 benign 分布上验过、下游误差很小、看着就对，于是被放过；
换到真实分布（有结构、有打平、有主导 key 长尾）就会漏掉重要 key。** 它的问题不是「写崩了」，
而是「只在送分分布上验过」——这种 **distribution-dependent false accept** 才是要拿去喂 Claude Code 构造的东西。

**正确判法**：不是「数对了几个 index」，而是按被丢元素相对保留集合的 **value-gap 加权**：gap≈0 不罚、gap 大重罚
——天然分开「无害边界 swap」和「灾难性大值漏选」。极端做法：干脆别比 index，直接比最终下游输出
`softmax(q@k[I]/√d)·V[I]`，让正确性回到有数值距离的量上。

#### 真实代码对照：`vortex_torch` 的 approx top-k

机制可在 [Infini-AI-Lab/vortex_torch](https://github.com/Infini-AI-Lab/vortex_torch)（分支 v0.5，
`custom_ops/topk_output/flashinfer/approx/`）的 `kernel.cu` 逐行对上。它是自适应 1~2 轮的 8-bit radix select
（同目录 default 是 CUB 全排序精确版，k_96/128/256 是小 k 精确 radix 版，只有 approx 近似）。
`tolerate_ratio` 经 `dispatch.py` 烤成编译期常量 `__TOLERATE_RATIO__`：`=1.0` 永远单轮（最糙），
`=0.0` 边界桶有歧义就走两轮（最紧，默认值）。

`score_to_key32` 把 fp32 做单调 bit 变换（正数置符号位、负数全翻，符号位/负数都处理对了 →「bit 变换写错」
在这里不是隐患）。Pass 1 对最高字节 `>>24` 做直方图 + 反向前缀和，找到计数跨过 k 的边界桶 `tbin0`，
算还需从中抓 `last_remain0` 个；若 `last_remain0 <= tolerate_ratio*k` 单轮直接填，否则用次高字节 `>>16` 再细分一轮。

**emit 是理解全部近似性的关键**，机制是 `atomicAdd` 返回「加之前的旧值」= 领号机：

```c
if (bin > tbin0) {                                   // strict winner（高位 bit 明显大）
    const int pos = ::atomicAdd(&s_counter, 1);      // 从 0 往上领号
    index[pos] = idx;                                // 填到 index 前段，一个不少
} else if (bin == tbin0) {                           // 边界桶（高位 bit 跟保留项相同 = 值极接近）
    const int pos = ::atomicAdd(&s_last_remain, -1); // 从 last_remain0 往下领号
    if (pos > 0) index[target_k - pos] = idx;        // 抢到正号的填后段，号发完就丢
}
```

两个计数器从两端往中间填，正好拼满 `target_k`、不重叠，这部分精确正确。**近似只发生在边界桶**：
预算 `last_remain0` 个坑发完，后到线程 `pos<=0` 直接不写，留谁丢谁纯看 `atomicAdd` 谁先执行（**race**）。
三个结论直接对上前面的论断：

1. **strict winner 永不丢**——高位 bit 明显大，第一轮铁定入选，从构造上保证「所有 miss 都是边界 miss」。
   verifier 一旦发现 strict winner 被丢 = 真 bug，不是近似。
2. **index recall 当判据根本不行**：race 顺序不确定，同一输入跑两次选出的 index 集合都可能不同，
   它连「自己跟自己比」都过不了 100%，判对错只能比选中值或下游 `o`。
3. **`tolerate_ratio` 就是「近似激进程度 / false-accept」旋钮**：randn 下边界桶瘦，`ratio=1.0` 也没事；
   BF16/打平下边界桶巨胖，`ratio=1.0` 会把一大把任意顺序的 tie 糊进去——**distribution-dependent false accept**。
   验证应把 `tolerate_ratio` 当被测维度，在打平/低精度分布下扫它，看输出何时开始崩。

**一个 BF16 细节正是 false reject 的代码出处**：BF16 先 `__bfloat162float` 升 fp32（低 16 bit 补零），
所以两轮（16 bit）对 BF16 其实**精确**，但 BF16 真·相等的打平极多，这些全等值进边界桶按 race 填，
对稳定排序参考做 recall 必然 < 100%——**kernel 对，是指标在冤枉它。**

### 第 3 类：softmax / activation + FP8 —— 把「压扁」编进数字格式

第 1 类的压扁**再叠一层低精度**。先抓住低精度格式的关键性质：能表示的数特别少、量化格子特别粗。
FP8 的 E4M3 只有 3 个尾数位，每个 2 的幂区间里只能表示 **8 个值**。格子一粗有两个直接后果——
两个差得比格子还小的值被 snap 成同一个码（**强制打平**）；小于最小可表示档位的值直接变 0（**flush to zero**）。
这两条是第 3、4 类所有坑的根。它比单独的第 1 类更狠，因为「压扁」现在编进了**数字格式本身**。

**双重抹掉 → 更强 false accept。** softmax 已把尾部概率压到很小（0.001 / 0.0012 / 0.0008 量级），
再用 FP8 存这些值，这个量级附近 FP8 能表示的码就那么几个，三个本来不同的概率可能全被 snap 成同一个 FP8 码、
或都变 0。**「尾部到底算对算错」在 FP8 里根本编码不出来**——一个尾部全错的 kernel，输出跟正确的逐 bit 相同，
**误差严格为 0**，白送过关。第 1 类是 softmax 运算藏错误（值仍全精度）；第 3 类是**连藏的空间都没了，格式本身表示不了那点差别**。

**逼你开大容差 → 踩中 preflight。** 反过来，一个正确的 FP8 kernel 跟 fp32 reference 比，误差本来就大
（3 个尾数位，exp/求和的量化和累加顺序都引入难 bound 的误差）。为让它通过你必须开大容差——可一旦开大，
**「正确但 lossy 的 FP8 kernel」和「真有 bug 的 kernel」落进同一容差带，分不开了。** 这正是 §3 的核心两难。

### 第 4 类：low-bit FP4 / INT4 —— 两个方向同时崩到底

第 3 类推到极端，格子粗到离谱：FP4 的 E2M1 每区间只有 **2 个**可表示值，INT4 全程共 **16 个**等距档位。

**误差直接 round 成 0 → false accept 到极致。** 即 Zhuoming 说的「差个 0.0X 直接消失」。两值只要差得比量化步长小
（4-bit 下步长巨大），量化后 snap 成完全相同的码。一个高精度下会出问题的算法、或算出略微不同中间结果的 kernel，
在 FP4/INT4 下给出逐 bit 一样的输出，**误差严格为 0**，毫无悬念通过。这次不是运算藏错误，是**表示根本没地方放下这个差别，bug 彻底隐形**。
另外低于最小档位的小值还会 flush 成 0，作用在这些小值上的计算与输出无关，处理错了也看不见。

**正确 kernel 却跟参考差到天上 → false reject 到极致。** 一个正确的 INT4 kernel 跟 fp32 reference 比误差大得惊人，
任何「紧」容差都会把所有实现全 reject。到这一类，**拿 bit 级数值误差判这条路基本走不通**：要么做量化对量化的比较
（但参考又取谁？），要么放弃逐元素误差，改用**下游 / 任务级指标**（看模型整体输出还好不好）。

**第 1、3、4 类是递进**：第 1 类 = 运算藏小值错误（值全精度）；第 3 类 = 运算藏 + FP8 表示不了那点差别 + 逼开大容差；
第 4 类 = 格子粗到误差直接归零（放过坏的到极致）+ 正确 kernel 跟 fp32 差太远（冤枉好的到极致），数值误差判据彻底失效。
第 2 类是另一根轴：正确性根本不是数值量，连「用误差衡量」的前提都不成立。

---

## 3. 贯穿四类的核心两难：preflight 的容差 trade-off

现在流程是 preflight：候选 kernel 不满足数值误差就直接 filter 掉（见 [README.md](README.md) §3.3 recheck）。
这里有个**根本性、无法用一个阈值卡死**的 trade-off：

- **一边是 false reject**：低精度 kernel（FP8/INT4、近似 topk）本来对，但很难满足紧的数值误差。为让它过，得**开大容差**。
- **另一边是 false accept**：容差一开大，它就不再是「数值误差」，而变成「实际误差」——一个真算错的 kernel
  可以用别的方式钻这个松掉的容差蒙混过关。

**本质矛盾**：有些 kernel 满足数值误差但其实错了，有些不满足却是对的；为救 topk/FP8「实际对却被判错」而放宽容差，
代价是放进更多假阳性。这两类（该放过的 lossy kernel、该杀的 bug）落在同一容差区间，**单调一个 magic number 永远分不开**。

**所以 preflight 不该是一个标量阈值，而应是 multi-agent 判定逻辑里专门处理的一个分支：按算子类型切换判据**——
连续输出比数值误差、topk 这种选择类按 value-gap 加权或直接比下游输出、低比特退化到任务级指标，
**而不是所有 kernel 共用一个 allclose 容差。** 这是本文档对 [verifier/compare.py](verifier/compare.py) 现状的核心改造诉求。

---

## 4. 综合目标：sparse attention（一个现成的多盲区靶子）

sparse attention 把前面几类叠在一起，是个能同时踩到多个盲区的现成验证目标。

dense attention 不用 top-k：`o = softmax(q@kᵀ/√d)·V`，每个 query 对所有 key 算权重。
top-k 出现在 sparse attention 里省算力：长上下文 key 特别多（几万、几十万），但 softmax 出来只有少数 key 权重大，
对所有 key 都算太浪费。于是先看分数 `q@k/√d`，只挑分最高的 k 个 key，只对这 k 个做 softmax 和加权和：

```
o = softmax(q@k[I]/√d) · V[I],   I = top-k 选出的 2000 个 key 的下标
```

你要验的那个 kernel（Quest、vortex 这一类高效推理方向）干的就是这件事。

**迷惑性在于它同时踩多个点**：`I` 这一步是**第 2 类**（top-k 选择），后面 softmax 是**第 1 类**（压扁），
而测试图省事用的输入又正好是 `torch.randn()`——所有分布里最 benign、最送分的那个。
测试时手头没真模型也不需要，只要形状和 dtype 对的数就能跑 kernel，于是直接 `torch.randn(B,H,N)` 当假 logits。
但真实 attention logits 根本不长这样：**有扎堆、有主导 key 的长尾，BF16/FP8 下还有大量打平。**
结果：一个只在 randn 上验过的近似 attention kernel，可能漏掉权重大的重要 key，
但 verifier 因为只在 benign 分布上测过，把它放了过去——**distribution-dependent false accept**，
且这里 top-k 的压扁和 softmax 的压扁还会叠加，盲区更深。

---

## 5. 下一步计划（落到行动）

### 5.0 贯穿设计的两条原则（力气往哪使）

1. **灰区优先，别在粗暴 bug 上浪费火力。** strict-winner 被丢、大值漏选这类**实现 bug 产出又大又明显的误差，任何容差都一抓一个准**——那是 verifier 最不费劲、最不会被骗的区域。真正的痛点全在**灰区**：误差小到分不清「可接受的近似」和「真的算错」，典型就是 distribution-dependent false accept（只在 randn 上验过的过度近似）。所有对抗构造和裁判升级都应瞄准灰区，而不是去重复抓那些数值阈值本来就能抓的东西。
2. **近似激进度旋钮当被测维度去扫。** 当被验 kernel 带「近似程度」参数（如 vortex 的 `tolerate_ratio`），不要只在默认值上测——在**打平/低精度分布**下扫这个旋钮，看输出从哪个值开始崩。这把「它在送分分布上看着对」和「它实际有多激进」分开。

### 5.1 给 Claude Code 的对抗测试构造任务（四类 × 两方向）

统一反思路：**别再拿 `torch.randn` 这种送分输入测**（它又连续又分散、几乎不打平），
而是专门构造能踩中各类软肋的分布。每一类按「往哪个方向失效 + 触发它的具体输入构造」两栏写。

| 类 | false accept 构造（验「放过坏的」） | false reject 构造（验「误杀好的」） |
|---|---|---|
| **1. softmax/activation** | 让主导项和尾部 logit 差距极大，往**尾部/死区**注入错误，看 verifier 是否放过 | 给**主导项**加小扰动（0.5 量级），看是否被 `exp` 放大成误杀 |
| **2. sort/topk** | 构造**只在 randn 上看着对、换真实分布漏掉大值**的激进近似（扫 `tolerate_ratio`）；区分「无害边界 swap」vs「漏掉远高于 cutoff 的大值」 | 构造大量**打平 + cutoff 附近扎堆**（含 BF16/FP8 量化后离散值），让 bit-exact 正确 kernel 因 index recall<100% 被冤枉 |
| **3. softmax+FP8** | 构造**尾部概率全落进 FP8 同一个码或 flush 到 0** 的输入，验证尾部错误被双重抹掉（误差严格为 0） | 确认正确 FP8 kernel 是否因容差太紧被误杀 |
| **4. FP4/INT4** | 构造**真实差距 < 量化步长**的输入，验证误差是否被 round 成 0 而隐形 | 确认逐元素误差判据是否已退化到必须改用下游/任务级指标 |

### 5.2 代码改动点（把上面的诉求落进现有架构）

现有架构见 [README.md](README.md) §3。改动按「单一容差 → 按算子类型分支判据」这条主线：

0. **新增 [verifier/classify.py](verifier/classify.py)（分诊台）**：在判对错**之前**先判算子属于哪类。只跑可信 reference（problem.txt 的纯 PyTorch），CPU + 小输入，不碰 kernel/Triton/GPU。检测层：`D1` dtype（→低比特轴）、`D2` 输出整型/「输出是输入子集」（→选择类）、`D3` 增益探针（保数量级 vs 压扁）、`D4` 指纹（softmax 和为1/范围、饱和扫描、跳变）。产出 `{op_class, precision, judge, signals}` 喂给 compare/recheck 选判据、喂给 debate。指纹全不触发才回退 LLM。
1. **[verifier/compare.py](verifier/compare.py)**：从单一 `compare_outputs` 升级成**按算子类切换的判据集**：
   - 连续保数量级类（matmul / elementwise）→ 维持现数值误差。
   - 压扁小值类（softmax / activation）→ 支持在 activation **之前**比、或对被压区域加权。
   - 选择类（sort / topk）→ **value-gap 加权**判据，或直接比下游输出；**禁用裸 index recall 当判据**。
   - 低比特类（FP8 / FP4 / INT4）→ 量化感知 reference / 退化到任务级指标。
2. **[verifier/recheck.py](verifier/recheck.py) 的固定电池**：现固定 case 用的输入（standard/noncontig/odd/empty）
   全是 benign。**新增「分布维度」**：clustered logits、tied/打平值、量化后离散值、主导 key 长尾——
   把 §5.1 的构造固化成可复现的固定项（呼应 README §7 limitation #1「skeptic 临场想 → 不稳定」）。
3. **[agents/skeptic.py](agents/skeptic.py) / [agents/judge.py](agents/judge.py)**：
   skeptic 发 claim 时**带上算子类**并按本类软肋出对抗输入；judge 的严重性裁断**按算子类分支**
   （连续/选择/低比特用不同「这算 bug 还是预期近似」的标准）。这正是 §3 说的
   「preflight 应是 multi-agent 里专门处理的一个分支」。

### 5.3 红队数据集 entry（acceptance test）

把四类各做成 `dataset/_advprec_*` 负样本（与 roadmap §10 Tier3 的 `_hack_*` 同结构、同验证逻辑），
每个 entry 内放一个「在 benign 分布下数值误差通过、但实际算错」的 kernel + 触发软肋的对抗分布。
跑 `kv-run _advprec_*`，**期望** 5.2 改造后的判据能抓到、改造前抓不到——这就是整套精度判据升级的可量化标尺。

### 5.4 实施顺序（建议）

0. **先建分诊台 `classify.py`**（纯 CPU、零 GPU 依赖、零回归风险），立刻在现有 entry 上验证 A/B 自动分类（elem_add→A、softmax→B、relu→B），再加一个 topk problem 验证 C。这是整套改造的地基与第一个可量化产出。
1. **再建第 1、2 类的红队 entry**（softmax 尾部错误 + topk 边界带），它们最便宜、机制最清楚，立刻给标尺。
2. **再升级 compare.py 的选择类判据**（value-gap / 下游输出），用第 2 类 entry 回归验证。
3. **recheck 加分布维度**（打平 / 长尾 / 量化离散），呼应 README §7 limitation #1 把不稳定项固化。
4. **最后碰 FP8 / FP4**（第 3、4 类）——需要量化感知 reference，工程量最大，放后面。
5. **综合靶子**：sparse attention（§4）作为四类叠加的端到端验证目标。

---

## 6. 与其它文档的关系

- [roadmap.md](roadmap.md) §10 —— 正交的「防生成器作弊」轴；本文档是「防判据失效」轴。两者都要。
- [README.md](README.md) §3.4 / §4 / §7 —— 本文档是对「单一 allclose 容差」「三结果未合并」「in-scope 未定」
  这几个已知 limitation 的正面回应与升级路径。

## 7. 已知局限（诚实清单）

精度轴这套（classify → precision_recheck → judge → combine + 类感知 debate）已端到端跑通，
但下面这些是**已知的、用启发式换工程量的取舍**。记在这里，等真撞上再改，避免误以为是无懈可击的：

1. **D3 增益探针的 `±8` 是写死的经验值**（[classify.py](verifier/classify.py) `d3_gain`）。
   它靠把坐标推到 ±8 去探"死区/饱和区"。**饱和区比 ±8 更靠外的算子（如缩放过的 logits）会探不到 →
   被误归成 preserve（漏判）**。这不是 100% 保证，是启发式。修法：把写死的 ±8 换成**自适应外扫**
   ——一直往外扫到增益趋平或撞上 dtype 范围为止。建议先留着记为局限，等真出现再改（否则过早优化）。

2. **J2 选择裁判用 `max(gap)`，只抓"最严重的那一个漏选"**（[compare.py](verifier/compare.py) `judge_selection`）。
   漏一个大值 = 灾难，它能抓；但**「漏掉很多个各自只差一点点」的累积危害它抓不到**。修法：`max`→`sum`、
   或 `count(gap > 阈值)`、或干脆别比 index 直接比下游输出 `softmax·V`（终极方案）。

3. **classify 的「缩小尺寸」启发式会在 norm 类算子上崩**（`_CapSizes` 把输入维度 cap 到 16）。
   LayerNorm 的 `normalized_shape` 是 plain list 不被 cap，输入被 cap → 形状对不上 → 报 error（被捕获、
   不是误判）。**「特征维必须匹配学习权重」和「缩小尺寸」本质冲突**。修法：per-op shape 求解器，或真实形状路径。

4. **precision_recheck 的对抗输入构造是 first-cut**（[precision_recheck.py](verifier/precision_recheck.py)）。
   它只构造**单个主输入（rows=4, n=256）、对最后一维**做对抗。只适配「一元、沿最后一维」的算子
   （softmax/topk/activation）。**二元算子（如 elem_add 收两个张量）或需要特定形状的算子无法通用处理**。
   preserve 被 skip 所以 matmul 没事，但一个多输入的 compress/select 算子会崩。

5. **precision_recheck 需要 kernel 真能跑**。真实 Triton kernel 需要 GPU；CPU-only 环境 → error（不是裁决）。
   依赖 [gpu_pick.py](verifier/gpu_pick.py) 找到可用卡（本机 GPU 0 坏，探测分配会失败，gpu_pick 自动绕开用 1–5）。

6. **对抗分布本身是固定启发式**。precision_recheck 里的「扎堆 / 重尾」是按类写死的默认分布。
   **一个专门避开这些分布出错的 kernel 仍可能漏过**——这是 false-accept「测试覆盖永远证不到 0」的根本性质：
   FR（误杀）可封闭，FA（放过）只能靠不断加对抗分布持续压、压不到绝对零。

7. **D 类（FP8/FP4/INT4）目前只会 abstain，没有真正的下游/任务级裁判（J3）**。
   abstain 路径已验证（错误在低比特里湮灭、诚实弃权而非假装通过），但「弃权之后真去比下游任务」没建。
   且 INT4/FP4 非 torch 原生 dtype，要手动量化模拟。属有意延后（见 §5.4）。

**优先级判断**：这些里没有一个是"现在就会出错"的——它们是"未来遇到某类算子/某种 kernel 才暴露"的边界。
当前 dataset 全部正确处理。建议遇到时再针对性修，而不是预先全堵（过早优化）。最可能先撞上的是 #2（多漏选累积）
和 #4（多输入算子）。
