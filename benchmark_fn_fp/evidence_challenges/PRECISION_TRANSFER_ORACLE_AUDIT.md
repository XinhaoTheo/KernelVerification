# case_76–case_81 construction and oracle audit

2026-09-24。独立只读核验完成，没有修改案例、追加 GPU 或模型调用。**未发现阻止评测的数学、标签或公开材料一致性问题。**

固定新区间为 **203600–204111，共 512 个候选**，与 case_74/case_75 的 193600–193855 无交集。[完整搜索记录](private_data/search_precision_transfer.json)按顺序保留全部候选。[新 family](families/precision_transfer.py)的输入生成器、Neumaier 仿真、参考函数、指标函数逐 AST 比较与原机制一致，公开 kernel 模板及合同也完全相同；变化仅为种子区间和预设平衡选择规则。

对全部 **2,048 行**另用 Python `Fraction.from_float` 逐项精确求和，结果与 `math.fsum`、私有整数 dyadic oracle、保存的两个参考逐项完全相同。每个 Fraction 结果转 FP64 后仍能精确回转为同一 Fraction，没有末次转换误差。另以 Python `struct.pack/unpack('f')` 实现逐步 FP32 舍入的 Neumaier 执行，512 个候选输出全部与保存仿真相同；Python 标量顺序 FP64 累加也复现全部零参考。独立重算相对误差的最大差异为 `2.22e-16`，不影响标签或选择条件。

符合选择条件的候选为 **9 个合格、401 个非零输出不合格、102 个全零输出不合格**。严格按各组升序取前三、前二和首个，完全复现以下分配：

| 案例 | Seed | 真值 | 相对误差（约） | 输出性质 | 初始 FP64 probe |
|---|---:|---|---:|---|---|
| case_76 | 203635 | trust | 7.975729e-8 | 非零，正确补偿和 | reject |
| case_77 | 203600 | reject | 0.885630952 | 非零，丢失部分小量 | reject |
| case_78 | 203795 | trust | 3.378736e-8 | 非零，正确补偿和 | reject |
| case_79 | 203601 | reject | 0.661099153 | 非零，丢失部分小量 | reject |
| case_80 | 203820 | trust | 5.725814e-8 | 非零，正确补偿和 | reject |
| case_81 | 203604 | reject | 1 | 全零 | trust |

阈值为 `1e-5`；三个合格例低于半阈值，三个不合格例超过两倍阈值。case_77/case_79 的非零失败输出确保“非零即合格”不能解释整个 cohort；case_81 保留零输出但错误的对照。这里审查的是构造规则符合预设，不代表此机制已经证明 debate 优势。

六份公开 kernel 与模板替换唯一 seed 后逐字一致；AST 提取公开生成器后重新执行，得到的输入字节与私有生成器及 answer key 哈希一致。公开文件不含私有真值，源码/合同均小于 12,000 字符。真实初始 T4 probe 已按现有流程附在合同后，附加后的 problem 哈希与 answer key、[冻结 GPU 记录](private_data/validation_gpu.json)一致。

冻结记录六例均为 Tesla T4、十次重复、输出一致；实际输出逐值等于 CPU 仿真，精确参考、输入字节、shape/dtype/finite、输入不变性及初始 probe 结论均匹配。case_77 的最后归一化指标在两环境相差 `2.22e-16`：CPU 为 `0.8856309517807359`，冻结 T4 记录为 `0.8856309517807361`；输出与参考本身完全一致，因此只是末端 FP64 数值差异，无边界或标签疑问。

此审查仅验证源码、数值 oracle、候选范围、选择及冻结一致性；GPU 运行由 root 的既有任务完成。本审查没有从模型反馈改选 seed，也没有替换任何实验槽位。
