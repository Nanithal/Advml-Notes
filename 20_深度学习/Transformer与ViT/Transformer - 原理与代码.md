---
title: "Transformer - 原理与代码"
aliases:
  - "讲解Transformer"
分类: "Transformer与ViT"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "Transformer"
  - "自注意力"
  - "Attention"
  - "深度学习"
  - "知识库/笔记"
  - "主题/Transformer与ViT"
time: "2026-10-15T22:40:00"
状态: "已整理"
摘要: "围绕“Transformer - 原理与代码”，整理句话定义与全局地图、为什么要有 Transformer：RNN 与 CNN 的瓶颈、RNN 的三个死穴。"
更新日期: "2026-10-30"
上级目录: "[[20_深度学习/Transformer与ViT/总结 - Transformer与ViT]]"
相关笔记:
  - "[[20_深度学习/Transformer与ViT/Transformer - 入门详解]]"
  - "[[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解]]"
  - "[[20_深度学习/工具与实现/PyTorch - 张量基础]]"
  - "[[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法]]"
论文条目:
  - "[[论文来源目录#transformer]]"
  - "[[论文来源目录#vit]]"
论文标题:
  - "Attention Is All You Need"
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
发表载体:
  - "NIPS"
  - "ICLR"
发表年份:
  - "2017"
  - "2021"
发表类型:
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
  - "https://openreview.net/forum?id=YicbFdNTTy"
预印本链接:
  - "https://arxiv.org/abs/2010.11929"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[20_深度学习/Transformer与ViT/总结 - Transformer与ViT|Transformer与ViT总结]]

# Transformer - 原理与代码

**Transformer 是把"注意力"这一件事做到极致的序列建模架构**：它彻底丢掉了循环（RNN）和卷积（CNN），只用**自注意力 + 前馈网络**堆叠出一个可以完全并行训练、且任意两个位置之间只隔一层网络的模型。

本文不走"论文翻译"的路子，而是按 **动机 $\rightarrow$ 直觉 $\rightarrow$ 数学 $\rightarrow$ 形状 $\rightarrow$ 代码 $\rightarrow$ 本质** 的顺序，把 Transformer 拆到能自己手写一遍的程度。

阅读路线：

```Plaintext
为什么要它 → 整体骨架 → 注意力直觉 → 数学公式（手算一遍）
        → 多头 / 位置编码 / FFN / 残差+LN → 解码与掩码
        → 形状追踪 → 复杂度 → 训练细节与结果
        → 代码实现 → 本质与代价 → 常见误区 → 演进脉络
```

---

### 一、一句话定义与全局地图

**Transformer = 编码器（Encoder）+ 解码器（Decoder）**，每一侧都由 $N=6$ 个**完全相同的层**堆叠而成。原始论文（Vaswani et al., *Attention Is All You Need*, NIPS 2017）的每个层长这样：

| 位置 | 编码器一层 | 解码器一层 |
|---|---|---|
| 子层 1 | 多头**自**注意力 | 多头**掩码自**注意力（只能看左边） |
| 子层 2 | — | 多头**交叉**注意力（Q 来自解码器，K/V 来自编码器） |
| 子层 3 | 逐位置前馈网络 FFN | 逐位置前馈网络 FFN |
| 每个子层统一包装 | `LayerNorm(x + Sublayer(x))` | 同左 |

超参数（base 模型）：

| 符号 | 含义 | 值 |
|---|---|---|
| $d_{model}$ | 所有层的隐藏维度 | 512 |
| $h$ | 注意力头数 | 8 |
| $d_k=d_v$ | 每个头的 Q/K/V 维度 | $512/8=64$ |
| $d_{ff}$ | FFN 中间层维度 | 2048 |
| $N$ | 编码器/解码器层数 | 6 |
| $P_{drop}$ | Dropout | 0.1 |

一句话记住它是怎么工作的：**每个词先变成向量，再和所有其他词"对话"（注意力）一次，然后各自过一个 MLP（FFN），重复 6 遍；解码时对每个位置屏蔽掉未来，再和编码器的输出"对话"一次。**

---

### 二、为什么要有 Transformer：RNN 与 CNN 的瓶颈

#### （1）RNN 的三个死穴

以 $h_t = f(h_{t-1}, x_t)$ 为代表的循环结构：

1. **无法并行**：第 $t$ 步必须等第 $t-1$ 步算完。序列长度 $n$ 就意味着 $n$ 次串行操作，GPU 的并行度被序列长度锁死（论文里叫 $O(n)$ 的 sequential operations）。
2. **长距离依赖要"翻山越岭"**：第 1 个词的信息要传到第 100 个词，必须经过 100 次非线性变换，梯度在途中不断衰减或爆炸。信息路径长度是 $O(n)$。
3. **显存与吞吐的矛盾**：想并行就得一次喂很多样本，但序列一长，反向传播要保存整条时间链。

LSTM/GRU 的门控只是缓解，没有根治。

#### （2）CNN 的问题

ConvS2S、ByteNet 这类用卷积做序列的模型：

- 并行性好，但**感受野要一层层堆**才能覆盖长距离：路径长度 $O(\log_k n)$ 或 $O(n/k)$；
- 卷积核只看局部窗口，远距离依赖要靠深度换；
- 每层计算量 $O(k \cdot n \cdot d^2)$，$k$ 是卷积核宽度，长序列上很贵。

#### （3）注意力给出的答案

**让任意两个位置直接相连，路径长度变成 $O(1)$，而且整段序列可以一次算完。**

| 层类型 | 每层复杂度 | 串行操作数 | 最大路径长度 |
|---|---|---|---|
| **自注意力** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| 循环（RNN） | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积（CNN） | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| 受限自注意力 | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

（$n$ 是序列长度，$d$ 是表示维度，$k$ 是卷积核宽度，$r$ 是邻域大小。）这张表就是论文的核心论点：**当 $n < d$ 时，自注意力比循环层更便宜，而且并行度碾压。**

---

### 三、注意力的直觉：从"查字典"说起

先别管矩阵。注意力要做的事只有一句话：

> **当前这个位置，应该从哪些位置、各取多少信息？**

把它想成一次**软查询（soft lookup）**：

- **Query（查询，$q$）**：我现在的需求是什么？
- **Key（键，$k$）**：每个位置"能被检索的标签"是什么？
- **Value（值，$v$）**：每个位置"真正携带的内容"是什么。

流程：

1. 拿我的 $q$ 和所有位置的 $k$ 逐一算**相似度**（点积）；
2. 相似度换成**权重**（softmax，保证非负且和为 1）；
3. 按权重对所有 $v$ 做**加权平均**，得到我的新表示。

和传统"硬查字典"的区别：字典只能命中一个词条（权重是 one-hot），注意力是**所有词条都取一点，按相关度分配比例**——所以它可微、可学习。

**自注意力（Self-Attention）** 就是 $q,k,v$ 全部来自**同一个输入序列**：序列自己跟自己对话，因此每个词的表示会被"整个句子的上下文"重新改写。这是 Transformer 的关键——它不是把上下文压成一个固定向量（像 Seq2Seq 的 encoder 那样），而是让每个词**动态地、按需地**去别的词那里取信息。

---

### 四、缩放点积注意力：公式与手算

#### （1）公式

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

逐项拆：

| 项 | 形状 | 含义 |
|---|---|---|
| $Q$ | $n \times d_k$ | $n$ 个 query |
| $K^\top$ | $d_k \times m$ | $m$ 个 key 转置 |
| $QK^\top$ | $n \times m$ | 两两相似度分数矩阵 |
| $\text{softmax}(\cdot)$ | $n \times m$ | 每一行归一化成权重（行和为 1） |
| $V$ | $m \times d_v$ | $m$ 个 value |
| 输出 | $n \times d_v$ | 每个 query 的新表示 |

#### （2）为什么要除以 $\sqrt{d_k}$

假设 $q$ 与 $k$ 的每个分量都是**均值 0、方差 1 且相互独立**，那么点积

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

的均值是 0，**方差是 $d_k$**。维度一大，点积的数值就会很大，softmax 会被推向**极端饱和区**——最大的那个分数几乎拿走全部权重，其余都是 0，梯度趋近于消失。

除以 $\sqrt{d_k}$ 后方差被拉回 1，softmax 落在"温和"的区域，梯度正常。这也解释了另一件事：**维度越大越必须缩放**，否则注意力会退化成"硬取最大值（argmax）"。

> 论文还比较了**加性注意力** $\text{score}=v^\top\tanh(W_1q+W_2k)$：两者在 $d_k$ 小时精度相当，但点积注意力能用高度优化的矩阵乘法，**更快更省显存**，所以被选中。

#### （3）手算一个 3 个 token 的例子

设 $d_k=2$，当前 query $q=[1,0]$，两个 key/value：

$$
k_1=[1,0],\ k_2=[0,1];\qquad v_1=[1,0],\ v_2=[0,1]
$$

**第一步，算分数并缩放：**

$$
\frac{q\cdot k_1}{\sqrt{2}}=\frac{1}{1.414}=0.707,\qquad
\frac{q\cdot k_2}{\sqrt{2}}=\frac{0}{1.414}=0
$$

**第二步，softmax：**

$$
e^{0.707}=2.028,\quad e^{0}=1.000,\quad \text{和}=3.028
$$

$$
\alpha_1=\frac{2.028}{3.028}=0.670,\qquad \alpha_2=\frac{1.000}{3.028}=0.330
$$

**第三步，加权求和：**

$$
\text{out}=0.670\cdot[1,0]+0.330\cdot[0,1]=[0.670,\ 0.330]
$$

结果解读：$q$ 和 $k_1$ 更像，所以从 $v_1$ 取了 67%，从 $v_2$ 取了 33%。**整个过程全是可微的乘加**，$W^Q,W^K,W^V$ 靠梯度自己学。

#### （4）PyTorch 一行版

```python
attn = torch.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d_k), dim=-1) @ V
```

---

### 五、自注意力、交叉注意力与掩码注意力

三种用法，区别只在于 **Q 和 K/V 分别来自哪里**：

| 名称 | Q 来自 | K/V 来自 | 用途 |
|---|---|---|---|
| **自注意力** Self-Attention | $X$ | $X$ | 编码器内部；解码器的掩码自注意力 |
| **交叉注意力** Cross-Attention | 解码器当前状态 | **编码器输出** | 解码时"看原文" |
| **掩码自注意力** Masked Self-Attention | $X$ | $X$（未来位置被屏蔽） | 解码器内部，保证不看答案 |

#### 掩码是怎么做的

训练时解码器一次性看到**整句目标句**，但生成第 $t$ 个词时**绝不能偷看第 $t+1$ 个词以后的内容**，否则模型学会的只是"抄答案"。做法简单粗暴：

$$
\text{scores} = \frac{QK^\top}{\sqrt{d_k}} + M,\qquad
M_{ij}=\begin{cases}0, & j \le i\\ -\infty, & j > i\end{cases}
$$

$- \infty$ 经过 softmax 变成 **严格的 0**，因此未来位置权重为零。这就是**因果掩码（causal mask）**，也叫上三角掩码。

实际代码里通常用 `masked_fill(mask == 0, -1e9)` 或 `float('-inf')`：

```python
causal = torch.tril(torch.ones(L, L)).bool()           # 下三角为 True
scores = scores.masked_fill(~causal, float('-inf'))
```

还要注意：**padding mask**（把补齐的 `<pad>` 位置屏蔽掉）要和因果掩码**取与**后一起使用，否则 batch 里长短不一的句子会互相干扰。

---

### 六、多头注意力：为什么要 8 个"小注意力"

#### （1）动机

单头注意力只有**一种**相似度度量方式，等于用一个子空间去理解所有关系。但语言里的关系是多样的：主谓、指代、修饰、位置邻近……

**多头 = 让模型在 $h$ 个不同的表示子空间里并行地做注意力**，各自关注不同的模式，最后拼接融合。类比 CNN 的多个卷积核：每个头是一种"关系探测器"。

#### （2）公式

$$
\text{head}_i = \text{Attention}\big(QW_i^Q,\ KW_i^K,\ VW_i^V\big)
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O
$$

投影矩阵维度：

$$
W_i^Q, W_i^K \in \mathbb{R}^{d_{model}\times d_k},\quad
W_i^V \in \mathbb{R}^{d_{model}\times d_v},\quad
W^O \in \mathbb{R}^{h d_v \times d_{model}}
$$

论文取 $h=8$，$d_k=d_v=d_{model}/h=64$。**关键点：把维度切小而不是复制 8 份**，所以多头注意力的**总参数量和总计算量与单头（$d_k=d_v=512$）几乎一样**，只是把"一个大视角"换成"8 个不同的小视角"。

#### （3）参数量速算（base 模型，单层）

| 模块 | 计算 | 参数量 |
|---|---|---|
| $W^Q+W^K+W^V$ | $3 \times 512\times512$ | 786,432 |
| $W^O$ | $512\times512$ | 262,144 |
| FFN | $512\times2048 + 2048\times512$ | 2,097,152 |
| 合计（每层约） | | ≈ 3.15M |

6 层编码器 + 6 层解码器 → 论文报告 base 模型约 **65M** 参数。

---

### 七、位置编码：让"词袋"重新有顺序

#### （1）为什么必须加

自注意力对输入是**置换等变**的：把输入序列打乱，输出也只是同样被打乱，注意力算出的分数矩阵完全不变。也就是说——**"我打你"和"你打我"在它眼里一模一样**。可语言、时序、图像块的位置都极其重要。

解决方式：**给每个位置额外加一个向量**（加，而不是拼接），把位置信息注入到词向量里。

#### （2）正弦位置编码

$$
PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right),\qquad
PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中 $pos$ 是位置，$i$ 是维度下标（$2i$ 表示偶数维）。

四个要点：

1. **每个维度是一对不同频率的正余弦波**，频率从 $2\pi$ 到 $10000\cdot 2\pi$ 按几何级数排列——低维变化快（编码精细位置），高维变化慢（编码粗粒度位置）；
2. **值域有界**（$[-1,1]$），和词嵌入相加不会破坏尺度；
3. **相对位置可由线性变换得到**：$PE_{pos+k}$ 可以写成 $PE_{pos}$ 乘以一个与 $k$ 有关的旋转矩阵，这给模型"相对位移"的可学线索；
4. **不需要训练、可外推**到训练时没见过的长度（虽然外推效果有限）。

#### （3）和可学习位置编码的对比

论文的消融显示：**学习式位置嵌入（learned）与正弦式几乎完全等价**（dev BLEU 25.7 vs 25.8）。作者选择正弦版本，主要是为了**能处理比训练集更长的序列**。

> 后续演进：BERT/GPT 用可学习绝对位置；GPT-2/3 沿用可学习；再往后 **RoPE（旋转位置编码）**、**ALiBi（线性偏置）**、**相对位置编码**成为主流——它们都在解决"长上下文外推"这个正弦编码没彻底解决的问题。

---

### 八、前馈网络 FFN：每个位置各过各的 MLP

$$
\text{FFN}(x) = \max(0,\ xW_1+b_1)\,W_2+b_2
$$

- **逐位置（position-wise）**：对序列里每个位置**独立地、用同一套参数**做两次线性变换 + ReLU。位置之间在这里**没有信息交换**（交换发生在注意力层）。
- 维度：$512 \to 2048 \to 512$，即 $d_{ff}=4\,d_{model}$。
- 直观理解：**注意力负责"跨位置混信息"，FFN 负责"在每个位置内部做非线性加工"**。后来的研究（如 FFN 作为 key-value memory 的解读）认为它存储了大量事实性知识。

注意：原始论文用 ReLU，后来 GPT/BERT 系普遍换成 **GELU**。

---

### 九、残差连接与层归一化

每个子层的统一封装：

$$
\text{output} = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)
$$

- **残差连接**：让梯度有一条"高速公路"，使 6~100 层的深层堆叠可训练（和 ResNet 同样的动机）。也保证了"注意力可以退化成恒等映射"这一安全起点。
- **LayerNorm**：对**单个样本的特征维度**做归一化（$\mu,\sigma$ 在 $d_{model}$ 维上统计），与 batch size 无关，因此不受变长序列、小 batch 影响——这是它优于 BatchNorm 的关键。

#### Post-LN vs Pre-LN（工程上非常重要）

| 形式 | 结构 | 特点 |
|---|---|---|
| **Post-LN**（原论文） | `LN(x + Sublayer(x))` | 表达能力强，但需要**学习率 warmup** 才能稳住，深层易发散 |
| **Pre-LN**（后续主流） | `x + Sublayer(LN(x))` | 训练稳定、几乎不需要 warmup，深层模型首选 |

现在从零训练大模型基本都用 Pre-LN（或 RMSNorm + Pre-LN）。

---

### 十、解码器如何生成文本：训练与推理的差别

#### （1）训练时（teacher forcing，一次算完）

- 目标句右移一位（前面加 `<bos>`）后**整句喂进解码器**；
- 靠因果掩码保证每个位置只能看到自己及左边；
- **一次前向就能并行计算所有位置的损失**，这是 Transformer 训练效率高的核心原因。

#### （2）推理时（自回归，一个一个吐）

```Plaintext
输入 <bos>
  → 预测第 1 个词 → 把第 1 个词接到输入后面
  → 预测第 2 个词 → 把第 2 个词接到输入后面
  → ... 直到输出 <eos> 或达到长度上限
```

所以训练是**并行**的，推理是**串行**的。现代推理引擎的优化（KV Cache、PagedAttention、投机采样）几乎都在针对这个"串行"环节：

- **KV Cache**：已生成 token 的 $K,V$ 只算一次缓存下来，新 token 只算自己的 Q，把每步复杂度从 $O(n^2)$ 降到 $O(n)$。

#### （3）束搜索（Beam Search）

论文推理用 **beam size = 4**，并加**长度惩罚** $\alpha=0.6$：

$$
\text{score}(Y) = \frac{\log P(Y|X)}{\text{lp}(Y)},\qquad
\text{lp}(Y)=\frac{(5+|Y|)^\alpha}{(5+1)^\alpha}
$$

另外还做了 **checkpoint 平均**（最后若干轮参数取平均），以及用大 batch 提高吞吐。

---

### 十一、张量形状全流程追踪（$B$ 为 batch size，$L$ 为序列长度）

**编码器：**

| 步骤 | 操作 | 形状 |
|---|---|---|
| 0 | 输入 token id | $(B, L)$ |
| 1 | 词嵌入 $\times\sqrt{d_{model}}$ | $(B, L, 512)$ |
| 2 | 加位置编码 | $(B, L, 512)$ |
| 3 | 线性投影成 Q/K/V | $3\times(B, L, 512)$ |
| 4 | 拆成 $h=8$ 个头 | $(B, 8, L, 64)$ |
| 5 | 分数 $QK^\top/\sqrt{64}$ | $(B, 8, L, L)$ |
| 6 | softmax → 加权 $V$ | $(B, 8, L, 64)$ |
| 7 | 拼头 + $W^O$ | $(B, L, 512)$ |
| 8 | 残差 + LayerNorm | $(B, L, 512)$ |
| 9 | FFN（$512\to2048\to512$）+ 残差 + LN | $(B, L, 512)$ |
| 10 | 重复 6 层 → **编码器输出 memory** | $(B, L, 512)$ |

**解码器**多两层注意力，形状逻辑相同，只是交叉注意力里 $L_q$ 是目标句长度、$L_k$ 是源句长度，分数矩阵因此是**矩形** $(B, 8, L_{tgt}, L_{src})$。

---

### 十二、训练细节与论文结果

#### （1）数据

| 任务 | 数据 | 规模 | 词表 |
|---|---|---|---|
| EN-DE | WMT 2014 | 4.5M 句对 | 约 37k BPE |
| EN-FR | WMT 2014 | 36M 句对 | 32k word-piece |

#### （2）优化器与学习率（**warmup 是原始论文的标志性设计**）

Adam：$\beta_1=0.9,\ \beta_2=0.98,\ \epsilon=10^{-9}$，并让学习率按步数变化：

$$
\text{lrate} = d_{model}^{-0.5}\cdot\min\big(\text{step}^{-0.5},\ \text{step}\cdot\text{warmup}^{-1.5}\big),\qquad \text{warmup}=4000
$$

即**先线性升温 4000 步，再按步数平方根衰减**。前半段的"热身"是 Post-LN 结构能稳定训练的关键。

#### （3）正则化

- **Dropout 0.1**：加在每个子层输出上、且**加在词嵌入与位置编码之和上**；
- **Label Smoothing 0.1**：让模型不要过度自信，代价是困惑度变差但**BLEU 变好**。

#### （4）训练成本

| 模型 | 步数 | 硬件 | 时间 |
|---|---|---|---|
| base（$d_{model}=512$，$h=8$，$d_{ff}=2048$） | 100k | 8×P100 | **12 小时** |
| big（$d_{model}=1024$，$h=16$，$d_{ff}=4096$，dropout 0.3） | 300k | 8×P100 | **3.5 天** |

#### （5）主要结果（BLEU）

| 模型 | EN-DE | EN-FR |
|---|---|---|
| ConvS2S（ensemble） | 26.36 | 41.29 |
| GNMT + RL（ensemble） | 26.30 | 41.16 |
| MoE | 26.03 | 40.56 |
| **Transformer（base）** | 27.3 | 38.1 |
| **Transformer（big）** | **28.4** | **41.8** |

- EN-DE 上 **big 模型单模型 28.4 BLEU，比此前最好的集成模型还高 2 个点以上**；
- EN-FR 单模型 41.8 BLEU，训练成本只是同类模型的零头（base 模型训练成本约 $3.3\times10^{18}$ FLOPs）；
- 这直接说明了它的价值：**不是"精度略好"，而是"精度更高 + 训练更便宜 + 可并行"三件事同时做到。**

---

### 十三、消融实验与注意力可视化

#### （1）关键消融结论

- **单头 vs 多头**：单头（$d_k=d_v=512$）dev BLEU 掉到 24.9，明显差于 8 头的 25.8 → **多头确实有用，而且不是简单增加参数**；
- **头数与头维度**：$h$ 太小或太大都变差，$h=8,\ d_k=64$ 是甜点；
- **模型维度**：加大 $d_{model}$ 与 $d_{ff}$ 都稳定提升（$d_{model}=1024$、$d_{ff}=4096$、dropout 0.3 可达 dev 26.4）；
- **Dropout 必须要有**：$P_{drop}=0$ 明显变差，$0.1\sim0.2$ 最佳；
- **位置编码方案几乎无差别**：正弦 25.8 vs 可学习 25.7。

#### （2）注意力可视化（论文最直观的部分）

把 8 个头的注意力权重画出来，可以看到：

- 有的头学会了**跟踪句法结构**（如动词-宾语、介词短语）；
- 有的头学会了**指代消解**——例如 "The animal didn't cross the street because **it** was too tired" 中，`it` 对 `animal` 的注意力显著更高；
- 多头呈现**分工**：不同头关注不同尺度/语法角色。

#### （3）额外验证：英语成分句法分析

用 4 层 Transformer 做**仅 WSJ 数据**的句法分析：**91.3 F1**；再用约 17M 句半监督数据预训练后：**92.7 F1**，超过不少 RNN 语法分析器。这说明 Transformer **不只擅长翻译**，其表征具有通用性——这也是后来 BERT/GPT 把"Transformer + 大规模预训练"变成通用范式的前奏。

---

### 十四、PyTorch 从零实现（可直接跑通的最小骨架）

```python
import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)      # (B, h, Lq, Lk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = scores.softmax(dim=-1)                          # 行和为 1
        return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被头数整除"
        self.h, self.d_k = h, d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.attn = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        # (B, L, d_model) -> (B, h, L, d_k)
        Q = self.w_q(q).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.h, self.d_k).transpose(1, 2)

        out, attn = self.attn(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(self.drop(out)), attn


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    """Post-LN 版本（原论文写法）；改成 Pre-LN 只需把 LayerNorm 挪到子层输入前。"""

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, mask)[0]))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(d_model) for _ in range(3))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        # 1) 掩码自注意力：只能看左边
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)[0]))
        # 2) 交叉注意力：Q 来自解码器，K/V 来自编码器输出
        x = self.norm2(x + self.drop(self.cross_attn(x, memory, memory, src_mask)[0]))
        # 3) 逐位置 FFN
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)        # 偶数维 sin
        pe[:, 1::2] = torch.cos(pos * div)        # 奇数维 cos
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def causal_mask(L, device=None):
    """下三角为 1，其余为 0；masked_fill(mask==0) 屏蔽未来"""
    return torch.tril(torch.ones(L, L, device=device)).bool()


if __name__ == "__main__":
    B, L, d_model = 2, 6, 512
    x = torch.randn(B, L, d_model)
    mha = MultiHeadAttention(d_model, h=8)
    out, attn = mha(x, x, x, causal_mask(L))
    print(out.shape, attn.shape)      # torch.Size([2, 6, 512]) torch.Size([2, 8, 6, 6])
```

**上手建议**：先跑通上面这段，再自己补 `Encoder`/`Decoder` 的 6 层堆叠 + 词嵌入 + 输出投影 + `CrossEntropyLoss`，用一个小规模复制任务（如反转序列、加法）验证训练是否收敛。这类玩具任务能在几秒内暴露 mask 写错、维度转置错误、位置编码没加等典型 bug。

---

### 十五、为什么 Transformer 能赢：本质与代价

#### （1）四个本质优势

1. **路径长度 $O(1)$**：任意两个位置直接交互，长距离依赖不再"翻山越岭"；
2. **训练完全并行**：整条序列一次前向，GPU/TPU 利用率高，这才是"能训到 100B 参数"的前提；
3. **表达能力强且结构统一**：同一套模块既能建模语言、也能建模图像块、音频帧、蛋白质序列；
4. **局部性不是硬编码的**：位置关系靠位置编码与学习得到，因此**可塑、可迁移**——代价是数据饥渴（弱归纳偏置）。

#### （2）三个必须知道的代价

| 代价 | 具体表现 | 后续对策 |
|---|---|---|
| **$O(n^2)$ 复杂度** | 长度 $n$ 的序列要算 $n\times n$ 的分数矩阵，显存与算力平方增长 | 稀疏注意力、Longformer/BigBird、线性注意力、**FlashAttention**（不改变数学，只优化 IO） |
| **缺少归纳偏置** | 小数据上不如 CNN/RNN，需要大规模预训练 | 蒸馏（DeiT）、强数据增强、自监督（MAE/DINO） |
| **位置信息是"外加"的** | 绝对位置编码外推差；长度超出训练范围性能骤降 | 相对位置、**RoPE**、ALiBi、长度插值 |

---

### 十六、常见误区 Q&A

**Q1：Q、K、V 是三个不同的输入吗？**
不是。它们都来自**同一个**输入张量，只是各自乘了不同的投影矩阵 $W^Q,W^K,W^V$。交叉注意力里 Q 与 K/V 才来自不同来源（解码器 vs 编码器）。

**Q2：多头是不是把模型变大了 $h$ 倍？**
不是。$d_k=d_v=d_{model}/h$，**总维度守恒**，参数量和 FLOPs 与单头基本相同。多头是"分视角"，不是"堆参数"。

**Q3：为什么除以 $\sqrt{d_k}$，而不是 $\sqrt{d_{model}}$？**
缩放针对的是**点积的方差**，点积长度等于 $d_k$，所以除以 $\sqrt{d_k}$。用 $d_{model}$ 会过度压制分数（多头场景下两者正好差 $h$ 倍）。

**Q4：位置编码为什么是"加"而不是"拼接"？**
相加保持维度不变、参数不增；高维空间里"加"已经足够让模型分辨位置模式（消融实验显示拼接式与相加式效果差别很小）。

**Q5：Decoder 的自注意力和 Cross-Attention 有什么区别？**
自注意力：Q/K/V 都来自解码器自身，且**必须加因果掩码**；交叉注意力：Q 来自解码器，K/V 来自编码器输出，**不加因果掩码**（原文全部可见），只加 padding mask。

**Q6：训练时解码器一次性输入整句，不算"作弊"吗？**
不算。因果掩码保证了第 $t$ 个位置**看不到**第 $t$ 个之后的 token，等价于逐个生成，但计算并行。

**Q7：LayerNorm 为什么不用 BatchNorm？**
LayerNorm 在**单样本的特征维度**上统计，不依赖 batch；变长序列、小 batch、推理时行为一致，这些都是 NLP 的刚需。

**Q8：Transformer 就是 BERT/GPT 吗？**
不是。Transformer 是**架构**（2017），BERT/GPT 是基于它的**预训练范式**（Encoder-only / Decoder-only，2018-2020）。架构相同，训练目标和用法不同。

---

### 十七、演进脉络（一页看完）

```Plaintext
2017  Transformer（Encoder-Decoder，机器翻译）        ← 本文
2018  GPT-1（Decoder-only，生成式预训练）
2018  BERT（Encoder-only，掩码语言模型，双向）
2019  GPT-2 / Transformer-XL（长文本）
2020  GPT-3（175B，上下文学习）/ T5（统一 text-to-text）
2020  ViT（把图像切成 patch 当 token）
2020  DeiT（蒸馏解决数据效率）/ Longformer、BigBird（稀疏注意力）
2021  Swin（层级 + 窗口注意力）/ CLIP（图文对比）
2021  RoPE、ALiBi（更好的位置编码）
2022  FlashAttention（IO 感知的精确注意力）/ MAE（掩码自编码）
2022+  PaLM、LLaMA、Mistral、Qwen…（Pre-LN/RMSNorm + RoPE + SwiGLU + KV Cache）
2023+  多模态（GPT-4V、LLaVA）、MoE 稀疏化、超长上下文
```

**看懂 2017 年那篇论文，等于拿到了后面所有模型的阅读说明书。**

---

### 十八、一页速查表

| 项目 | 内容 |
|---|---|
| 核心公式 | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| 三种注意力 | 自注意力 / 掩码自注意力 / 交叉注意力 |
| 关键超参 | $d_{model}=512,\ h=8,\ d_k=d_v=64,\ d_{ff}=2048,\ N=6$ |
| 参数构成 | 每层 ≈ 3.15M（注意力 1.05M + FFN 2.1M） |
| 位置编码 | $\sin/\cos$，频率 $10000^{2i/d_{model}}$，与可学习版等价 |
| 子层封装 | `LayerNorm(x + Sublayer(x))`（Post-LN；现代多用 Pre-LN） |
| 训练配方 | Adam($\beta_2=0.98$) + 4000 步 warmup + dropout 0.1 + label smoothing 0.1 |
| 复杂度 | 自注意力 $O(n^2d)$、串行 $O(1)$、路径长度 $O(1)$ |
| 结果 | EN-DE 28.4 BLEU / EN-FR 41.8 BLEU（big，单模型） |
| 一句话 | **用注意力把任意两位置直接连起来，用并行换掉循环，用数据换掉归纳偏置。** |

### 关联笔记

- [[20_深度学习/训练基础/Inductive Bias - 归纳偏置|Inductive Bias - 归纳偏置]]：Transformer 的"弱归纳偏置"及其代价
- [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]：把本架构直接搬到图像分类的 ViT
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程|ViT Attack - 发展历程]]：注意力架构的对抗鲁棒性视角

<!-- knowledge-links:start -->

## 笔记联系

- [[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]]：需要直觉时回到入门笔记。
- [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]]：把注意力公式对应到张量形状与代码实现。
- [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]]：进一步区分 Encoder 与 Decoder 的职责。
- [[20_深度学习/工具与实现/PyTorch - 张量基础|PyTorch - 张量基础]]：需要补充张量概念时阅读 PyTorch 基础笔记。
- [[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法|SGM - 跳跃梯度方法]]：以结构感知攻击为例理解残差连接的反向路径。

## 论文出处

- [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)：NIPS，2017，会议论文。
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)：ICLR，2021，会议论文。
  [预印本](https://arxiv.org/abs/2010.11929)。

<!-- knowledge-links:end -->
