---
tags:
  - Transformer
  - Encoder
  - Decoder
  - 自注意力
  - 交叉注意力
time: 2026-10-15T23:05:00
---
上一篇 [[讲解Transformer]] 讲的是 Transformer 这套"积木"本身；这一篇只回答一个问题：**Encoder 和 Decoder 到底各自在干什么，为什么要有两个，什么时候只需要一个。**

先给一张表，后面全部围绕它展开：

| 维度 | **Encoder（编码器）** | **Decoder（解码器）** |
|---|---|---|
| 一句话 | 把输入**读成**一组带上下文的向量 | 在"只能看左边"的约束下**一个词一个词往外写** |
| 注意力可见性 | **双向**（每个位置能看到全句，包括左右） | 自注意力**因果掩码**（只能看自己及左边） |
| 注意力种类 | 只有自注意力 | 自注意力 + **交叉注意力**（看 Encoder 输出） |
| 输出是什么 | $n$ 个上下文化向量（memory） | 每个位置一个词表分布 logits |
| 训练目标 | 掩码语言建模 / 去噪重构 | 自回归下一词预测（teacher forcing） |
| 训练并行度 | 完全并行 | **训练并行，推理串行** |
| 每层参数量（$d_{model}=512$） | ≈ 3.15M | ≈ 4.2M（多一组交叉注意力） |
| 典型代表 | BERT、RoBERTa、ViT | GPT、LLaMA、原始 Transformer 的解码器 |

---

### 一、为什么需要"编码器 + 解码器"这两个角色

根源在于**任务的形状不同**：

- **理解型任务**（分类、命名实体识别、抽取式问答、检索）：输入是一整段完整文本，**没有"未来"的概念**，因此每个位置理应看到左右两侧的全部信息 → 需要**双向**处理 → **Encoder**。
- **生成型任务**（翻译、摘要、对话、代码补全）：输出是一个词一个词产生的，生成第 $t$ 个词时**绝不能看到第 $t$ 个之后的词**，否则训练时会直接抄答案 → 需要**因果**约束 → **Decoder**。
- **序列到序列任务**（翻译、语音识别、摘要）：既有"完整读入的源序列"，又有"逐步生成的目序列"，两个需求叠加 → **Encoder + Decoder**。

原始论文（*Attention Is All You Need*）服务的正是第三类：**机器翻译**。所以它天然是 Encoder-Decoder 结构。

> 一个容易忽略的点：Encoder 和 Decoder **共享几乎全部组件**——多头注意力、FFN、残差 + LayerNorm、位置编码、dropout 全是一样的。真正的差别只有三处：**掩码、多不多一路交叉注意力、损失函数**。理解了这三处，就理解了 Encoder 与 Decoder 的全部分工。

---

### 二、Encoder 详解：一次读完、双向理解

#### （1）输入处理

```Plaintext
token id (B, L)
  → 词嵌入 Embedding        → (B, L, d_model)
  → × sqrt(d_model)         （缩放，配合后面的权重共享与 LN）
  → + 正弦位置编码 PE        → (B, L, d_model)
```

为什么要乘 $\sqrt{d_{model}}$：词嵌入初始化方差是 $1/d_{model}$，而位置编码幅度是 $O(1)$，不缩放的话位置信息会盖住词信息；缩放后两者量级匹配（论文用固定的 $\sqrt{d_{model}}$，可学习缩放也可）。

#### （2）一层内部只做两件事

$$
z' = \text{LN}\big(z + \text{MultiHeadSelfAttn}(z)\big)
$$

$$
z_{\text{out}} = \text{LN}\big(z' + \text{FFN}(z')\big)
$$

- **双向自注意力**：每个位置 $i$ 对所有位置 $j$（含 $j>i$）算权重，输出是"整句的加权摘要"。这是 Encoder 与 Decoder 最大的信息流差异。
- **逐位置 FFN**：$d_{model}\to d_{ff}\to d_{model}$，位置之间不交换信息，只做非线性加工。
- 两个动作都是"残差 + LayerNorm"包装，重复 $N=6$ 层。

#### （3）Encoder 的输出是什么

**不是一个句子向量，而是 $n$ 个向量**：

$$
H = \text{Encoder}(X) \in \mathbb{R}^{n \times d_{model}}
$$

其中第 $i$ 行 $h_i$ 表示"第 $i$ 个 token 在**整句语境**下的表示"。这一点常被误解：

- 做**分类**（如 BERT 的 `[CLS]`）时，人们才把这 $n$ 个向量**池化**成 1 个（取 `[CLS]`、或平均池化）；
- 做**机器翻译**时，$H$ 原封不动地喂给 Decoder——**每个解码步都重新"扫一遍"整个 $H$**，而不是用一个固定向量。这正是 Transformer 相对早期 Seq2Seq 的进步：**没有固定长度瓶颈**。

#### （4）双向注意力矩阵长什么样

$n=4$ 的可见性矩阵（行 = query 位置，列 = key 位置），Encoder 全部可见：

```Plaintext
        k1  k2  k3  k4
 q1     ✓   ✓   ✓   ✓
 q2     ✓   ✓   ✓   ✓
 q3     ✓   ✓   ✓   ✓
 q4     ✓   ✓   ✓   ✓      ← 没有任何位置被屏蔽（除 padding）
```

#### （5）形状追踪

| 步骤 | 形状 |
|---|---|
| token id | $(B, L)$ |
| 词嵌入 × $\sqrt{d_{model}}$ | $(B, L, 512)$ |
| + 位置编码 | $(B, L, 512)$ |
| Q/K/V 投影并拆头 | $3\times(B, 8, L, 64)$ |
| 分数 / softmax | $(B, 8, L, L)$ |
| 输出 $W^O$ | $(B, L, 512)$ |
| **6 层后 memory $H$** | $(B, L, 512)$ ← 交给 Decoder |

#### （6）代码

```python
import math
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """双向编码器：只做「自注意力 + FFN」，没有任何掩码（除 padding）"""

    def __init__(self, vocab, d_model=512, N=6, h=8, d_ff=2048,
                 dropout=0.1, max_len=5000, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, src):
        # 1) 词嵌入缩放 + 位置编码
        x = self.pe(self.embed(src) * self.scale)          # (B, L, d_model)
        # 2) padding mask：屏蔽补齐位置，形状 (B, 1, 1, L)
        mask = (src != self.pad_id)[:, None, None, :]
        # 3) 堆叠 N 层
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)                                 # (B, L, d_model) = memory
```

（`EncoderLayer`、`MultiHeadAttention`、`PositionalEncoding` 的实现见 [[讲解Transformer]] 第十四节。）

#### （7）Encoder 的三条关键性质

1. **上下文化（contextualized）**：同一个词在不同句子里得到不同向量（"苹果手机"与"吃苹果"）；
2. **置换等变**：打乱输入，输出只是被打乱 → 位置信息必须靠位置编码外加；
3. **不能直接用于生成**：双向意味着第 $t$ 个位置已经知道了第 $t+1$ 个词，用来做下一词预测等于作弊（loss 会直接掉到 0，模型学不到东西）。

---

### 三、Decoder 详解：因果约束下的逐步生成

#### （1）一层内部做三件事（编码器只有前两件的后半）

```Plaintext
x ──► ① 掩码多头自注意力 ──► 残差 + LN
   ──► ② 多头交叉注意力（Q=x，K/V=memory）──► 残差 + LN
   ──► ③ 逐位置 FFN        ──► 残差 + LN ──► 下一层
```

| 子层 | Q 来自 | K / V 来自 | 掩码 | 作用 |
|---|---|---|---|---|
| ① 掩码自注意力 | 解码器自身 | 解码器自身 | **因果掩码（+ padding）** | 建模"已生成内容"的内部依赖 |
| ② 交叉注意力 | 解码器自身 | **Encoder 输出 $H$** | 只有 padding | **唯一一处源句信息入口** |
| ③ FFN | — | — | — | 逐位置非线性加工 |

**请注意交叉注意力的 Q/K/V 来源**：Q 是解码器当前状态（"我现在想找什么"），K/V 是编码器输出（"原文里有什么"）。分数矩阵因此是**矩形**：$(B, h, L_{tgt}, L_{src})$，而不是方阵。

#### （2）因果掩码：Decoder 的灵魂

训练时把整句目标句一次性喂进去，靠掩码保证第 $t$ 个位置看不到后面：

$$
M_{ij} = \begin{cases} 0, & j \le i\\ -\infty, & j > i\end{cases}
\qquad
\text{scores} = \frac{QK^\top}{\sqrt{d_k}} + M
$$

$n=4$、对角线及以下可见的因果矩阵（`tril`）：

```Plaintext
        k1  k2  k3  k4
 q1     ✓   ×   ×   ×
 q2     ✓   ✓   ×   ×
 q3     ✓   ✓   ✓   ×
 q4     ✓   ✓   ✓   ✓      ← 每个位置只能看自己和左边
```

对比 Encoder 那张"全 ✓"的矩阵，这就是二者最本质的差别。

**两种掩码必须区分**（实际写代码时最常见的一类 bug）：

| 掩码 | 形状 | 用在哪 | 屏蔽什么 |
|---|---|---|---|
| padding mask | $(B, 1, 1, L_k)$ | Encoder 自注意力、Decoder 交叉注意力、Decoder 自注意力 | 补齐的 `<pad>` |
| causal mask | $(1, 1, L_q, L_k)$ | **仅** Decoder 自注意力 | 未来位置 |

Decoder 自注意力要**两者取与**：

```python
def pad_mask(seq, pad_id=0):                     # (B, L) -> (B, 1, 1, L)
    return (seq != pad_id)[:, None, None, :]

def causal_mask(L, device=None):                 # (1, 1, L, L)
    return torch.tril(torch.ones(L, L, device=device)).bool()[None, None]

# Decoder 第一层子层用：
mask = causal_mask(tgt.size(1), tgt.device) & pad_mask(tgt)   # 广播成 (B,1,L,L)
```

#### （3）训练：一次前向算全部位置

- 目标句右移一位并加 `<bos>`：`[<bos>, y1, y2, ..., y_{m-1}]`
- 标签：`[y1, y2, ..., ym]`
- 因果掩码保证位置 $t$ 的输出只依赖 $\le t$ 的输入，因此**一次前向就能并行算所有位置的交叉熵**

$$
\mathcal{L} = -\sum_{t=1}^{m}\log P_\theta\big(y_t \mid y_{<t}, X\big)
$$

#### （4）推理：自回归 + KV Cache

```Plaintext
输入 <bos>
  ├─ 第 1 步：预测 y1
  ├─ 第 2 步：输入 [<bos>, y1] → 预测 y2
  ├─ 第 3 步：输入 [<bos>, y1, y2] → 预测 y3
  └─ ... 直到 <eos>
```

朴素实现每步都要重算整段序列的 K/V，浪费巨大。**KV Cache** 的思想：已生成位置的 $K,V$ 只算一次并缓存，新 token 只算自己的 $Q$。

| 方案 | 每步计算量 | 显存 |
|---|---|---|
| 无缓存 | $O(n^2 d)$ | $O(1)$ 额外 |
| **有 KV Cache** | $O(n d)$ | $+2\cdot N_{layer}\cdot h\cdot d_k\cdot n\cdot B\cdot \text{字节数}$ |

以 LLaMA-7B 量级粗算（32 层、32 头、$d_k=128$、fp16）：**每 token 约 0.5 MB**，4096 长度单条序列就要约 2 GB —— 所以长上下文推理的瓶颈经常是**显存带宽而非算力**，PagedAttention、MLA 等优化都在针对这里。

注意：**Encoder 没有 KV Cache 问题**，它一次算完就固定不动；Decoder 的交叉注意力中，$K/V$ 来自 Encoder，也只需算一次即可全程复用。

#### （5）形状追踪（$L_{src}=n$ 为源句长，$L_{tgt}=m$ 为目标句长）

| 步骤 | 形状 |
|---|---|
| 目标 token id | $(B, m)$ |
| 词嵌入 + 位置编码 | $(B, m, 512)$ |
| ① 掩码自注意力分数 | $(B, 8, m, m)$ |
| ② 交叉注意力分数 | $(B, 8, m, n)$ ← **矩形** |
| ③ FFN 输出 | $(B, m, 512)$ |
| 输出投影（与词嵌入权重共享） | $(B, m, \text{vocab})$ |

#### （6）代码

```python
class Decoder(nn.Module):
    """因果解码器：掩码自注意力 + 交叉注意力 + FFN"""

    def __init__(self, vocab, d_model=512, N=6, h=8, d_ff=2048,
                 dropout=0.1, max_len=5000, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab, bias=False)
        # 输出投影与词嵌入权重共享（论文做法）
        self.proj.weight = self.embed.weight
        self.scale = math.sqrt(d_model)

    def forward(self, tgt, memory, src_pad=None):
        x = self.pe(self.embed(tgt) * self.scale)           # (B, m, d_model)
        # 交叉注意力只看 padding mask；自注意力要看 因果 & padding
        tgt_mask = causal_mask(tgt.size(1), tgt.device) & pad_mask(tgt, self.pad_id)
        for layer in self.layers:
            x = layer(x, memory, src_pad, tgt_mask)
        return self.proj(self.norm(x))                      # (B, m, vocab)


@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50, pad_id=0):
    """贪心解码：推理时序列串行增长"""
    memory = encoder(src)                                   # 编码一次，反复使用
    src_pad = pad_mask(src, pad_id)
    ys = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
    for _ in range(max_len):
        logits = decoder(ys, memory, src_pad)
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)   # 贪心
        ys = torch.cat([ys, next_tok], dim=1)
        if (next_tok == eos_id).all():
            break
    return ys
```

> 工程细节：真实实现会用 `past_key_values` 缓存 Decoder 自身的 K/V，并让 `ys` 只传最后 1 个 token；上面这版为了可读性每步重算整段（数学等价，只是慢）。把贪心换成「每步保留 top-$k$ 条候选路径」就是 **beam search**（原始论文用 beam=4 + 长度惩罚）。

---

### 四、三处差别的深入解释

#### （1）双向 vs 因果：为什么不能"两头都占"

| 需求 | 双向注意力 | 因果注意力 |
|---|---|---|
| 判断"这句话是正面还是负面" | ✅ 需要看完整句 | ❌ 只看左边会丢一半信息 |
| 生成"下一句是什么" | ❌ 会偷看答案，loss 崩塌 | ✅ 与推理时条件一致 |

核心原则：**训练时的可见性必须与推理时严格一致**。Encoder 不需要"续写"，所以可以双向；Decoder 要逐步续写，所以必须因果。

（这也是 **prefix-LM** 的出发点：把 prompt 当双向的"前缀"，把续写部分当因果的——它说明 Encoder 并非绝对必要，而是"一段双向可见的区域"。）

#### （2）多出来的交叉注意力：唯一的信息入口

- Encoder 与 Decoder 之间**没有任何直接连接**，只有交叉注意力的 K/V；
- 每层都有一次交叉注意力，**浅层看局部词对齐、深层看语义重组**；
- 交叉注意力的 $K,V$ 与解码步无关 → **可以预计算并复用**，这是推理优化的第一个抓手；
- 交叉注意力是**参数增量来源**：$W_Q^{cross},W_K^{cross},W_V^{cross},W_O^{cross}$ 共 $4d_{model}^2=1{,}048{,}576$，使解码器每层从 3.15M 涨到 4.2M（约 +33%）。

#### （3）损失函数：目标的差别决定了架构的选择

| 目标 | 公式直观 | 可见性 | 用在哪 |
|---|---|---|---|
| **CLM**（因果语言建模） | $-\sum_t \log P(y_t\mid y_{<t})$ | 因果 | Decoder-only（GPT） |
| **MLM**（掩码语言建模） | $-\sum_{i\in \mathcal{M}}\log P(x_i\mid x_{\setminus \mathcal{M}})$ | 双向 | Encoder-only（BERT） |
| **Seq2Seq 交叉熵** | $-\sum_t \log P(y_t\mid y_{<t},X)$ | Encoder 双向 + Decoder 因果 | Encoder-Decoder（原始 Transformer、T5） |

#### （4）一个隐性差别：并行度

- Encoder：训练与推理**都**能一次并行处理整段输入；
- Decoder：训练可并行（teacher forcing），**推理只能串行**（自回归）。

这就是"训练快、生成慢"的根本原因，也是后来一切推理加速技术（KV Cache、投机解码、连续批处理）的着力点。

---

### 五、Encoder 与 Decoder 如何协作：一图流

以翻译 "I love NLP" → "我爱自然语言处理" 为例：

```Plaintext
源句 token ──► [词嵌入 + 位置编码]
                     │
                     ▼
        ┌──────────────────────────┐
        │  Encoder × 6             │  双向自注意力 + FFN
        │  （每个位置都看全句）      │
        └──────────────────────────┘
                     │  memory H  (B, n, 512)  ← 只算一次
                     ▼
        ┌──────────────────────────┐
        │  Decoder × 6             │
        │  ① 掩码自注意力（看已生成）│
        │  ② 交叉注意力（Q=自己,    │◄──── H 作为 K/V 输入
        │      K/V=H）              │
        │  ③ FFN                   │
        └──────────────────────────┘
                     │
                     ▼
            线性层（与词嵌入共享权重）
                     │
                     ▼
        每个位置的词表分布 → 自回归地取词
        <bos> → 我 → 爱 → 自然 → 语言 → 处理 → <eos>
```

**与早期 Seq2Seq 的关键区别**：早期模型把整个源句压成一个**固定长度向量** $c$，句子一长就丢信息（信息瓶颈）。Transformer 保留了 $n$ 个向量，让解码器**每一步都能重新查询整句**——这是质量提升的主要来源之一。

**什么时候用这个结构**：输入和输出是**两段不同长度、不同模态或不同语言**的序列——机器翻译、文本摘要、语音识别、代码生成（自然语言→代码）、T5 的 text-to-text 全家桶。

---

### 六、三种架构范式：到底该选哪个

| | **Encoder-only** | **Decoder-only** | **Encoder-Decoder** |
|---|---|---|---|
| 代表 | BERT、RoBERTa、DeBERTa、ELECTRA | GPT 系列、LLaMA、Qwen、Mistral | 原始 Transformer、T5、BART、mBART |
| 可见性 | 全双向 | 全因果 | 编码双向 + 解码因果 |
| 预训练目标 | MLM / 替换检测 | CLM（下一词预测） | Span 去噪 / 序列到序列 |
| 擅长 | 理解、分类、NER、抽取式 QA、句向量、检索 | 生成、对话、代码、通用 few-shot/ICL | 条件生成、翻译、摘要、结构化转换 |
| 训练并行度 | 最高 | 高（训练） | 中（解码侧稍重） |
| 推理成本 | 一次前向，最低 | 自回归，逐 token | 编码一次 + 自回归解码 |
| 主要短板 | 不能直接生成自由文本 | 只能条件于左侧，双向信息靠"预训练语言知识"补 | 结构复杂、参数更多、工程更重 |

#### 为什么现在的大模型几乎都是 Decoder-only？

1. **目标统一**：任何任务都能写成"给定前文，预测下一个 token"，不需要为每种任务改造结构；
2. **训练/推理一致**：训练时怎么算，推理时就怎么用，不存在"训练双向、推理单向"的错位；
3. **可扩展性最好**：结构最简、算子最少，最容易在硬件上堆大规模；
4. **上下文学习**：因果建模 + 大规模数据自然涌现出 few-shot、思维链等能力；
5. **工程友好**：只有一路注意力，KV Cache、连续批处理、投机解码都只针对一种模式优化；
6. **双向信息不丢**：在 decoder-only 里，"双向理解"由"更长的上文 + 预训练知识"近似实现——**当数据与规模足够时，单向约束造成的损失小于结构统一带来的收益**（与 [[Inductive Bias 归纳偏置]] 里"数据够多时弱偏置反超"是同一个道理）。

> 反过来，在**数据量不大、任务明确、要求低延迟**的场景（文本分类、向量检索、信息抽取），Encoder-only 依然是性价比最高的选择——这正是 BERT 系至今活跃在检索与 RAG 中的原因。

---

### 七、常见误区 Q&A

**Q1：Encoder 的输出是一个"句子向量"吗？**
不是。它是 $n\times d$ 的矩阵，每个 token 一个向量。只有当你要做分类/检索时，才额外做池化（`[CLS]`、平均或注意力池化）得到 1 个向量。

**Q2：Decoder 必须有 Encoder 吗？**
不需要。Decoder 的交叉注意力是可选的：把交叉注意力整个去掉，就是 **decoder-only**（GPT）；把 Encoder 保留、Decoder 不加掩码只做交叉注意力，就是某些**prefix-LM / 交叉编码**变体。

**Q3：Decoder 的交叉注意力为什么不用因果掩码？**
它面对的是**完整给定的源句**，源句里没有"未来"概念，全部可见是合理的；这里只需屏蔽 padding。因果约束只针对**正在生成的目标序列**。

**Q4：Encoder 双向注意力会不会造成"信息泄漏"？**
不会。Encoder 的任务是理解，双向正是它的目的。真正要防泄漏的是"**预测第 $t$ 个词时看到它自己及之后**"——那属于 Decoder 自注意力的职责。

**Q5：训练时把整句都喂给 Decoder，不就把答案给了它吗？**
因果掩码让位置 $t$ 看不到 $t$ 之后的内容，因此第 $t$ 步的有效输入只有 $y_{<t}$，与推理时完全一致；被"看到"的右侧 token 在自注意力里权重为 0，不参与计算。

**Q6：Encoder 每层的输出都要存吗？**
不用，只在下一层用；但做**跨层分析**（如 probing、Attention Rollout）或 **U-Net 式跳连**（部分变体）时会保留。

**Q7：Decoder 的每层都必须有交叉注意力吗？**
原始 Transformer 是每层都有。后续也有隔层插入、共享 K/V 投影等变体，用于降低参数与显存。

**Q8：为什么 Decoder 的 FFN 参数不改，整体却重了 1/3？**
因为多了一整套 Q/K/V/O 投影（$4d_{model}^2$）。以 base 为例：编码器每层 3.15M，解码器每层 4.19M。

**Q9：Encoder 和 Decoder 的 LayerNorm 位置一样吗？**
结构上一样，但**深层解码器对 LN 位置更敏感**：原始 Post-LN 需要 warmup 才能稳；现代实现普遍用 **Pre-LN + RMSNorm**，深层解码器才训得动。

**Q10：KV Cache 缓存的是 Encoder 的还是 Decoder 的？**
主要是 **Decoder 自身的 K/V**（逐 token 增长）。交叉注意力里来自 Encoder 的 K/V 是**固定**的，可以算一次全程复用——这也是 Encoder-Decoder 结构在推理上的一个天然优势。

---

### 八、一页速查表

| 项目 | Encoder | Decoder |
|---|---|---|
| 核心组件 | 自注意力 + FFN + 残差/LN | 掩码自注意力 + 交叉注意力 + FFN + 残差/LN |
| 掩码 | 仅 padding | padding + **因果掩码** |
| 注意力矩阵 | 方阵 $(B,h,L,L)$ | 自注意力 $(B,h,m,m)$ + 交叉注意力 $(B,h,m,n)$ |
| 输出 | memory $H\in\mathbb{R}^{B\times n\times d}$ | logits $\in\mathbb{R}^{B\times m\times V}$ |
| 目标函数 | MLM / 去噪 | CLM / Seq2Seq 交叉熵 |
| 训练并行 | 完全并行 | 并行（teacher forcing） |
| 推理 | 一次前向 | 自回归，逐 token；受益于 KV Cache |
| 参数（每层） | ≈ 3.15M | ≈ 4.19M |
| 何时用 | 理解、分类、检索、抽取 | 生成；没有 Encoder 时即为 GPT 式模型 |
| 一句话 | **把整句话读进每个 token 的向量里** | **在只看左边的约束下，把答案一个词一个词写出来** |

**记住三条线**：

1. **可见性线**：双向是理解，因果是生成；
2. **信息线**：Encoder 与 Decoder 之间只靠交叉注意力的 K/V 连接；
3. **工程线**：Encoder 贵在训练一次，Decoder 贵在推理每步。

### 关联笔记

- [[讲解Transformer]]：注意力、多头、位置编码、FFN、完整 PyTorch 实现
- [[Inductive Bias 归纳偏置]]：为什么"数据够多时结构约束可以让位给统一架构"
- [[An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale论文]]：只保留 Encoder 的视觉版本
