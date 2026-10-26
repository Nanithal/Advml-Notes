---
title: "MSA - 多头自注意力机制详解"
aliases:
  - "MSA多头自注意力机制详解"
分类: "Transformer与ViT"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[20_深度学习/Transformer与ViT/索引 - Transformer与ViT|Transformer与ViT索引]]

# MSA - 多头自注意力机制详解

# MSA：多头自注意力机制详解

在 **ViT（Vision Transformer）** 中，**MSA** 指的是 **Multi-Head Self-Attention，多头自注意力机制**。

它的核心作用是：**让每个图像块根据当前图像的内容，从其他图像块中收集信息，更新自己的特征。**

本文按照“输入是什么 → 单头怎么算 → 多头怎么组合 → 在 ViT 中如何使用”的顺序解释。

## 1. MSA 处理的是什么？

MSA 接收的是一组向量，也称为一组 **token**。

在 ViT 中，图像先被切成若干个 patch（图像块），每个 patch 再被映射成一个特征向量。

例如，把一张图像分为 4 个 patch，得到：

$$
X=\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}\in\mathbb R^{4\times d}
$$

其中：

- $x_i$：第 $i$ 个 patch 的特征向量，视为一个行向量。
- $d$：每个向量的维度。
- $X$：把所有 token 按行堆叠得到的矩阵。

假设这是一张猫的照片，不同 patch 可能分别包含猫耳朵、眼睛、身体和背景。

**自注意力要做的事，就是让“耳朵区域的特征”能够结合眼睛、身体等区域的信息。**

注意，这里的 token 是特征向量；进入后续层后，它通常已经融合过其他区域的信息，不能始终视为原始 patch 本身。

## 2. 为什么叫“自注意力”？

“自”表示：**用于查询的信息和被查询的信息，都来自同一组输入。**

例如，对于图像中的第一个 token：

> 用第一个 token 去查询当前这组 token，判断应该从各个 token 中吸收多少信息。

它也可以关注自己，并不是只关注其他 token。

理解自注意力，首先要弄清楚三个量：

| 符号 | 名称 | 直观含义 |
|---|---|---|
| $Q$ | Query，查询 | 我需要什么信息？ |
| $K$ | Key，键 | 我能以什么特征被匹配？ |
| $V$ | Value，值 | 我实际提供什么信息？ |

这只是帮助理解的比喻，实际计算中，它们都是数值向量。

## 3. Q、K、V 从哪里来？

设输入为 $X\in\mathbb R^{n\times d}$，其中 $n$ 是 token 数量。

对同一个输入 $X$，分别进行三次线性变换：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其中，$W_Q,W_K,W_V$ 是通过训练学习到的参数矩阵。为便于说明，本文公式省略可选的偏置项。

先考虑一个注意力头，假设：

$$
W_Q,W_K\in\mathbb R^{d\times d_k},\qquad W_V\in\mathbb R^{d\times d_v}
$$

那么：

$$
Q,K\in\mathbb R^{n\times d_k},\qquad V\in\mathbb R^{n\times d_v}
$$

对于第 $i$ 个 token：

$$
q_i=x_iW_Q,\qquad k_i=x_iW_K,\qquad v_i=x_iW_V
$$

**同一个 token 会产生三个不同的表示，分别承担查询、匹配和提供内容的作用。**

因此，“自注意力”不表示 $Q=K=V$，而是表示它们的**输入来源相同**。

## 4. 单头自注意力具体怎么算？

完整公式是：

$$
\boxed{\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$

这是 Transformer 使用的缩放点积注意力（Scaled Dot-Product Attention）。下面把公式拆开。

### 4.1 计算 token 之间的匹配分数

第 $i$ 个 token 对第 $j$ 个 token 的分数为：

$$
s_{ij}=\frac{q_i k_j^\top}{\sqrt{d_k}}
$$

这里的点积衡量的是：在学习得到的查询、键空间中，二者有多匹配。它不是直接比较原始图像像素。

一次性计算所有 token 对，就得到：

$$
S=\frac{QK^\top}{\sqrt{d_k}}\in\mathbb R^{n\times n}
$$

例如，4 个 token 对应：

$$
S=\begin{bmatrix}
s_{11}&s_{12}&s_{13}&s_{14}\\
s_{21}&s_{22}&s_{23}&s_{24}\\
s_{31}&s_{32}&s_{33}&s_{34}\\
s_{41}&s_{42}&s_{43}&s_{44}
\end{bmatrix}
$$

**第 $i$ 行表示：第 $i$ 个 token 对所有 token 的匹配分数。**

### 4.2 把分数转换为注意力权重

对 $S$ 的**每一行**分别做 softmax：

$$
a_{ij}=\frac{\exp(s_{ij})}{\sum_{t=1}^{n}\exp(s_{it})}
$$

得到注意力矩阵：

$$
A=\operatorname{softmax}(S)\in\mathbb R^{n\times n}
$$

每一行的权重之和为 1：

$$
\sum_{j=1}^{n}a_{ij}=1
$$

例如，第一个 token 的注意力权重可能是：

$$
A_{1,:}=[0.1,\;0.6,\;0.2,\;0.1]
$$

表示它在这次信息聚合中，分配给第二个 token 的权重最大。

注意，一般来说：

$$
a_{ij}\ne a_{ji}
$$

“耳朵关注眼睛的程度”，不必等于“眼睛关注耳朵的程度”。不同的查询向量和键向量会形成有方向性的匹配。

### 4.3 按照权重，对 Value 加权求和

第 $i$ 个 token 的输出为：

$$
z_i=\sum_{j=1}^{n}a_{ij}v_j
$$

所以刚才第一个 token 的输出就是：

$$
z_1=0.1v_1+0.6v_2+0.2v_3+0.1v_4
$$

把所有 token 的输出放在一起：

$$
Z=AV\in\mathbb R^{n\times d_v}
$$

**Q 和 K 决定“从哪里取、取多少”，V 决定“实际取到什么”。**

特别注意：**注意力矩阵 $A$ 是权重，真正的单头输出是 $AV$。**

## 5. 用一个数值例子理解信息聚合

下面的数值是为了演示计算而人为设定的，并非论文实验结果。

假设有 3 个 token，第一个 token 对它们的注意力权重为：

$$
[0.2,\;0.5,\;0.3]
$$

对应的 Value 为：

$$
v_1=[1,0],\qquad v_2=[0,2],\qquad v_3=[3,1]
$$

那么：

$$
\begin{aligned}
z_1
&=0.2[1,0]+0.5[0,2]+0.3[3,1]\\
&=[0.2,0]+[0,1]+[0.9,0.3]\\
&=\boxed{[1.1,\;1.3]}
\end{aligned}
$$

第一个 token 的新表示因此融合了三个 token 的信息。

**自注意力不会只挑出一个 token，而是按照权重混合多个 token 的信息。**

这里的权重针对第一个 token；第二、第三个 token 会使用各自的权重，得到各自的输出。

## 6. 为什么要除以平方根？

点积需要累加 $d_k$ 个乘积：

$$
q_i k_j^\top=\sum_{r=1}^{d_k}q_{ir}k_{jr}
$$

为了分析其尺度，假设各分量相互独立，均值为 0、方差为 1。于是，每个乘积满足：

$$
\mathbb E[q_{ir}k_{jr}]=0,\qquad
\operatorname{Var}(q_{ir}k_{jr})=1
$$

独立项求和后：

$$
\operatorname{Var}(q_i k_j^\top)=d_k
$$

因此，点积的标准差是 $\sqrt{d_k}$。维度增大时，分数的典型幅度也会增大。

过大的分数差异容易让 softmax 输出接近：

$$
[0,\;1,\;0,\ldots]
$$

这会使 softmax 的部分梯度变得很小。因为对于 $a_j=\operatorname{softmax}(s)_j$，其导数为：

$$
\frac{\partial a_j}{\partial s_t}=a_j(\delta_{jt}-a_t)
$$

其中 $\delta_{jt}$ 在 $j=t$ 时为 1，否则为 0。当输出非常接近 one-hot 分布时，上述导数趋近于 0。

除以 $\sqrt{d_k}$ 后，在上述假设下：

$$
\operatorname{Var}\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right)=1
$$

这样可以控制分数尺度，帮助稳定训练。这里的独立、单位方差是用于解释缩放因子的分析假设，不是训练过程中必须严格成立的条件。

## 7. “多头”是什么意思？

前面只有一组 $Q,K,V$，对应一个注意力头。

多头注意力会使用多组不同的投影参数：

$$
Q^{(r)}=XW_Q^{(r)},\qquad
K^{(r)}=XW_K^{(r)},\qquad
V^{(r)}=XW_V^{(r)}
$$

其中，$r=1,\ldots,h$，$h$ 是头数。

每个头独立计算：

$$
Z^{(r)}=\operatorname{softmax}\left(\frac{Q^{(r)}(K^{(r)})^\top}{\sqrt{d_k}}\right)V^{(r)}
$$

因此，不同头可以产生不同的注意力矩阵，从不同表示子空间聚合信息。

可以直观地想象，有些头更关注邻近区域，有些头更关注较远区域；但**不能把每个头固定理解为“颜色头”“形状头”或“边缘头”**，其实际行为由训练决定。

### 7.1 多头划分的是特征维度

**多头通常沿特征维度划分表示，每个头仍然接收全部 token。**

例如，输入是：

$$
X\in\mathbb R^{197\times768}
$$

使用 12 个头，并采用常见的等宽配置，每个头维度为：

$$
d_k=d_v=\frac{768}{12}=64
$$

于是，每个头都有：

$$
Q^{(r)},K^{(r)},V^{(r)}\in\mathbb R^{197\times64}
$$

每个头的注意力矩阵都是：

$$
A^{(r)}\in\mathbb R^{197\times197}
$$

所以 12 个头都能处理全部 197 个 token，而不是每个头只处理其中一部分 token。

### 7.2 实现时为什么经常先投影再拆分？

工程实现中，常常先用一个线性层一次性生成所有头的 Q、K、V，然后 reshape、transpose 成多头形式。

例如，包含 batch 维时，可以把每种投影后的表示从：

$$
(B,n,d)
$$

整理为：

$$
(B,h,n,d/h)
$$

其中 $B$ 是 batch size。

这与“每个头使用自己的一组投影矩阵”在数学上等价：大投影矩阵的不同列块，就是不同头的投影参数。拆分的是**投影后的特征维度**。

## 8. 多个头的结果怎么合并？

先沿特征维度拼接：

$$
Z_{\mathrm{cat}}=\operatorname{Concat}\left(Z^{(1)},\ldots,Z^{(h)}\right)
$$

再做一次输出线性变换：

$$
\boxed{\operatorname{MSA}(X)=\operatorname{Concat}\left(Z^{(1)},\ldots,Z^{(h)}\right)W_O}
$$

$W_O$ 用于混合不同头产生的特征。一般有：

$$
W_O\in\mathbb R^{(h d_v)\times d}
$$

沿用上面的例子，整个维度变化是：

| 阶段 | 形状，不含 batch 维 |
|---|---|
| 输入 $X$ | $197\times768$ |
| 每个头的 $Q,K,V$ | $197\times64$ |
| 每个头的注意力矩阵 | $197\times197$ |
| 每个头的输出 | $197\times64$ |
| 12 个头拼接 | $197\times768$ |
| 输出投影后 | $197\times768$ |

**token 数量保持不变，每个 token 的特征被更新。**

## 9. MSA 在 ViT 中放在哪里？

原始 ViT 的一个 Transformer Block 可以写成：

$$
X'=X+\operatorname{MSA}(\operatorname{LN}(X))
$$

$$
Y=X'+\operatorname{MLP}(\operatorname{LN}(X'))
$$

其中：

- **LN**：对特征做层归一化。
- **MSA**：让不同 token 交换信息。
- **MLP**：对每个 token 的特征分别进行非线性变换，不在 token 之间直接混合信息。
- **残差连接 $+$**：将原表示与子模块输出相加。

这里 LN 位于子模块之前，因此称为 Pre-LN 结构。MSA 本身与包含 LN、残差连接、MLP 的完整 Transformer Block 需要区分。

### 9.1 为什么例子里有 197 个 token？

以 $224\times224$ 图像、$16\times16$ patch 为例，patch 数量是：

$$
\frac{224}{16}\times\frac{224}{16}=196
$$

加上一个用于分类的 $[\mathrm{CLS}]$ token，共有 197 个 token。

输入序列可以写成：

$$
X_0=[x_{\mathrm{CLS}};\,e_1;\,\ldots;\,e_{196}]+E_{\mathrm{pos}}
$$

其中 $e_i$ 是 patch embedding，$E_{\mathrm{pos}}$ 是位置嵌入。

$[\mathrm{CLS}]$ 也参与自注意力，并逐层聚合图像信息。原始 ViT 使用最后一层对应的分类 token 表示完成图像分类。

## 10. 阅读 ViT Attack 论文时，重点区分什么？

可以特别关注这三个位置：

$$
\underbrace{QK^\top}_{\text{匹配分数的未缩放形式}}
\quad\longrightarrow\quad
\underbrace{A}_{\text{注意力权重}}
\quad\longrightarrow\quad
\underbrace{AV}_{\text{聚合后的特征}}
$$

| 对象 | 表示什么 | 阅读方法时需要检查的问题 |
|---|---|---|
| $Q,K$ | 查询和匹配表示 | 方法是否改变了匹配关系？ |
| $QK^\top/\sqrt{d_k}$ | softmax 之前的分数 | 操作作用于分数，还是归一化后的权重？ |
| $A$ | token 之间的信息聚合权重 | 操作改变了哪些 token 之间的权重？ |
| $V$ | 实际被聚合的内容 | 方法是否改变了所传递的特征？ |
| $AV$ | 单个头聚合后的输出 | 方法影响的是输出特征，还是注意力分布？ |

这些是理解方法时的分析位置，不代表每篇攻击论文都会直接修改模型内部变量。若攻击只修改输入图像，内部的 Q、K、V、注意力权重和输出也可能随之变化；具体机制需要结合论文定义和实验判断。

## 11. 常见混淆

| 容易混淆的说法 | 准确理解 |
|---|---|
| 自注意力意味着 $Q=K=V$ | 三者来自同一输入，但使用不同投影，通常不相等。 |
| 注意力矩阵就是 MSA 输出 | 注意力矩阵是 $A$；单头输出是 $AV$，多头还需要拼接与输出投影。 |
| 每个头只处理一部分 patch | 标准全局 MSA 中，每个头都处理全部 token。 |
| 多头就是把同一计算重复几遍 | 不同头使用不同投影参数，可以形成不同的匹配关系。 |
| MSA 和 MHA 完全是两个模块 | MHA 是多头注意力的通称；MSA 强调其为自注意力，即 Q、K、V 来自同一序列。 |
| 注意力越大就一定说明越重要 | 权重表示本次聚合的系数；对最终预测的影响还取决于 Value、输出投影、残差和后续计算。 |

## 12. 核心公式回顾

对每个头 $r$：

$$
Q^{(r)}=XW_Q^{(r)},\qquad
K^{(r)}=XW_K^{(r)},\qquad
V^{(r)}=XW_V^{(r)}
$$

$$
A^{(r)}=\operatorname{softmax}\left(\frac{Q^{(r)}(K^{(r)})^\top}{\sqrt{d_k}}\right)
$$

$$
Z^{(r)}=A^{(r)}V^{(r)}
$$

最后：

$$
\operatorname{MSA}(X)=\operatorname{Concat}(Z^{(1)},\ldots,Z^{(h)})W_O
$$

**MSA 的计算过程：生成 Q、K、V → 计算匹配分数 → softmax 得到权重 → 加权聚合 V → 拼接多头结果 → 输出投影。**

## 参考资料

1. Vaswani et al. **Attention Is All You Need**，2017。重点参阅 §3.2.1（缩放点积注意力）与 §3.2.2（多头注意力）。[论文页面](https://arxiv.org/abs/1706.03762)
2. Dosovitskiy et al. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**。重点参阅 §3.1（ViT 架构）与 Appendix A（多头自注意力）。[论文页面](https://arxiv.org/abs/2010.11929)

文中的猫图像类比、数值演示和部分维度推演用于辅助理解，不是论文实验结论。


---

## 笔记联系

- [[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]]：将多头自注意力放回 Transformer 全局结构。
- [[20_深度学习/Transformer与ViT/Transformer - 原理与代码|Transformer - 原理与代码]]：把注意力公式对应到张量形状与代码实现。
- [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]：对照自注意力如何作用于图像 patch 序列。
