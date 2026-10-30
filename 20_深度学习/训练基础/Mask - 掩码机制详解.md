---
title: "Mask - 掩码机制详解"
aliases:
  - "机器学习中的掩码-Mask详解"
tags:
  - "知识库/笔记"
  - "主题/训练基础"
笔记类型: "概念笔记"
分类: "训练基础"
状态: "已整理"
摘要: "比较 Padding、Causal、Random、Feature 和 Loss Mask，并区分掩码与 Dropout。"
更新日期: "2026-10-30"
上级目录: "[[20_深度学习/训练基础/总结 - 训练基础]]"
相关笔记:
  - "[[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - Towards Transferable Adversarial Attacks on Vision Transformers]]"
  - "[[30_对抗机器学习/防御与鲁棒性/Gradient Masking - 梯度掩蔽详解]]"
论文条目:
  - "[[论文来源目录#transformer]]"
  - "[[论文来源目录#bert]]"
  - "[[论文来源目录#mae]]"
论文标题:
  - "Attention Is All You Need"
  - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
  - "Masked Autoencoders Are Scalable Vision Learners"
发表载体:
  - "NIPS"
  - "NAACL-HLT"
  - "CVPR"
发表年份:
  - "2017"
  - "2019"
  - "2022"
发表类型:
  - "会议论文"
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
  - "https://aclanthology.org/N19-1423/"
  - "https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[20_深度学习/训练基础/总结 - 训练基础|训练基础总结]]

# Mask - 掩码机制详解

## 一、什么是 Mask？

**掩码（Mask）**是一种控制数据或计算参与范围的机制。它可以用于隐藏输入信息、限制注意力连接、选择有效标签，或者进行随机特征遮挡。

Mask 并不总是简单的 0/1 乘法：不同任务中，掩码的含义和作用位置不同。

### 1. 基础示例

假设输入为：

$$
X=[2,5,8,3,7]
$$

构造掩码：

$$
M=[1,0,1,0,1]
$$

逐元素相乘：

$$
X'=X\odot M=[2,0,8,0,7]
$$

其中 $\odot$ 是 Hadamard 积（逐元素乘法）。

**注意：**将输入置零不意味着该位置在所有后续计算中都被彻底忽略。例如，带偏置的线性层仍可能将零输入映射为非零输出。

## 二、常见 Mask 类型

| 类型 | 作用 | 典型应用 |
|---|---|---|
| Padding Mask | 忽略填充位置 | NLP、Transformer |
| Causal Mask | 阻止访问未来位置 | GPT、自回归模型 |
| Attention Mask | 控制注意力连接 | Transformer、ViT |
| Random Mask | 随机隐藏输入内容 | BERT、MAE |
| Feature Mask | 选择或遮挡特征 | 特征选择、鲁棒学习 |
| Loss Mask | 选择参与损失计算的位置 | 序列训练、部分标注 |

Padding Mask 和 Causal Mask 都是 Attention Mask 的常见形式。

## 三、Padding Mask：填充掩码

在批量处理不同长度的句子时，需要将短句填充到相同长度：

```text
句子 A：I love deep learning
句子 B：I love AI [PAD]
```

有效位置掩码可以写为：

$$
M_A=[1,1,1,1],\qquad M_B=[1,1,1,0]
$$

在注意力计算中，可以将填充 Key 对应的分数设为 $-\infty$，使其在 Softmax 后获得零权重。实际实现中还需根据模型结构处理填充 Query 和后续损失计算。

## 四、Causal Mask：因果掩码

自回归语言模型预测当前 Token 时，不应访问未来 Token。

对于长度为 4 的序列，加性 Causal Mask 为：

$$
M=\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

行对应 Query，列对应 Key。第 $i$ 个 Query 只能访问自身及之前的位置。

### 1. 在 Attention 中如何使用？

标准缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

加入加性 Mask 后：

$$
\boxed{\operatorname{Attention}(Q,K,V,M)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}+M\right)V}
$$

掩码在 **Softmax 之前**作用。

例如原始分数：

$$S=[2,1,3]$$

对应 Softmax 约为：

$$\operatorname{softmax}(S)=[0.245,0.090,0.665]$$

屏蔽第三个位置：

$$S+M=[2,1,-\infty]$$

则：

$$\operatorname{softmax}(S+M)=[0.731,0.269,0]$$

第三个位置不再分配到注意力权重。

## 五、Random Mask：随机掩码

### 1. BERT 的 Masked Language Modeling（MLM）

原句：

```text
I love machine learning
```

遮挡后：

```text
I love [MASK] learning
```

模型根据上下文预测被选中的词：

$$P(\text{machine}\mid\text{I love [MASK] learning})$$

BERT 原始 MLM 方案选择约 15% 的 Token 作为预测目标；选中位置并非全部替换为 `[MASK]`，还包含随机替换和保持原词不变的情况。

### 2. ViT 中的 MAE（Masked Autoencoder）

MAE 将图像切分成 Patch，随机遮挡较大比例（原论文典型设置为 75%），Encoder 只处理可见 Patch，Decoder 尝试重建被遮挡内容。

若图像分成 $14\times14=196$ 个 Patch：

$$X=\{x_1,x_2,\ldots,x_{196}\}$$

设可见索引集合为 $\mathcal V$，被遮挡索引集合为 $\mathcal M$，Encoder 接收 $X_{\mathcal V}$。

重建损失可写为：

$$
\boxed{L=\frac{1}{|\mathcal M|}\sum_{i\in\mathcal M}\|x_i-\hat x_i\|_2^2}
$$

其中 $x_i$ 为原始 Patch，$\hat x_i$ 为重建结果。实际实现可能对像素进行归一化。

**意义：**模型需要利用可见区域学习图像结构与语义，而不是简单复制全部输入。

## 六、Feature Mask：特征掩码

给定：

$$x=[x_1,x_2,x_3,x_4],\quad M=[1,0,1,0]$$

则：

$$x'=x\odot M=[x_1,0,x_3,0]$$

常见用途包括：

- **特征选择：**保留特定特征进行训练或推理。
- **数据增强：**随机遮挡部分输入特征。
- **鲁棒性评估：**测试模型对信息缺失的敏感性。
- **可解释性：**遮挡局部区域并观察预测变化。

Mask 本身不等于对抗攻击，但可以作为攻击、鲁棒性分析或防御方法中的操作。

## 七、Loss Mask：损失掩码

假设每个位置的损失为：

$$\ell=[0.2,0.8,0.3,0.9]$$

有效标签掩码为：

$$M=[1,0,1,0]$$

只在有效位置计算平均损失：

$$
L=\frac{\sum_i M_i\ell_i}{\sum_iM_i}=\frac{0.2+0.3}{2}=0.25
$$

在实际代码中，应处理有效位置数为零的情况。

## 八、PyTorch 实现

### 1. 普通元素掩码

```python
import torch

x = torch.tensor([2., 5., 8., 3., 7.])
mask = torch.tensor([1., 0., 1., 0., 1.])
result = x * mask
print(result)
# tensor([2., 0., 8., 0., 7.])
```

### 2. Causal Attention Mask

```python
import torch
import torch.nn.functional as F

scores = torch.tensor([
    [2., 1., 3.],
    [1., 4., 2.],
    [3., 2., 1.]
])

# 在 masked_fill 中，True 表示屏蔽该位置
mask = torch.triu(
    torch.ones(3, 3, dtype=torch.bool),
    diagonal=1
)

masked_scores = scores.masked_fill(mask, float('-inf'))
attention = F.softmax(masked_scores, dim=-1)
print(attention)
```

**API 注意：**不同接口的布尔掩码语义可能相反。比如 `torch.nn.functional.scaled_dot_product_attention` 的布尔 `attn_mask` 中，`True` 表示允许参与注意力；而上例 `masked_fill` 的 `True` 表示需要屏蔽。

### 3. 随机特征掩码

```python
import torch

x = torch.randn(4, 8)
mask_ratio = 0.5
mask = torch.rand_like(x) > mask_ratio
masked_x = x * mask
print(masked_x)
```

这段代码独立地以约 50% 的概率保留各元素。

## 九、Mask 和 Dropout 的区别

Dropout 训练时也会使用随机 Mask。以 inverted dropout 为例：

$$h'=\frac{h\odot M}{1-p}$$

其中 $M_i\sim\operatorname{Bernoulli}(1-p)$，$p$ 为丢弃概率。

| 对比 | Mask | Dropout |
|---|---|---|
| 概念 | 信息选择、遮挡或计算约束机制 | 正则化技术 |
| 是否随机 | 不一定 | 训练时通常随机 |
| 主要目的 | 控制可见数据、连接或损失 | 缓解过拟合 |
| 推理时 | 可能继续使用 | 标准 Dropout 通常关闭 |

因此，**Dropout 会使用 Mask，但 Mask 的用途远不止 Dropout。**

## 十、总结

可以从三个层次理解 Mask：

1. **输入层面：**控制模型能看到什么，例如 BERT MLM、MAE、Feature Mask。
2. **计算层面：**控制信息如何交互，例如 Padding Mask、Causal Mask、Attention Mask。
3. **目标层面：**控制哪些位置参与训练损失，例如 Loss Mask。

最关键的区别是：

- **元素 Mask：**常见操作是 $X\odot M$。
- **Attention Mask：**常见操作是 Softmax 前添加 $0/-\infty$ 掩码。
- **自监督 Mask：**隐藏部分输入，让模型预测或重建缺失信息。
- **Loss Mask：**选择哪些位置产生训练损失。

对于 Transformer、ViT 和对抗鲁棒性研究，建议优先掌握 **Causal Mask、Padding Mask、MAE 的 Random Mask**，并注意不同 PyTorch API 的掩码语义。

<!-- knowledge-links:start -->

## 笔记联系

- [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]]：把掩码作用位置对应到注意力权重计算。
- [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]]：对应 Encoder 与 Decoder 的可见范围。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - Towards Transferable Adversarial Attacks on Vision Transformers|论文 - Towards Transferable Adversarial Attacks on Vision Transformers]]：比较模型中的计算掩码与 PatchOut 中的随机选择。
- [[30_对抗机器学习/防御与鲁棒性/Gradient Masking - 梯度掩蔽详解|Gradient Masking - 梯度掩蔽详解]]：区分计算掩码与梯度掩蔽这两个概念。

## 论文出处

- [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)：NIPS，2017，会议论文。
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/)：NAACL-HLT，2019，会议论文。
- [Masked Autoencoders Are Scalable Vision Learners](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)：CVPR，2022，会议论文。

<!-- knowledge-links:end -->
