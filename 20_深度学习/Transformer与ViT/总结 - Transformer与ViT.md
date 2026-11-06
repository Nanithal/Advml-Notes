---
title: "总结 - Transformer与ViT"
aliases:
  - "索引 - Transformer与ViT"
tags:
  - "知识库/目录总结"
  - "主题/Transformer与ViT"
笔记类型: "目录总结"
分类: "Transformer与ViT"
状态: "已整理"
摘要: "由注意力、代码与 Encoder/Decoder 分工进入 ViT，再对照 CNN 理解局部与全局建模、归纳偏置和表示迁移，并区分对抗迁移。"
更新日期: "2026-11-06"
上级目录: "[[20_深度学习/总结 - 深度学习]]"
相关笔记:
  - "[[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]"
  - "[[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - 入门详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - 原理与代码]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解]]"
  - "[[20_深度学习/Transformer与ViT/ViT与CNN - 局部特征、全局关系与迁移性]]"
论文条目:
  - "[[论文来源目录#vit]]"
  - "[[论文来源目录#transformer]]"
  - "[[论文来源目录#vit_cnn]]"
  - "[[论文来源目录#intriguing_vit]]"
  - "[[论文来源目录#feature_transfer]]"
论文标题:
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  - "Attention Is All You Need"
  - "Do Vision Transformers See Like Convolutional Neural Networks?"
  - "Intriguing Properties of Vision Transformers"
  - "How transferable are features in deep neural networks?"
发表载体:
  - "ICLR"
  - "NIPS"
  - "NeurIPS"
  - "NeurIPS"
  - "NIPS"
发表年份:
  - "2021"
  - "2017"
  - "2021"
  - "2021"
  - "2014"
发表类型:
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://openreview.net/forum?id=YicbFdNTTy"
  - "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
  - "https://proceedings.neurips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html"
  - "https://proceedings.neurips.cc/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html"
  - "https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html"
预印本链接:
  - "https://arxiv.org/abs/2010.11929"
  - "https://arxiv.org/abs/2108.08810"
  - "https://arxiv.org/abs/2105.10497"
  - "https://arxiv.org/abs/1411.1792"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[20_深度学习/总结 - 深度学习|深度学习总结]]

# 总结 - Transformer与ViT

## 内容总结

由入门解释进入多头自注意力、代码实现和 Encoder/Decoder 分工，再阅读 ViT 原论文。模型本身的机制为理解 Token、Patch、注意力和梯度攻击提供基础。

- 从 Q、K、V 与张量形状理解注意力计算。
- 把 ViT 的 Patch 序列与 Transformer 编码器对应起来。
- 掩码机制与攻击中的梯度掩蔽需要区分。
- 对照 ViT 与 CNN 时，区分低层特征、局部纹理依赖、下游表示迁移与对抗扰动迁移。

## 建议阅读路线

[[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]] → [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]] → [[20_深度学习/Transformer与ViT/Transformer - 原理与代码|Transformer - 原理与代码]] → [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]] → [[20_深度学习/训练基础/Inductive Bias - 归纳偏置|Inductive Bias - 归纳偏置]] → [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]] → [[50_汇报与写作/ViT/ViT - 原论文汇报提纲|ViT - 原论文汇报提纲]]

[[20_深度学习/训练基础/Inductive Bias - 归纳偏置|归纳偏置]] → [[20_深度学习/Transformer与ViT/ViT与CNN - 局部特征、全局关系与迁移性|ViT与CNN - 局部特征、全局关系与迁移性]]：从架构与预训练共同理解特征迁移，避免把“低层特征更少”当成普遍优势。

## 本目录笔记

- [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]]：围绕“MSA - 多头自注意力机制详解”，整理MSA：多头自注意力机制详解、MSA 处理的是什么？、为什么叫“自注意力”？。
- [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]]：围绕“Transformer - Encoder与Decoder详解”，整理为什么需要"编码器 + 解码器"这两个角色、Encoder 详解：一次读完、双向理解、输入处理。
- [[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]]：围绕“Transformer - 入门详解”，整理Transformer 究竟是什么？、先理解 Transformer 解决了什么问题、Transformer 的定义。
- [[20_深度学习/Transformer与ViT/Transformer - 原理与代码|Transformer - 原理与代码]]：围绕“Transformer - 原理与代码”，整理句话定义与全局地图、为什么要有 Transformer：RNN 与 CNN 的瓶颈、RNN 的三个死穴。
- [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]：围绕“论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale”，整理研究背景与核心问题、方法：Vision Transformer（ViT）、整体流程（论文图 1）。

- [[20_深度学习/Transformer与ViT/ViT与CNN - 局部特征、全局关系与迁移性|ViT与CNN - 局部特征、全局关系与迁移性]]：解释局部卷积、全局交互、纹理与形状倾向，指出低层特征也可通用，以及表示迁移不等于攻击迁移。

## 主要论文出处

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)（ICLR，2021，会议论文）
- [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)（NIPS，2017，会议论文）

- [Do Vision Transformers See Like Convolutional Neural Networks?](https://proceedings.neurips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html)（NeurIPS，2021，会议论文）
- [Intriguing Properties of Vision Transformers](https://proceedings.neurips.cc/paper/2021/hash/c404a5adbf90e09631678b13b05d9d7a-Abstract.html)（NeurIPS，2021，会议论文）
- [How transferable are features in deep neural networks?](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html)（NIPS，2014，会议论文）

## 继续阅读

[[论文来源目录|论文来源目录]] · [[待完善清单|待完善清单]]
