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
摘要: "由入门解释进入多头自注意力、代码实现和 Encoder/Decoder 分工，再阅读 ViT 原论文。模型本身的机制为理解 Token、Patch、注意力和梯度攻击提供基础。"
更新日期: "2026-10-30"
上级目录: "[[20_深度学习/总结 - 深度学习]]"
相关笔记:
  - "[[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]"
  - "[[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - 入门详解]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - 原理与代码]]"
  - "[[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解]]"
论文条目:
  - "[[论文来源目录#vit]]"
  - "[[论文来源目录#transformer]]"
论文标题:
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  - "Attention Is All You Need"
发表载体:
  - "ICLR"
  - "NIPS"
发表年份:
  - "2021"
  - "2017"
发表类型:
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://openreview.net/forum?id=YicbFdNTTy"
  - "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html"
预印本链接:
  - "https://arxiv.org/abs/2010.11929"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[20_深度学习/总结 - 深度学习|深度学习总结]]

# 总结 - Transformer与ViT

## 内容总结

由入门解释进入多头自注意力、代码实现和 Encoder/Decoder 分工，再阅读 ViT 原论文。模型本身的机制为理解 Token、Patch、注意力和梯度攻击提供基础。

- 从 Q、K、V 与张量形状理解注意力计算。
- 把 ViT 的 Patch 序列与 Transformer 编码器对应起来。
- 掩码机制与攻击中的梯度掩蔽需要区分。

## 建议阅读路线

[[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]] → [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]] → [[20_深度学习/Transformer与ViT/Transformer - 原理与代码|Transformer - 原理与代码]] → [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]] → [[20_深度学习/训练基础/Inductive Bias - 归纳偏置|Inductive Bias - 归纳偏置]] → [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]] → [[50_汇报与写作/ViT/ViT - 原论文汇报提纲|ViT - 原论文汇报提纲]]

## 本目录笔记

- [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]]：围绕“MSA - 多头自注意力机制详解”，整理MSA：多头自注意力机制详解、MSA 处理的是什么？、为什么叫“自注意力”？。
- [[20_深度学习/Transformer与ViT/Transformer - Encoder与Decoder详解|Transformer - Encoder与Decoder详解]]：围绕“Transformer - Encoder与Decoder详解”，整理为什么需要"编码器 + 解码器"这两个角色、Encoder 详解：一次读完、双向理解、输入处理。
- [[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]]：围绕“Transformer - 入门详解”，整理Transformer 究竟是什么？、先理解 Transformer 解决了什么问题、Transformer 的定义。
- [[20_深度学习/Transformer与ViT/Transformer - 原理与代码|Transformer - 原理与代码]]：围绕“Transformer - 原理与代码”，整理句话定义与全局地图、为什么要有 Transformer：RNN 与 CNN 的瓶颈、RNN 的三个死穴。
- [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]：围绕“论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale”，整理研究背景与核心问题、方法：Vision Transformer（ViT）、整体流程（论文图 1）。

## 主要论文出处

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)（ICLR，2021，会议论文）
- [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)（NIPS，2017，会议论文）

## 继续阅读

[[论文来源目录|论文来源目录]] · [[待完善清单|待完善清单]]
