---
title: "总结 - ViT与基础模型攻击"
aliases:
  - "索引 - ViT与基础模型攻击"
tags:
  - "知识库/目录总结"
  - "主题/ViT与基础模型攻击"
笔记类型: "目录总结"
分类: "ViT与基础模型攻击"
状态: "已整理"
摘要: "先比较 ViT 的白盒鲁棒性与跨架构迁移，再阅读 Token 攻击以及 PNA/PatchOut 等结构感知方法，最后借助综述扩展到 SAM 和视觉语言模型。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/总结 - 对抗机器学习]]"
相关笔记:
  - "[[30_对抗机器学习/ViT与基础模型攻击/SAM Attack - 分割基础模型攻击发展历程]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/VLM Attack - 视觉语言模型攻击发展历程]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献资料与研究问题]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - Adversarial Token Attacks on Vision Transformers]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - Towards Transferable Adversarial Attacks on Vision Transformers]]"
论文条目:
  - "[[论文来源目录#sam]]"
  - "[[论文来源目录#clip]]"
  - "[[论文来源目录#vit]]"
  - "[[论文来源目录#robustvit]]"
  - "[[论文来源目录#token]]"
  - "[[论文来源目录#pna]]"
  - "[[论文来源目录#patchfool]]"
  - "[[论文来源目录#tgr]]"
  - "[[论文来源目录#mig]]"
  - "[[论文来源目录#vdc]]"
论文标题:
  - "Segment Anything"
  - "Learning Transferable Visual Models From Natural Language Supervision"
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  - "On the Robustness of Vision Transformers to Adversarial Examples"
  - "Adversarial Token Attacks on Vision Transformers"
  - "Towards Transferable Adversarial Attacks on Vision Transformers"
  - "Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?"
  - "Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization"
  - "Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients"
  - "Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection"
发表载体:
  - "ICCV"
  - "ICML"
  - "ICLR"
  - "ICCV"
  - "CVPR Workshop on Transformers for Computer Vision"
  - "AAAI"
  - "ICLR"
  - "CVPR"
  - "ICCV"
  - "AAAI"
发表年份:
  - "2023"
  - "2021"
  - "2021"
  - "2021"
  - "2022"
  - "2022"
  - "2022"
  - "2023"
  - "2023"
  - "2024"
发表类型:
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "研讨会论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html"
  - "https://proceedings.mlr.press/v139/radford21a.html"
  - "https://openreview.net/forum?id=YicbFdNTTy"
  - "https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html"
  - "https://ameya005.github.io/publications/T4WCVPR_AdversarialRobustnessTransformers_final.pdf"
  - "https://ojs.aaai.org/index.php/AAAI/article/view/20169"
  - "https://openreview.net/forum?id=28ib9tf6zhr"
  - "https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html"
  - "https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html"
  - "https://ojs.aaai.org/index.php/AAAI/article/view/28541"
预印本链接:
  - "https://arxiv.org/abs/2010.11929"
  - "https://arxiv.org/abs/2110.04337"
  - "https://arxiv.org/abs/2109.04176"
  - "https://arxiv.org/abs/2203.08392"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/总结 - 对抗机器学习|对抗机器学习总结]]

# 总结 - ViT与基础模型攻击

## 内容总结

先比较 ViT 的白盒鲁棒性与跨架构迁移，再阅读 Token 攻击以及 PNA/PatchOut 等结构感知方法，最后借助综述扩展到 SAM 和视觉语言模型。

- 将 Token 的扰动约束与 PatchOut 的优化机制区分。
- 阅读 ViT 攻击前可回到多头注意力和掩码机制。
- 比较结论时同时查看威胁模型与 Top-1 等指标定义。

## 建议阅读路线

[[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]] → [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程|ViT Attack - 发展历程]] → [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]] → [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献资料与研究问题|ViT Attack - 文献资料与研究问题]] → [[30_对抗机器学习/ViT与基础模型攻击/SAM Attack - 分割基础模型攻击发展历程|SAM Attack - 分割基础模型攻击发展历程]] → [[30_对抗机器学习/ViT与基础模型攻击/VLM Attack - 视觉语言模型攻击发展历程|VLM Attack - 视觉语言模型攻击发展历程]]

## 本次补充的阅读线索

[[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解|Threat Model - 威胁模型详解]] → [[10_基础与算法/评价指标/Top-1 Accuracy - 分类与鲁棒性评价指标|Top-1 Accuracy - 分类与鲁棒性评价指标]] → [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]] → [[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples|论文 - On the Robustness of Vision Transformers to Adversarial Examples]] → [[30_对抗机器学习/ViT与基础模型攻击/论文 - Adversarial Token Attacks on Vision Transformers|论文 - Adversarial Token Attacks on Vision Transformers]] → [[30_对抗机器学习/ViT与基础模型攻击/论文 - Towards Transferable Adversarial Attacks on Vision Transformers|论文 - Towards Transferable Adversarial Attacks on Vision Transformers]] → [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]

## 本目录笔记

- [[30_对抗机器学习/ViT与基础模型攻击/SAM Attack - 分割基础模型攻击发展历程|SAM Attack - 分割基础模型攻击发展历程]]：围绕“SAM Attack - 分割基础模型攻击发展历程”，整理研究背景：为什么 SAM 会成为对抗攻击的目标？、SAM 的工作机制、为什么传统分类攻击不能直接照搬？。
- [[30_对抗机器学习/ViT与基础模型攻击/VLM Attack - 视觉语言模型攻击发展历程|VLM Attack - 视觉语言模型攻击发展历程]]：围绕“VLM Attack - 视觉语言模型攻击发展历程”，整理VLM Attack 的研究背景与起源、什么是视觉语言模型？、年：CLIP 为相关攻击研究提供重要基础。
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程|ViT Attack - 发展历程]]：围绕“ViT Attack - 发展历程”，整理对抗攻击领域中的 ViT Attack 发展历程：从基础攻击到架构感知与迁移性优化、首先明确：什么是 ViT Attack？、什么是 Vision Transformer？。
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]：围绕“ViT Attack - 文献综述与分类”，整理Vision Transformer 对抗攻击（ViT Attack）领域文献综述与研究脉络、领域的奠基性、开创性与关键性工作、ViT Attack 领域的分类框架。
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献资料与研究问题|ViT Attack - 文献资料与研究问题]]：围绕“ViT Attack - 文献资料与研究问题”，整理Vision Transformer 对抗攻击（ViT Attack）领域文献综述与研究脉络、首先明确：这个领域的奠基性工作有哪些？、ViT Attack 领域的完整分类框架。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - Adversarial Token Attacks on Vision Transformers|论文 - Adversarial Token Attacks on Vision Transformers]]：通过 Token 和块稀疏威胁模型，比较 ViT 与 CNN 面对局部扰动时的鲁棒性。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples|论文 - On the Robustness of Vision Transformers to Adversarial Examples]]：比较 ViT、CNN 的白盒鲁棒性、跨模型迁移以及 SAGA 和集成防御的评估设置。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - Towards Transferable Adversarial Attacks on Vision Transformers|论文 - Towards Transferable Adversarial Attacks on Vision Transformers]]：围绕 PNA 与 PatchOut，理解注意力梯度和随机 Patch 更新对 ViT 迁移攻击的作用。

## 主要论文出处

- [Segment Anything](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html)（ICCV，2023，会议论文）
- [Learning Transferable Visual Models From Natural Language Supervision](https://proceedings.mlr.press/v139/radford21a.html)（ICML，2021，会议论文）
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)（ICLR，2021，会议论文）
- [On the Robustness of Vision Transformers to Adversarial Examples](https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html)（ICCV，2021，会议论文）
- [Adversarial Token Attacks on Vision Transformers](https://ameya005.github.io/publications/T4WCVPR_AdversarialRobustnessTransformers_final.pdf)（CVPR Workshop on Transformers for Computer Vision，2022，研讨会论文）
- [Towards Transferable Adversarial Attacks on Vision Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/20169)（AAAI，2022，会议论文）
- [Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?](https://openreview.net/forum?id=28ib9tf6zhr)（ICLR，2022，会议论文）
- [Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html)（CVPR，2023，会议论文）
- [Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html)（ICCV，2023，会议论文）
- [Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection](https://ojs.aaai.org/index.php/AAAI/article/view/28541)（AAAI，2024，会议论文）

## 继续阅读

[[论文来源目录|论文来源目录]] · [[待完善清单|待完善清单]]
