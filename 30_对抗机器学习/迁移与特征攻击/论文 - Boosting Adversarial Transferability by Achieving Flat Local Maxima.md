---
title: "论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima"
aliases:
  - "Boosting Adversarial Transferability by Achieving Flat Local Maxima"
tags:
  - "知识库/论文阅读"
  - "主题/迁移与特征攻击"
笔记类型: "论文阅读"
分类: "迁移与特征攻击"
状态: "已整理"
摘要: "围绕平坦局部极大值和梯度范数惩罚，理解 PGN 如何改善对抗样本迁移性。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击]]"
相关笔记:
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples]]"
  - "[[30_对抗机器学习/攻击方法/MIM - 动量迭代攻击详解]]"
  - "[[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Feature Importance-Aware Transferable Adversarial Attacks]]"
论文条目:
  - "[[论文来源目录#pgn]]"
论文标题:
  - "Boosting Adversarial Transferability by Achieving Flat Local Maxima"
发表载体:
  - "NeurIPS"
发表年份:
  - "2023"
发表类型:
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html"
预印本链接:
  - "https://arxiv.org/abs/2306.05225"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击|迁移与特征攻击总结]]

# 论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima

**作者**：Zhijin Ge, Hongying Liu, Xiaosen Wang, Fanhua Shang, Yuanyuan Liu
**发表**：NeurIPS 2023

---

## 1. 研究背景与核心问题

- **迁移攻击**（Transfer-based attack）：在白盒替代模型（surrogate model）上生成对抗样本，再将其迁移到黑盒目标模型，是现实世界中更具可行性的攻击方式。
- **核心痛点**：在白盒攻击中效果很好的对抗样本，往往**迁移性差**，难以欺骗黑盒模型，尤其是一些经过对抗训练的高级防御模型。
- **已有解释**：之前的工作将迁移性差归因于对抗样本陷入**较差的局部极大值**或**过拟合于替代模型**。
- **本文灵感**：受“**平坦极小值（flat minima）通常带来更好的模型泛化能力**”这一观察启发，作者假设：**位于平坦局部极大值区域的对抗样本，也具有更好的迁移性**。

---

## 2. 核心假设与理论推导

### 2.1 假设 1（Assumption 1）
给定局部区域半径 \(\zeta > 0\)，对于同一输入图像 \(x\) 的两个对抗样本 \(x_1^{adv}\) 和 \(x_2^{adv}\)，若 \(x_1^{adv}\) 邻域内的最大梯度范数小于 \(x_2^{adv}\) 邻域内的最大梯度范数（即 \(x_1^{adv}\) 处于更平坦的区域），则 \(x_1^{adv}\) 在不同模型间具有更好的迁移性。

### 2.2 直观解释
- 平坦区域意味着损失曲面在对抗样本附近**变化平缓**，梯度较小。
- 这种平坦性使得对抗样本**不依赖于替代模型的特定细节**，从而更容易迁移到其他模型。

### 2.3 优化目标
为了找到平坦区域的对抗样本，作者在原始损失函数中加入了一个**梯度范数惩罚项**：
$$
\max_{x^{adv} \in \mathcal{B}_{\epsilon}(x)} \left[ J(x^{adv}, y; \theta) - \lambda \cdot \max_{x' \in \mathcal{B}_{\zeta}(x^{adv})} \| \nabla_{x'} J(x', y; \theta) \|_2 \right]
$$
其中：
- \(J(x^{adv}, y; \theta)\) 是原始分类损失；
- \(\lambda \geq 0\) 是惩罚系数；
- \(\mathcal{B}_{\zeta}(x^{adv})\) 是以 \(x^{adv}\) 为中心、半径为 \(\zeta\) 的邻域；
- 第二项惩罚邻域内的最大梯度范数，促使对抗样本位于平坦区域。

### 2.4 直接优化的困难
- 直接计算邻域内的最大梯度（或二阶 Hessian 矩阵）**计算成本极高**，难以实际应用。
- 因此需要**近似优化方法**。

---

## 3. 核心方法：PGN（Penalizing Gradient Norm）

PGN 通过**有限差分法**近似 Hessian 矩阵，并**随机采样多个点平均梯度**来稳定更新，从而高效地找到平坦局部极大值。

### 3.1 近似优化

**第一步：简化目标函数**
由于邻域内的点 \(x'\) 与 \(x^{adv}\) 很接近，作者近似认为最大化 \(J(x^{adv})\) 和最大化 \(J(x')\) 效果等价。于是目标函数简化为：
$$
\mathcal{L}(x^{adv}, y; \theta) \approx J(x', y; \theta) - \lambda \cdot \| \nabla_{x'} J(x', y; \theta) \|_2, \quad \text{s.t. } x' \in \mathcal{B}_{\zeta}(x^{adv})
$$

**第二步：计算梯度**
对上述目标函数求梯度：
$$
\nabla_{x^{adv}} \mathcal{L} \approx \nabla_{x'} J(x', y; \theta) - \lambda \cdot \nabla_{x'}^2 J(x', y; \theta) \cdot \frac{\nabla_{x'} J(x', y; \theta)}{\| \nabla_{x'} J(x', y; \theta) \|_2}
$$
其中第二项涉及 **Hessian 矩阵与向量的乘积**（Hessian/vector product），计算成本极高。

**第三步：有限差分法近似 Hessian**
利用泰勒展开，Hessian 与向量的乘积可以近似为两个一阶梯度的差：
$$
\nabla_x^2 J(x, y; \theta) v \approx \frac{\nabla_x J(x + \alpha \cdot v, y; \theta) - \nabla_x J(x, y; \theta)}{\alpha}
$$
其中 \(v = -\frac{\nabla_x J(x, y; \theta)}{\| \nabla_x J(x, y; \theta) \|_2}\) 是归一化的梯度方向，\(\alpha\) 是差分步长。

**第四步：得到最终梯度近似（Corollary 1）**
将有限差分代入，得到第 \(t\) 次迭代的梯度近似：
$$
\nabla_{x_t^{adv}} \mathcal{L} \approx (1 - \delta) \cdot \nabla_{x_t'} J(x_t', y; \theta) + \delta \cdot \nabla_{x_t'} J(x_t' + \alpha \cdot v, y; \theta)
$$
其中 \(\delta = \lambda / \alpha\) 是平衡系数。

**直观理解**：最终梯度是**两个相邻点梯度的线性插值**，兼顾了当前点的损失和邻域内的平坦性。

### 3.2 稳定梯度：多样本平均

由于 \(x'\) 是随机采样的，会引入方差。PGN 在每次迭代中**随机采样 \(N\) 个点**，计算它们的梯度并取平均：
$$
\bar{g} = \frac{1}{N} \sum_{i=1}^N \left[ (1 - \delta) \cdot g_i' + \delta \cdot g_i^* \right]
$$
其中 \(g_i' = \nabla_{x_i'} J(x_i', y; \theta)\)，\(g_i^* = \nabla_{x_i^*} J(x_i^*, y; \theta)\)，\(x_i^* = x_i' - \alpha \cdot \frac{g_i'}{\|g_i'\|}\)。
这降低了随机采样的方差，使梯度估计更稳定。

### 3.3 完整算法（Algorithm 1）

**输入**：干净图像 \(x\)，真实标签 \(y\)，损失函数 \(J\)，模型参数 \(\theta\)。
**参数**：扰动幅度 \(\epsilon\)，最大迭代次数 \(T\)，动量衰减因子 \(\mu\)，平衡系数 \(\delta\)，采样半径上界 \(\zeta\)，采样点数 \(N\)。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/迁移与特征攻击/论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples|论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples]]：从跨模型迁移问题进入局部损失几何。
- [[30_对抗机器学习/攻击方法/MIM - 动量迭代攻击详解|MIM - 动量迭代攻击详解]]：比较动量累积与梯度范数惩罚的更新思路。
- [[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法|SGM - 跳跃梯度方法]]：比较局部损失几何与反向传播路径两个改进维度。
- [[30_对抗机器学习/迁移与特征攻击/论文 - Feature Importance-Aware Transferable Adversarial Attacks|论文 - Feature Importance-Aware Transferable Adversarial Attacks]]：比较损失几何优化与重要特征攻击。

## 论文出处

- [Boosting Adversarial Transferability by Achieving Flat Local Maxima](https://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html)：NeurIPS，2023，会议论文。
  [预印本](https://arxiv.org/abs/2306.05225)。

<!-- knowledge-links:end -->
