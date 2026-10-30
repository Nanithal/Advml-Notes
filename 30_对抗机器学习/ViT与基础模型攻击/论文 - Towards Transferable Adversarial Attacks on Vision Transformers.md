---
title: "论文 - Towards Transferable Adversarial Attacks on Vision Transformers"
aliases:
  - "Towards Transferable Adversarial Attack on Vision Transformers论文"
tags:
  - "知识库/论文阅读"
  - "主题/ViT与基础模型攻击"
笔记类型: "论文阅读"
分类: "ViT与基础模型攻击"
状态: "已整理"
摘要: "围绕 PNA 与 PatchOut，理解注意力梯度和随机 Patch 更新对 ViT 迁移攻击的作用。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/ViT与基础模型攻击/总结 - ViT与基础模型攻击]]"
相关笔记:
  - "[[20_深度学习/训练基础/Mask - 掩码机制详解]]"
  - "[[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - Adversarial Token Attacks on Vision Transformers]]"
  - "[[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类]]"
论文条目:
  - "[[论文来源目录#pna]]"
论文标题:
  - "Towards Transferable Adversarial Attacks on Vision Transformers"
发表载体:
  - "AAAI"
发表年份:
  - "2022"
发表类型:
  - "会议论文"
发表链接:
  - "https://ojs.aaai.org/index.php/AAAI/article/view/20169"
预印本链接:
  - "https://arxiv.org/abs/2109.04176"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/ViT与基础模型攻击/总结 - ViT与基础模型攻击|ViT与基础模型攻击总结]]

# 论文 - Towards Transferable Adversarial Attacks on Vision Transformers

**作者**：Zhipeng Wei, Jingjing Chen, Micah Goldblum, Zuxuan Wu, Tom Goldstein, Yu-Gang Jiang
**发表**：AAAI 2022

---

## 1. 研究背景与核心问题

- **Vision Transformer (ViT)** 在计算机视觉任务上表现优异，但依然容易受到对抗样本的攻击。
- **跨模型迁移性**（Cross-model transferability）允许攻击者在未知的黑盒模型上发起攻击。这种特性在 CNN 上已被广泛研究，但针对 ViT 的迁移攻击研究较少。
- **核心观察**：由于 ViT 与 CNN 在架构上有显著差异（ViT 接收图像块序列输入，使用多头自注意力机制），直接将 CNN 上的迁移攻击方法应用于 ViT 效果不佳。
- **本文目标**：设计专门针对 ViT 架构（同时考虑 **patch** 和 **self-attention**）的对抗攻击，以提高对抗样本在不同 ViT 之间、以及从 ViT 到 CNN 的迁移性。

---

## 2. 核心方法：双攻击框架（Dual Attack）

本文提出了一种双攻击框架，包含 **Pay No Attention (PNA)** 攻击和 **PatchOut** 攻击，分别针对 ViT 的自注意力机制和图像块输入特性。

### 2.1 Pay No Attention (PNA) 攻击

- **动机**：ViT 中的自注意力机制使得不同 patch 之间产生复杂的交互，这容易导致对抗样本过拟合于白盒模型，从而降低迁移性。
- **方法**：在反向传播计算梯度时，**忽略注意力权重对输入的梯度**（即将注意力权重视为常数，不通过注意力分支反传梯度）。
- **数学表达**：
  设输入 patch 嵌入为 $Z \in \mathbb{R}^{N \times D}$，注意力权重为：
  $$
  A = \text{softmax}(ZW^Q (ZW^K)^T / \sqrt{D_h})
  $$
  输出为 $Z' = A(ZW^V)$。其梯度分解为：
  $$
  \frac{\partial Z'}{\partial Z} = (\mathbb{I} \otimes A)\frac{\partial(ZW^V)}{\partial Z} + ((ZW^V)^T \otimes \mathbb{I})\frac{\partial A}{\partial Z}
  $$
  PNA 攻击直接设置 $\frac{\partial A}{\partial Z} = 0$，即忽略第二项。
- **效果**：迫使扰动仅利用特征表示，避免利用模型特有的注意力交互属性，从而提升迁移性。

### 2.2 PatchOut 攻击

- **动机**：受输入多样性（Diverse Inputs）启发，在每次迭代中只使用部分 patch，可以缓解过拟合。
- **方法**：在每次攻击迭代时，**随机采样一个 patch 子集** $T$ 进行优化，未被选中的 patch 保持不变。
- **损失函数**：
  $$
  \arg \max_{\delta} J(f(x + M \odot \delta), y) + \lambda \|\delta\|_2, \quad \text{s.t.} \|\delta\|_\infty < \epsilon
  $$
  其中 $M \in \{0,1\}^{H \times W \times C}$ 是掩码矩阵，仅在被选中的 patch 区域为 1；$\lambda \|\delta\|_2$ 是正则项，鼓励扰动具有较大的 $L_2$ 范数。

### 2.3 双攻击算法（Algorithm 1）

- **输入**：白盒 ViT 模型 $f$，干净图像 $x$ 及标签 $y$，扰动预算 $\epsilon$，迭代次数 $I$，每次使用的 patch 数 $T$。
- **流程**：
  1. 初始化 $\delta_0 = 0$，步长 $\alpha = \epsilon / I$。
  2. 迭代 $I$ 次：
     - 用 PatchOut 随机采样 $T$ 个 patch，生成掩码 $M$。
     - 计算带掩码的损失梯度。
     - 使用 PNA 反向传播（跳过注意力梯度）。
     - 更新扰动：$\delta_i = \text{clip}_\epsilon(\delta_{i-1} + \alpha \cdot g)$。
  3. 输出对抗样本 $x_{adv} = x + \delta_I$。

---

## 3. 实验设置

- **数据集**：ImageNet 2012 验证集，每个类别随机抽取一张正确分类的图像。
- **白盒模型（Surrogate）**：ViT-B/16、PiT-B、CaiT-S-24、Visformer-S。
- **黑盒模型（Victim）**：
  - **ViT 家族**：ViT-B/16、DeiT-B、TNT-S、LeViT-256、PiT-B、CaiT-S-24、ConViT-B、Visformer-S。
  - **CNN 家族**：Inception v3、Inception v4、Inception ResNet v2、ResNet v2-152。
  - **鲁棒训练 CNN**。
- **评价指标**：攻击成功率（ASR）和平均攻击成功率（MASR）。
- **攻击参数**：$\epsilon = 16$，迭代次数 $I = 10$，步长 $\alpha = 1.6$，图像尺寸 $224 \times 224$，patch 大小 $P = 16$，总 patch 数 $N = 196$。

---

## 4. 主要实验结果

### 4.1 迁移攻击 ViT（表 1）
- **对比基线**：FGSM、BIM、MI、DI、TI、SIM、SGM、IR、TAP、ATA、SE。
- **结果**：
  - TI 和 ATA 在 CNN 上表现好，但在 ViT 上表现极差（因依赖 CNN 的平移不变性等特性）。
  - SE 在部分 ViT 上表现不佳（因缺乏足够的 class token 构建集成）。
  - **本文方法**在多个黑盒 ViT 上取得最高的 MASR（平均 **58.67%**）。
  - 例如，ViT-B/16 上 ASR 达 **46.10%**，Visformer-S 上达 **58.60%**。

### 4.2 迁移攻击 CNN（表 2）
- **结果**：
  - 所有方法的 ASR 均有下降（ViT 到 CNN 的迁移更难）。
  - 本文方法仍显著优于基线：
    - 正常训练 CNN 上 MASR **42.61%**。
    - 鲁棒训练 CNN 上 MASR **22.38%**。
  - 证明了使用 ViT 作为白盒代理攻击 CNN 的可行性。

### 4.3 与现有方法结合（表 3）
- **MI + Ours**：ViTs 上 MASR 提升至 **59.70%**，鲁棒 CNN 上提升至 **30.13%**。
- **SGM + Ours**：ViTs 上 MASR 提升至 **63.48%**，鲁棒 CNN 上提升至 **31.37%**。
- 说明本文方法可与其他迁移攻击技术互补，进一步提升性能。

### 4.4 消融实验（表 4）
- **组件贡献**：
  - PNA 对提升迁移性贡献最大（ViTs 上从 26.97% 提升至 42.47%）。
  - PatchOut 和 $L_2$ 正则也分别有效。
  - **三者结合**达到最佳效果（ViTs 上 **59.15%**，正常 CNN **36.23%**，鲁棒 CNN **23.87%**）。
- **超参数**：
  - $T$ 的最优值为 **130**（总 patch 数 196）。
  - $\lambda$ 的最优值为 **0.1**。

---

## 5. 论文贡献总结

| 贡献 | 说明 |
| :--- | :--- |
| **PNA 攻击** | 提出在反向传播中忽略注意力权重的梯度，降低 patch 之间的交互，提升对抗样本的迁移性。 |
| **PatchOut 攻击** | 在每次迭代中随机采样部分 patch 进行优化，增加输入多样性，缓解过拟合。 |
| **双攻击框架** | 结合 PNA 和 PatchOut，同时处理 ViT 的自注意力机制和图像块输入，显著提升迁移性。 |
| **广泛验证** | 在白盒 ViT 迁移到黑盒 ViT、正常 CNN 和鲁棒 CNN 上均取得 SOTA 结果。 |
| **可组合性** | 可与 MI、SGM 等现有方法结合，进一步提升攻击成功率。 |
| **现象洞察** | 发现 ViT 与 CNN 的特征提取器差异没有想象中大，对抗样本可跨越巨大的架构鸿沟。 |

---

## 6. 结论

- ViT 的对抗迁移性可以通过**针对其架构特性**（自注意力 + patch 输入）设计的攻击方法显著提升。
- **PNA** 通过跳过注意力梯度，**PatchOut** 通过随机采样 patch，共同减少了对抗样本对白盒模型的过拟合。
- 本文方法不仅提高了 ViT 之间的迁移性，也提高了从 ViT 到 CNN 的迁移性，甚至对鲁棒训练的 CNN 也有效。
- 未来方向：探索对抗样本作为研究 ViT 与 CNN 特征空间相似性的工具。

<!-- knowledge-links:start -->

## 笔记联系

- [[20_深度学习/训练基础/Mask - 掩码机制详解|Mask - 掩码机制详解]]：回顾掩码控制计算参与范围的基础概念。
- [[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法|SGM - 跳跃梯度方法]]：比较残差分支梯度调整与注意力梯度处理。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - Adversarial Token Attacks on Vision Transformers|论文 - Adversarial Token Attacks on Vision Transformers]]：区分局部 Token 扰动约束与 PatchOut 的优化机制。
- [[20_深度学习/Transformer与ViT/MSA - 多头自注意力机制详解|MSA - 多头自注意力机制详解]]：回顾注意力矩阵与梯度流的计算方式。
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]：将 PNA/PatchOut 放入 ViT 专用攻击的分类。

## 论文出处

- [Towards Transferable Adversarial Attacks on Vision Transformers](https://ojs.aaai.org/index.php/AAAI/article/view/20169)：AAAI，2022，会议论文。
  [预印本](https://arxiv.org/abs/2109.04176)。

<!-- knowledge-links:end -->
