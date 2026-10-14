# 对抗攻击领域中的 ViT Attack 发展历程：从基础攻击到架构感知与迁移性优化

## 一、首先明确：什么是 ViT Attack？

ViT Attack 并不是某一种具有唯一标准定义的攻击算法，而是指**针对视觉 Transformer（Vision Transformer，ViT）模型开展的对抗攻击研究**。

这类研究主要关注：如何通过向输入图像添加微小扰动，使视觉 Transformer 做出错误的预测，以及如何让针对一个模型生成的对抗样本能够进一步攻击其他模型。

要理解 ViT Attack 的发展历程，需要先认识它所针对的模型。

### 1.1 什么是 Vision Transformer？

Vision Transformer（ViT）是一类将 Transformer 架构应用于计算机视觉任务的模型。

传统卷积神经网络（CNN）主要通过卷积操作提取图像的局部特征，而经典 ViT 则采用不同的处理方式。

以图像分类任务为例，ViT 的处理流程可以概括为：

1. **图像分块（Patch Partition）**：将输入图像划分为多个小图像块。
2. **嵌入映射（Patch Embedding）**：将每个图像块转换成向量，也就是 Token。
3. **位置编码（Position Embedding）**：为 Token 提供位置信息。
4. **自注意力计算（Self-Attention）**：建模不同 Token 之间的关系。
5. **特征变换（MLP）**：进一步处理 Token 的特征。
6. **分类预测**：利用分类 Token 或其他聚合方式输出预测结果。

与 CNN 相比，经典 ViT 的一个重要特点是：它能够通过自注意力机制建模不同图像区域之间的关系。

但这并不意味着 ViT 天然具有更强的安全性。

研究人员发现，ViT 同样容易受到对抗扰动影响，而且针对 ViT 生成的对抗样本与针对 CNN 生成的对抗样本，在攻击效果和跨模型迁移能力方面存在值得研究的差异。

这就引出了 ViT Attack 的核心问题。

### 1.2 ViT Attack 主要研究什么？

ViT Attack 的研究可以分成几个相互关联、但不能混为一谈的方向。

| 研究方向       | 核心问题                            | 主要目标                        |
| ---------- | ------------------------------- | --------------------------- |
| 白盒对抗攻击     | 已知模型参数和梯度，如何生成对抗样本？             | 提高对目标 ViT 的攻击成功率            |
| 黑盒对抗攻击     | 无法直接访问目标模型内部信息，如何攻击？            | 在信息受限的条件下实现攻击               |
| 跨 ViT 迁移攻击 | 针对一个 ViT 生成的样本，能否攻击另一个 ViT？     | 提高不同 Transformer 模型之间的迁移性   |
| 跨架构迁移攻击    | 针对 ViT 生成的样本，能否攻击 CNN？          | 提高 Transformer 与 CNN 之间的迁移性 |
| 架构感知攻击     | 如何利用 ViT 的 Token、自注意力等结构特点设计攻击？ | 提高攻击效率与迁移能力                 |
| 对抗鲁棒性研究    | ViT 为什么容易或不容易受到攻击？              | 分析漏洞并为防御方法提供依据              |

其中，**针对 ViT 的跨模型迁移攻击，是这条研究路线中非常重要的分支**。

不过，ViT Attack 的发展并不完全等同于迁移攻击的发展。早期研究还关注模型自身的鲁棒性、不同攻击方法的有效性以及 ViT 与 CNN 的安全性差异。

下面按照时间顺序展开。

---

# 二、第一阶段：ViT 对抗鲁棒性的初步研究（2021 年）

## 2.1 研究背景：CNN 对抗攻击已经积累了大量成果

在 ViT 成为计算机视觉领域的重要架构之前，对抗攻击研究主要围绕 CNN 展开。

当时，研究人员已经提出了多种经典攻击方法。

例如：

- FGSM：快速梯度符号法。
- BIM：基本迭代法。
- PGD：投影梯度下降攻击。
- MIM：动量迭代攻击。
- C&W：Carlini-Wagner 攻击。

这些方法从不同角度利用模型的梯度、损失函数或优化过程生成对抗样本。

但是，CNN 与 ViT 的内部结构存在明显区别。

CNN 通过卷积核提取局部特征，而 ViT **将图像转换为 Token**，并通过自注意力机制建立不同 Token 之间的联系。

因此，一个自然的问题出现了：

**针对 CNN 设计的攻击方法，是否同样适用于 ViT？ViT 是否比 CNN 更加鲁棒？**

2021 年的研究开始系统回答这些问题。

## 2.2 代表性工作一：On the Robustness of Vision Transformers

2021 年，Shao 等人的研究《On the Adversarial Robustness of Vision Transformers》对 ViT 的对抗鲁棒性进行了研究。

论文：[On the Adversarial Robustness of Vision Transformers](https://arxiv.org/abs/2103.15670?utm_source=chatgpt.com)

这项工作的主要价值在于，它不只是检验 ViT 是否会受到攻击，还尝试分析不同架构在对抗鲁棒性方面的差异。

研究主要关注以下问题。

**第一，ViT 是否比 CNN 更加鲁棒？**

研究人员比较了不同视觉架构在对抗扰动下的表现。

他们发现，ViT 在所研究的设置中表现出了相对较强的对抗鲁棒性。

但这里必须注意：这并不意味着所有 ViT 都比所有 CNN 更加鲁棒。具体结果会受到模型架构、训练方式、数据集和攻击方法等因素影响。

**第二，ViT 为什么可能表现出不同的鲁棒性？**

论文通过频率分析和特征可视化等方法，研究模型对不同图像信息的利用方式。

其中一个重要观察是：

ViT 学到的特征相对较少依赖某些高频模式，而这些高频模式可能包含容易被扰动利用的虚假相关信息。

这为理解 ViT 与 CNN 在对抗攻击方面的差异提供了一个重要方向。

**第三，改变 ViT 的特征提取方式会产生什么影响？**

研究还考察了引入卷积结构、Token-to-Token 模块等设计对模型鲁棒性的影响。

这说明，模型的对抗鲁棒性不仅与是否使用 Transformer 有关，也与具体的特征提取机制有关。

### 这项工作的历史意义

它将 ViT 的对抗鲁棒性问题带入了更加系统的研究框架。

此前的很多攻击方法主要围绕 CNN 的结构和行为展开，而这项研究表明：

1. ViT 同样存在对抗攻击风险。
2. 不同视觉架构的鲁棒性表现可能存在差异。
3. 模型所学习的特征及其频率偏好，可能影响攻击效果。

这为后续专门研究 ViT 攻击提供了基础。

## 2.3 代表性工作二：On the Robustness of Vision Transformers to Adversarial Examples

同样在 2021 年，Mahmood 等人发表了另一项重要研究。

论文：[On the Robustness of Vision Transformers to Adversarial Examples（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html?utm_source=chatgpt.com)

这项工作从多个方面研究了 ViT 的安全性。

### 方向一：使用传统攻击方法测试 ViT(白盒)

研究人员采用 FGSM、MIM、PGD、APGD 和 C&W 等攻击方法测试 ViT。

实验表明，ViT 并不能免疫常见的白盒对抗攻击。在论文所采用的设置下，强迭代攻击能够有效破坏模型的预测结果。

这说明：

**不能因为 ViT 使用了自注意力机制，就认为传统梯度攻击失效。**

传统攻击方法依然是研究 ViT 安全性的基础工具。

### 方向二：研究 CNN 与 ViT 之间的攻击迁移

假设攻击者知道一个 CNN 的结构和参数，却不知道目标 ViT 的具体情况。

攻击者可以先针对 CNN 生成对抗样本，再将其输入目标 ViT。

问题在于：这些样本能否成功攻击目标模型？

该研究发现，CNN 与 ViT 之间的对抗样本并不容易直接迁移。

这揭示了一个重要现象：

针对某种架构生成的对抗样本，可能高度依赖该架构所学习的特征和决策边界。

即使两种模型完成相同的分类任务，它们对同一幅图像的内部处理方式也可能存在差异。

### 方向三：提出 Self-Attention Blended Gradient Attack

该研究还提出了 Self-Attention Blended Gradient Attack，简称 SABG。

其核心思想是利用自注意力相关信息，对攻击所使用的梯度进行融合，从而改善跨架构攻击的效果。

这一思路的重要性在于：

研究人员开始不再满足于直接将现有攻击算法应用于 ViT，而是尝试利用 ViT 自身的结构特点设计攻击。

### 第一阶段的总体结论

2021 年的研究为后续工作提出了三个重要问题：

- 如何充分利用 ViT 的结构特点设计攻击？
- 如何提高针对 ViT 的对抗样本的跨模型迁移能力？
- 为什么 ViT 与 CNN 之间的迁移效果往往不够理想？

这三个问题构成了随后几年 ViT Attack 研究的重要基础。

参考：上述两篇论文分别发表于 2021 年的相关研究阶段。[arXiv](https://arxiv.org/abs/2103.15670?utm_source=chatgpt.com)

---

# 三、第二阶段：面向 ViT 结构的专门攻击方法（2021—2022 年）

进入这一阶段，研究重点逐渐从测试 ViT 是否容易受到攻击，转向：

**能否利用 ViT 的特有结构，设计比直接套用传统攻击方法更有效的攻击策略？**

其中最有代表性的工作之一，就是 PNA 与 PatchOut。

## 3.1 代表性工作：Towards Transferable Adversarial Attacks on Vision Transformers

论文：

[Towards Transferable Adversarial Attacks on Vision Transformers（AAAI 2022）](https://ojs.aaai.org/index.php/AAAI/article/view/20169?utm_source=chatgpt.com)

作者：Zhipeng Wei 等人。

该论文于 2021 年以预印本形式公开，并于 2022 年发表于 AAAI。

它是 ViT 对抗迁移攻击研究中具有代表性的早期工作。

### 3.1.1 研究动机

当时，传统攻击方法已经能够生成攻击 ViT 的对抗样本，但仍然面临一个问题：

**在白盒模型上攻击成功，并不代表在其他未知模型上同样有效。**

例如，攻击者使用 ViT-B/16 作为代理模型生成对抗样本。

生成的样本能够使 ViT-B/16 分类错误，但换成其他 ViT 或 CNN 后，攻击成功率可能明显下降。

因此，该研究认为，ViT 的迁移攻击应该同时考虑两个关键因素：

1. 图像是由多个 Patch 构成的。
2. Transformer 使用自注意力机制建模 Token 之间的关系。

基于这一认识，作者提出了一个包含两种攻击策略的框架：

- Pay No Attention（PNA）。
- PatchOut。

两种方法分别从梯度传播和图像 Patch 选择的角度改进攻击。

---

## 3.2 PNA：Pay No Attention

### 3.2.1 为什么需要研究注意力机制？

在 ViT 中，自注意力机制负责建模不同 Token 之间的关系。

简化地说，给定输入 Token 表示矩阵 \(X\)，注意力机制可以写为：

$$
[ Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V ]
$$

其中：

- $(Q)$：Query 矩阵。
- $(K)$：Key 矩阵。
- $(V)$：Value 矩阵。
- $(W_Q,W_K,W_V)$：可学习参数矩阵。

注意力权重通常计算为：

$$
[ A=\operatorname{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right) ]
$$

注意力输出为：

$$
[ Z=AV ]
$$

这里的 $(A)$ 描述不同 Token 之间的注意力关系。

在普通反向传播过程中，梯度会沿着模型的各个计算路径传播，其中包括与注意力计算相关的路径。

PNA 的核心思路是：

**在生成对抗样本时，跳过自注意力部分的梯度传播，使攻击更多地利用其他梯度路径。**

需要注意，这并不是删除模型的自注意力模块，也不是在正常推理时关闭注意力机制。

模型前向计算仍然使用注意力，而攻击时对反向传播过程进行调整。

### 3.2.2 PNA 为什么可能提高迁移性？

首先理解一个概念：对抗样本可能过度适应代理模型。

假设代理模型为 $(f_s)$，目标模型为 $(f_t)$。

攻击者通过优化代理模型的损失函数生成扰动：

$$
[ x_{\mathrm{adv}}=x+\delta ]
$$

其中：

- $(x)$ 是原始图像。
- $(\delta)$ 是对抗扰动。
- $(x_{\mathrm{adv}})$ 是对抗样本。

如果扰动过度利用代理模型中特有的决策特征，那么它可能只对代理模型有效。

相反，如果攻击过程能够减少对某些模型特有梯度信息的依赖，就有可能生成对其他模型也有效的扰动。

PNA 的设计正是基于这一思路。

不过，需要强调的是，跳过注意力梯度并不保证在所有模型和数据集上都能提高迁移性。它是一种经过实验验证的攻击策略，其效果依赖具体实验设置。

### 3.2.3 PNA 的历史意义

PNA 的价值在于，它将攻击设计与 ViT 的内部计算结构直接联系起来。

与简单地将 FGSM、MIM 等方法应用于 ViT 相比，PNA 更明确地考虑了 Transformer 的注意力机制。

这意味着 ViT Attack 开始进入一种新的研究阶段：

不再只研究怎样利用损失梯度，而是进一步研究**应该利用哪些梯度，以及如何改变梯度传播路径**。

---

## 3.3 PatchOut：从图像 Patch 的角度改进攻击

PNA 关注注意力梯度，而 PatchOut 关注另一个问题：

在每次迭代生成对抗样本时，是否有必要同时利用图像中的全部 Patch？

### 3.3.1 传统方法的问题

对于一幅图像，ViT 通常会将其划分**成多个 Patch**。

假设输入图像被划分成 \(N\) 个 Patch：
$$
[ X=\{x_1,x_2,\ldots,x_N\} ]
$$

传统攻击方法通常基于完整图像计算梯度，再更新全部像素对应的扰动。

但不同 Patch 对攻击目标的贡献可能不同。

如果每次迭代都按照同一种方式利用全部 Patch，攻击可能更容易利用代理模型中特定的局部特征。

这会影响生成的对抗样本在其他模型上的表现。

### 3.3.2 PatchOut 的核心思想

PatchOut 在攻击过程中随机选取部分 Patch，并基于选中的 Patch 进行扰动优化。

可以将这一过程理解为：

1. 将输入图像划分为多个 Patch。
2. 在每次迭代时随机选取其中一部分。
3. 基于选中的 Patch 计算相应的攻击梯度。
4. 根据梯度更新对抗扰动。
5. 在后续迭代中重新选择 Patch。

通过这种方式，攻击过程不再始终依赖完全相同的 Patch 组合。

这相当于为攻击过程引入了某种**随机化机制**。

### 3.3.3 为什么随机选择 Patch 可能有效？

直观地说，如果每次迭代都只依赖固定的图像区域，攻击可能会过度利用这些区域的特定信息。

而随机选择 Patch 可以让优化过程接触不同的局部区域，从而改变扰动的形成方式。

这可能降低攻击对某些特定特征的依赖，提高跨模型迁移能力。

但 PatchOut 并不意味着被选中的 Patch 必然比其他 Patch 更重要，也不意味着 Patch 越少，攻击效果就一定越好。

选择比例和随机化策略都会影响最终结果。

### 3.3.4 PNA 与 PatchOut 的区别

|比较维度|PNA|PatchOut|
|---|---|---|
|主要研究对象|自注意力相关的梯度传播|图像 Patch 的选择|
|核心操作|跳过注意力部分的梯度传播|每次迭代随机选择部分 Patch|
|主要目标|改变攻击所利用的梯度信息|改变攻击所利用的图像区域|
|研究层面|模型计算结构|输入图像的分块结构|
|共同目标|提高对抗样本的迁移能力|提高对抗样本的迁移能力|

论文提出的双重攻击框架能够结合这两种思路。

实验结果表明，这些方法能够改善不同 ViT 之间以及 ViT 与 CNN 之间的攻击迁移效果。[AAAI Publications](https://ojs.aaai.org/index.php/AAAI/article/view/20169?utm_source=chatgpt.com)

### 这一阶段的历史意义

PNA 与 PatchOut 的出现，标志着 ViT Attack 开始更加系统地利用 Transformer 的架构特点。

从研究方法来看，可以概括为：

- PNA：从反向传播机制入手。
- PatchOut：从输入 Patch 的选择入手。

这两条思路也为后续研究提供了启发：ViT 的攻击不仅可以在像素空间进行优化，还可以通过修改模型内部的梯度计算和特征处理方式来改进。

---

# 四、第三阶段：从普通梯度优化转向 Token 级梯度调节（2023 年）

进入 2023 年，ViT 迁移攻击的研究开始进一步关注一个问题：

**为什么传统的梯度优化方法在 ViT 上仍然可能产生迁移性不足的对抗样本？**

这时，研究人员开始更加细致地分析 ViT 内部不同层、不同 Token 的梯度。

## 4.1 代表性工作一：Token Gradient Regularization

论文：

[Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization](https://arxiv.org/abs/2303.15754?utm_source=chatgpt.com)

作者：Jianping Zhang、Yizhan Huang、Weibin Wu、Michael R. Lyu。

论文于 2023 年公开。

### 4.1.1 研究背景

在传统迭代攻击中，常见的做法是根据输入图像的梯度更新扰动。

以动量迭代攻击为例，可以用下列形式表示：

$$
[ g_{t+1} = \mu g_t+ \frac{\nabla_x L(x_t,y)} {\|\nabla_x L(x_t,y)\|_1} ]
$$

$$
[ x_{t+1} = \Pi_{\mathcal{B}(x,\epsilon)} \left[ x_t+\alpha\operatorname{sign}(g_{t+1}) \right] ]
$$

其中：

- $(x_t)$ 是第 $(t)$ 次迭代的对抗样本。
- $(L)$ 是攻击目标对应的损失函数。
- $(g_t)$ 是累计梯度。
- $(\mu)$ 是动量系数。
- $(\alpha)$ 是更新步长。
- $(\Pi_{\mathcal{B}(x,\epsilon)})$ 表示将结果投影到原始图像附近的允许扰动范围内。

这类方法通过梯度和动量改善扰动优化过程。

但 ViT 具有多层 Transformer Block，每一层又包含多个 Token。

因此，输入梯度只是反向传播过程的最终结果之一。

在中间层中，不同 Token 的梯度可能具有不同的变化规律。

如果只关注最终输入梯度，就可能忽略内部梯度的波动。

### 4.1.2 TGR 的核心思想

Token Gradient Regularization，简称 TGR，提出了针对 ViT 内部 Token 梯度的正则化思路。

它关注的问题是：

不同 Transformer Block 中的梯度变化，可能导致生成的对抗扰动过度依赖代理模型特有的特征。

TGR 尝试以 Token 为单位调节内部梯度，降低梯度方差，从而改善生成的对抗样本的迁移能力。

这里的 Token 级处理非常重要。

ViT 的中间表示通常可以写成：

$$
[ X^{(l)} = \left[ x_1^{(l)}, x_2^{(l)}, \ldots, x_N^{(l)} \right] ]
$$

其中 $(X^{(l)})$ 表示第 $(l)$ 层的 Token 表示。

TGR 的研究重点不是单纯调整最终输入图像的梯度，而是关注各层 Token 相关的反向传播梯度。

通过这种方式，它试图减轻中间层梯度波动带来的影响。

需要注意，以上表达用于说明其研究动机和处理对象，并非完整复现论文中的正则化公式。

### 4.1.3 TGR 与 PNA 有什么区别？

两者都利用了 ViT 的内部结构，但具体切入点不同。

|比较维度|PNA|TGR|
|---|---|---|
|主要关注对象|自注意力相关梯度路径|中间层 Token 梯度|
|核心思路|跳过部分注意力梯度传播|对 Token 级梯度进行正则化|
|关注的问题|如何改变攻击利用的梯度路径|如何降低中间层梯度的波动|
|架构利用方式|注意力机制|Token 与 Transformer Block|

可以看到，研究逐渐从对某个计算模块进行简单处理，发展到对模型内部梯度进行更加细粒度的分析和调节。

TGR 论文报告了针对 ViT 和 CNN 的迁移攻击实验，并展示了其相对于当时相关方法的改进效果。[arXiv](https://arxiv.org/abs/2303.15754?utm_source=chatgpt.com)

---

## 4.2 代表性工作二：Momentum Integrated Gradients

论文：

[Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients（ICCV 2023）](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html?utm_source=chatgpt.com)

作者：Wenshuo Ma、Yidong Li、Xiaofeng Jia、Wei Xu。

这篇论文于 2023 年发表于 ICCV。

它主要研究的问题是：

**如何提高对抗样本在 ViT 与 CNN 之间的迁移能力？**

### 4.2.1 为什么跨架构迁移比较困难？

ViT 和 CNN 虽然都能够完成图像分类任务，但它们处理图像的方式存在差异。

CNN 通过卷积等操作提取局部特征，再逐步组合成更高层次的表示。

ViT 则通过 Patch Embedding 和自注意力机制处理图像 Token。

因此，两种模型的梯度可能指向不同的方向。

即使某个扰动能够有效改变 ViT 的预测结果，也不意味着相同的扰动能够有效改变 CNN 的预测结果。

这就需要研究一种能够缓解梯度差异的攻击方法。

### 4.2.2 什么是 Integrated Gradients？

Integrated Gradients，简称 IG，是一种用于分析模型预测对输入特征依赖程度的方法。

给定基线输入 \(x'\) 和目标输入 \(x\)，其基本思想是沿着两者之间的路径累积梯度。

第 \(i\) 个输入维度的积分梯度可以写为：

$$
[ \operatorname{IG}_i(x) = (x_i-x'_i) \int_0^1 \frac{\partial F(x'+\alpha(x-x'))} {\partial x_i} \,d\alpha ]
$$

其中：

- $(F)$ 是模型输出的某个标量函数。
- $(x')$ 是基线输入。
- $(x)$ 是实际输入。
- $(\alpha)$ 用于描述从基线到输入的路径。

与只在某个输入点计算梯度相比，IG 考虑了整条路径上的梯度信息。

需要注意，IG 原本是用于归因分析的方法，而 MIG 将其用于对抗攻击的梯度构造。

### 4.2.3 MIG 的核心思想

Momentum Integrated Gradients，简称 MIG，结合了两个思路：

1. 利用积分梯度构造攻击方向。
2. 利用动量机制积累多个迭代阶段的梯度信息。

作者观察到，在其研究设置中，不同模型的积分梯度具有比普通梯度更高的相似性。

基于这一现象，MIG 使用积分梯度引导对抗扰动的更新，并通过动量积累不同迭代阶段的信息。

它的目标是降低攻击对单个模型局部梯度的依赖，提高跨模型迁移性。

### 4.2.4 MIG 的历史意义

如果说 PNA 关注的是注意力梯度路径，TGR 关注的是中间层 Token 梯度，那么 MIG 关注的则是：

**如何利用更具代表性的梯度信息，提高不同架构之间攻击方向的一致性。**

这一研究表明，ViT Attack 并不一定要完全依赖 ViT 特有的内部模块，也可以从梯度估计和梯度累积方式入手改善攻击效果。

论文报告的实验表明，MIG 在不同 ViT 和 CNN 模型之间表现出了较强的迁移能力。[Open Access CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html?utm_source=chatgpt.com)

---

# 五、第四阶段：从单一架构优化转向跨架构共同特征研究（2023 年）

2023 年，ViT Attack 领域出现了另一条值得关注的路线：

**与其只研究如何更好地攻击 ViT，不如进一步分析 ViT 和 CNN 为什么会受到不同类型的对抗扰动影响，并利用两种架构之间的共同点提高迁移性。**

这一思路代表了从模型内部梯度优化，向架构间特征差异分析的扩展。

## 5.1 代表性工作：Understanding and Improving Adversarial Transferability of Vision Transformers and Convolutional Neural Networks

论文：

[Understanding and Improving Adversarial Transferability of Vision Transformers and Convolutional Neural Networks](https://www.sciencedirect.com/science/article/pii/S0020025523010599?utm_source=chatgpt.com)

该研究发表于 2023 年的 _Information Sciences_。

论文的研究重点是 ViT 与 CNN 之间的对抗迁移性。

### 5.1.1 研究动机

前面的研究已经发现：

针对一个模型生成的对抗样本，不一定能够有效攻击另一种架构。

那么，这种差异究竟来自哪里？

研究人员将注意力放到了不同模型的**归纳偏置**（Inductive Bias）上。

### 5.1.2 什么是归纳偏置？

归纳偏置可以理解为：

**模型由于自身结构、训练方式等因素，在学习数据规律时表现出的某些倾向。**

例如，CNN 的局部连接和卷积操作会影响它对局部空间模式的建模方式。

经典 ViT 则通过 Patch Token 和自注意力机制建立特征之间的联系。

这些结构差异会影响模型学习什么特征，以及如何利用这些特征完成分类。

因此，两个模型即使训练在相同的数据集上，也可能形成**不同的决策方式**。

### 5.1.3 Inductive Bias Attack 的核心思路

论文提出了 Inductive Bias Attack。

它的目标是通过研究和利用 ViT 与 CNN 的归纳偏置差异，提高跨架构攻击的迁移性。

其方法涉及两个方面：

**第一，抑制 ViT 的某些独有归纳偏置。**

研究通过平滑 Patch 边缘处的梯度等方式，尝试减弱扰动对 ViT 特有结构的依赖。

**第二，利用两类模型的共同特征偏好。**

研究还分析了原始图像与对抗样本的高频、低频信息，并利用这些信息设计攻击目标，以改善跨架构迁移。

这里的关键不是简单地认为高频信息一定有害、低频信息一定有益，而是通过实验分析不同模型在不同频率信息上的表现，再据此设计攻击策略。

### 5.1.4 这一工作的历史意义

这一阶段的研究视角进一步扩大了。

早期工作主要研究：

- 怎样攻击 ViT。
- 怎样修改 ViT 的注意力梯度。
- 怎样利用 ViT 的 Patch 结构。

而 Inductive Bias Attack 则进一步提出：

- ViT 与 CNN 的特征偏好有什么区别？
- 哪些模型特征是特定架构独有的？
- 哪些特征可能被不同架构共同利用？
- 能否减少对架构独有特征的依赖，从而提高迁移性？

这使 ViT Attack 不再只是模型内部的梯度优化问题，也成为分析不同视觉架构的特征学习机制和决策行为的研究方向。[科学通报](https://www.sciencedirect.com/science/article/pii/S0020025523010599?utm_source=chatgpt.com)

---

# 六、第五阶段：虚拟密集连接与更细粒度的梯度利用（2023—2024 年）

随着迁移攻击研究的推进，研究人员开始进一步关注 ViT 的层间信息传递方式。

这里的关键问题是：

**ViT 的中间层中包含丰富的特征信息，但常规攻击过程不一定能够充分利用这些信息。**

## 6.1 代表性工作：Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection

论文：

[Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/view/28541?utm_source=chatgpt.com)

作者：Jianping Zhang、Yizhan Huang、Zhuoer Xu、Weibin Wu、Michael R. Lyu。

该工作发表于 AAAI 2024。

### 6.1.1 研究背景

在 ViT 中，输入图像经过多个 Transformer Block，逐步形成不同层次的特征表示。

可以用下面的方式简化表示：

$$
[ X^{(1)} \rightarrow X^{(2)} \rightarrow X^{(3)} \rightarrow \cdots \rightarrow X^{(L)} ]
$$

其中 $(X^{(l)})$ 表示第 $(l)$ 层的特征。

不同层可能学习到不同的特征信息。

但如果攻击方法主要依赖某个固定的前向计算过程和最终输入梯度，就可能无法充分利用不同层之间的特征联系。

### 6.1.2 Virtual Dense Connection 的核心思想

Virtual Dense Connection，简称 VDC，研究如何通过虚拟密集连接改善 ViT 的对抗迁移性。

可以先从概念上理解 Dense Connection。

密集连接的基本思想是，让某一层能够直接利用之前多个层的特征，而不是只接收紧邻上一层的输出。

VDC 则借鉴了这一思想，在攻击框架中引入虚拟密集连接机制，以改善攻击过程中不同层之间的信息利用。

需要特别注意：

VDC 并不是简单地将原始 ViT 永久改造成一个具有标准 DenseNet 结构的网络。它的重点是利用虚拟连接机制改进对抗攻击过程。

### 6.1.3 为什么这可能有助于迁移攻击？

不同模型的中间层表示并不完全相同。

如果攻击过程能够更充分地利用多个层次的信息，就有可能减少对某一层特定特征的过度依赖。

因此，VDC 试图从层间信息利用的角度改善攻击迁移性。

不过，不能仅凭这一动机就断言所有中间层信息都有助于迁移攻击。最终效果仍然需要实验验证。

### 6.1.4 历史意义

这一工作体现了 ViT Attack 研究的一种发展趋势：

从输入端和注意力机制，进一步扩展到模型内部的层间特征传递。

将前面几类方法放在一起看：

|方法|主要切入点|
|---|---|
|PNA|自注意力相关梯度路径|
|PatchOut|图像 Patch 的选择|
|TGR|中间层 Token 梯度|
|MIG|积分梯度与动量|
|Inductive Bias Attack|不同架构的特征偏置|
|VDC|Transformer Block 之间的信息利用|

这说明 ViT Attack 已经形成多条相互关联的技术路线，而不是单纯依赖某一种优化算法。

---

# 七、第六阶段：前向传播与反向传播的协同优化（2024 年）

此前不少迁移攻击方法主要通过修改梯度、正则化梯度或改变反向传播过程来提高攻击效果。

但这类方法存在一个值得继续研究的问题：

**如果只改变反向传播过程，是否能够充分利用 ViT 的结构特点？**

2024 年的部分工作开始更加明确地同时考虑**前向传播与反向传播**。

## 7.1 代表性工作：Forward-Backward Transferable Adversarial Attack

论文：

[Towards transferable adversarial attacks on vision transformers for image classification](https://www.sciencedirect.com/science/article/pii/S1383762124000924?utm_source=chatgpt.com)

该论文发表于 _Journal of Systems Architecture_，2024 年。

作者提出 Forward-Backward Transferable Adversarial Attack 框架，简称 FBTA。

### 7.1.1 研究动机

迁移攻击通常使用一个可访问的代理模型生成对抗样本，然后将其应用于未知的目标模型。

为了提高迁移性，传统研究主要从两个方向改进：

- **前向方向**：通过输入变换或模型状态变化，增加攻击所面对的模型多样性。
- **反向方向**：通过调整梯度，降低扰动对代理模型的过度依赖。

FBTA 认为，ViT 的中间层结构能够为这两个过程提供更多可利用的信息。

### 7.1.2 DTA：Dropout-based Transferable Attack

FBTA 的前向部分提出 Dropout-based Transferable Attack，简称 DTA。

其思路是在 ViT 的注意力模块和 MLP 模块中引入 Dropout，以增加中间状态的多样性。

这里的 Dropout 不只是通常意义上用于训练时正则化的工具，而是被用于攻击框架中的模型状态多样化。

直观地说，通过改变部分中间计算状态，攻击可以减少对单一确定前向计算路径的依赖。

这种设计具有一定的集成效应：攻击过程会接触不同的中间状态，而不是始终依赖完全相同的内部表示。

### 7.1.3 BGC：Backpropagation Gradient Clipping

FBTA 的反向部分提出 Backpropagation Gradient Clipping，简称 BGC。

它的目标是对 ViT 中间层的梯度进行细致调节。

此前的研究已经表明，直接使用原始梯度可能使扰动过度适应代理模型。

BGC 尝试通过梯度裁剪或限制梯度中的极端值，改善梯度信息的利用方式。

不过，这并不意味着所有较大的梯度都没有价值。

梯度裁剪的效果取决于具体的裁剪策略和实验设置。

### 7.1.4 FBTA 的特点

FBTA 将两个方向结合起来：

**[ \text{前向传播多样化} + \text{反向传播梯度调节} ]**

前者试图增加中间状态的多样性，后者试图减少梯度中的不利波动。

这体现了一种更系统的攻击设计思路：

不是孤立地修改某一个梯度公式，而是同时考虑模型前向计算和反向传播过程。

论文报告的实验表明，该方法在其测试的 ViT 和鲁棒 CNN 等模型上取得了迁移攻击性能提升。[科学通报](https://www.sciencedirect.com/science/article/pii/S1383762124000924?utm_source=chatgpt.com)

---

# 八、第七阶段：从反向传播优化走向前向传播精细化（2025 年）

2025 年，ViT 迁移攻击研究继续沿着架构感知的方向发展。

其中一项代表性工作将研究重点放在前向传播本身。

## 8.1 代表性工作：Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement

论文：

[Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement（CVPR 2025）](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Improving_Adversarial_Transferability_on_Vision_Transformers_via_Forward_Propagation_Refinement_CVPR_2025_paper.html?utm_source=chatgpt.com)

作者：Yuchen Ren 等人。

该论文发表于 CVPR 2025。

### 8.1.1 为什么需要 Forward Propagation Refinement？

此前不少方法主要研究反向传播过程，例如：

- 调节注意力梯度。
- 对 Token 梯度进行正则化。
- 利用多层信息改进梯度。
- 裁剪中间层梯度。

这些方法能够改善攻击，但它们主要从梯度形成和传播的角度入手。

而前向传播决定了输入图像经过模型时如何形成中间表示。

如果前向传播所产生的特征过于依赖某些特定的模型状态，那么后续计算得到的梯度也可能受到这些状态的影响。

因此，该研究提出：

能否直接优化代理模型的前向传播过程，使其更适合生成具有迁移性的对抗样本？

### 8.1.2 Forward Propagation Refinement 的核心思想

该方法简称 FPR，主要针对 ViT 的两个关键模块开展研究：

1. Attention Maps。
2. Token Embeddings。

其中，Attention Maps 描述 Token 之间的注意力分配关系。

Token Embeddings 则是输入图像经过分块、嵌入映射等操作后形成的向量表示。

论文提出两种对应的方法。

**第一，Attention Map Diversification（AMD）。**

AMD 的目标是增加部分注意力图的多样性。

其思路与前面 DTA 的模型状态多样化有一定关联，但二者并不是同一种方法。

DTA 在注意力模块和 MLP 模块中引入 Dropout，而 AMD 则专门关注注意力图的多样化。

**第二，Token Embedding Refinement。**

这一部分关注 Token 嵌入的精细化处理，尝试通过调整前向传播中的 Token 表示，提高对抗样本的迁移性。

需要注意，Token Embedding Refinement 是对该研究另一个方向的概括性描述；具体实现和优化目标应以论文正文为准。

### 8.1.3 FPR 的历史意义

FPR 进一步说明，ViT Attack 的研究不再局限于：

> 给定模型，如何通过更好的梯度更新方式生成对抗样本？

研究问题逐渐变成：

> 如何改变攻击过程中代理模型的内部计算状态，使生成的对抗样本更容易迁移到其他模型？

这使前向传播本身也成为攻击优化的重要研究对象。

从研究思路看，FPR 与 FBTA 都涉及前向传播优化，但它们有不同的具体方法和技术设计，不能将二者视为同一个算法。[Open Access CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Improving_Adversarial_Transferability_on_Vision_Transformers_via_Forward_Propagation_Refinement_CVPR_2025_paper.html?utm_source=chatgpt.com)

---

# 九、将整个发展历程串联起来：ViT Attack 的研究主线

前面介绍了多篇具有代表性的论文。为了真正理解这个领域，还需要将它们放在同一条发展主线上。

这里按照研究问题的演进，而不只是按照论文发表年份进行总结。

## 9.1 第一条主线：从验证漏洞到理解架构差异

早期研究主要回答：

- ViT 是否容易受到对抗攻击？
- 传统攻击方法是否适用于 ViT？
- ViT 与 CNN 的鲁棒性有什么区别？

代表性工作是 2021 年的两项鲁棒性研究。

这些研究建立了一个基本认识：

ViT 同样存在对抗攻击风险，但它的特征学习方式与 CNN 存在差异。

这意味着，不能简单地假设针对 CNN 有效的攻击策略在 ViT 上具有相同效果。

## 9.2 第二条主线：从通用攻击到架构感知攻击

随后，研究人员开始利用 ViT 的结构特点设计专门的攻击方法。

代表性工作包括：

- PNA：调整自注意力相关梯度的传播。
- PatchOut：随机选择部分图像 Patch。
- TGR：在 Token 级别处理梯度。
- VDC：改善不同层之间的信息利用。

这些方法虽然切入点不同，但具有一个共同特点：

**它们不再将 ViT 仅仅看作一个能够计算梯度的分类器，而是将其内部结构视为攻击设计的重要组成部分。**

## 9.3 第三条主线：从单模型攻击到跨模型迁移攻击

攻击一个白盒模型，与攻击一个未知的黑盒模型，是两个不同的问题。

在迁移攻击中，研究人员通常利用代理模型生成对抗样本，再将样本应用于其他模型。

这要求扰动不仅能够改变代理模型的预测，还需要具有跨模型的有效性。

MIG、Inductive Bias Attack 等工作从不同角度研究这一问题：

- MIG：利用积分梯度和动量。
- Inductive Bias Attack：研究不同架构的特征偏置。
- TGR：调节中间层 Token 梯度。
- VDC：利用不同层之间的信息联系。

这些方法的共同目标是减少攻击对单个代理模型的过度依赖。

## 9.4 第四条主线：从反向传播优化到前向与反向协同优化

到了 2024—2025 年，研究开始更加系统地考虑攻击过程中模型内部的计算过程。

FBTA 同时涉及前向状态多样化和反向梯度调节。

FPR 则将重点放在前向传播中的注意力图和 Token 嵌入上。

这反映出一个重要趋势：

攻击迁移性不仅与最终的输入梯度有关，也可能受到中间特征、注意力状态以及不同层的信息传递方式影响。

不过，这些研究路线并不是严格的替代关系。新的方法并不意味着此前的方法已经失效，它们是在不同条件和目标下探索新的技术方案。

---

# 十、ViT Attack 主要技术路线对照表

下面对前面介绍的代表性工作进行集中比较。

|时间|代表性工作|核心研究问题|主要技术思路|
|---|---|---|---|
|2021|On the Adversarial Robustness of Vision Transformers|ViT 是否比 CNN 更鲁棒？|比较不同架构的对抗鲁棒性|
|2021|On the Robustness of Vision Transformers to Adversarial Examples|ViT 的攻击效果与跨架构迁移能力如何？|传统攻击评估、跨架构分析、SABG|
|2022|PNA|如何利用 ViT 的注意力结构改善迁移性？|跳过注意力相关的部分梯度传播|
|2022|PatchOut|是否需要在每次迭代中使用全部 Patch？|随机选择部分 Patch 进行优化|
|2023|TGR|如何利用 ViT 的中间层 Token 梯度？|Token 级梯度正则化|
|2023|MIG|如何改善 ViT 与 CNN 之间的迁移性？|积分梯度与动量结合|
|2023|Inductive Bias Attack|如何利用不同架构的共同与独有特征偏置？|特征偏置与频率信息分析|
|2024|VDC|如何利用 ViT 不同层之间的信息？|虚拟密集连接|
|2024|FBTA|如何同时改善前向和反向过程？|DTA 与 BGC|
|2025|FPR|如何通过前向传播优化改善迁移性？|注意力图多样化与 Token 嵌入精细化|

需要强调，表中列出的是具有代表性的研究，并非对所有 ViT Attack 论文的穷尽性统计。部分论文的预印本年份与正式发表年份不同，因此这里主要按照其正式发表或研究公开阶段进行梳理。

---

# 十一、从研究方法看，ViT Attack 领域经历了哪些变化？

如果你想真正理解这个领域，而不仅仅是记住论文名称，最值得关注的是下面四个变化。

## 11.1 从直接使用输入梯度，到研究梯度的来源

传统梯度攻击通常关注：

$$
[ \nabla_x L(x,y) ]
$$

也就是损失函数对输入图像的梯度。

但 ViT 的计算过程包含多层 Transformer Block、多个 Token 和注意力机制。

因此，研究人员开始进一步分析：

- 哪些计算路径对梯度产生了重要影响？
- 不同 Token 的梯度是否具有不同特点？
- 不同层的梯度为什么会产生波动？
- 是否可以通过修改梯度传播过程改善迁移性？

PNA 和 TGR 分别从注意力梯度路径与 Token 梯度的角度切入。

## 11.2 从固定输入优化，到改变输入区域与内部状态

PatchOut 表明，攻击过程不一定需要在每次迭代中以相同方式使用全部 Patch。

DTA 则通过 Dropout 增加中间计算状态的多样性。

这两种方法并不相同，但都体现出一种思路：

攻击不一定只能依靠固定的输入和固定的内部计算状态来完成。

通过改变输入区域的选择方式，或者改变模型内部的计算状态，可能改善生成的对抗样本对不同模型的适应能力。

## 11.3 从单模型内部优化，到研究不同模型之间的共同性

MIG 与 Inductive Bias Attack 都关注跨模型迁移问题，但采取了不同的研究路径。

MIG 关注不同模型的积分梯度之间的相似性。

Inductive Bias Attack 则进一步研究不同架构在特征偏置和频率信息利用方面的差异。

这意味着研究人员开始从更广泛的角度理解迁移性：

攻击效果不仅与某一个模型的局部优化有关，还与不同模型的特征学习方式是否具有共同点有关。

## 11.4 从单一模块优化，到前向与反向过程的协同设计

FBTA 和 FPR 代表了这种研究思路的进一步发展。

研究人员开始同时考虑：

- 前向传播如何形成中间表示。
- 注意力图如何影响特征交互。
- Token 嵌入如何影响后续计算。
- 反向传播如何形成输入梯度。
- 中间层梯度如何影响最终扰动。

这使 ViT Attack 的研究从简单的输入梯度优化，扩展到对模型内部计算过程的综合研究。

---

# 十二、ViT Attack 领域还有哪些值得研究的问题？

截至 2026 年 10 月，上述论文能够勾勒出 ViT Attack 的一条重要发展主线，但它们不能代表这个领域的全部研究。

从这些工作的研究思路出发，还可以总结出几个值得关注的问题。

### 12.1 不同 ViT 架构之间的迁移性

ViT 并不是单一模型。

不同模型可能采用不同的：

- Patch 大小。
- Token 处理方式。
- 注意力结构。
- 网络深度。
- 特征聚合机制。

因此，一个攻击方法在某种 ViT 上表现良好，并不意味着它对其他 ViT 同样有效。

研究如何减少对特定架构的依赖，仍然是重要的问题。

### 12.2 ViT 与 CNN 之间的迁移性

ViT 与 CNN 的结构差异，为跨架构迁移攻击带来了挑战。

现有研究尝试通过梯度调整、积分梯度、归纳偏置分析和前向传播优化等方式改善迁移性。

但不同数据集、模型和攻击预算下的结果可能不同，因此仍需要更系统的实验分析。

### 12.3 攻击效果与计算成本之间的平衡

某些方法需要计算多次前向传播、分析中间层特征，或者额外处理不同 Token 的梯度。

这些操作可能提高攻击效果，但也可能增加计算成本。

因此，研究人员需要同时考察：

- 攻击成功率。
- 扰动大小。
- 计算时间。
- 显存消耗。
- 不同模型之间的迁移效果。

仅仅比较攻击成功率，并不足以全面评价一种方法。

### 12.4 不同视觉任务中的适用性

ViT 不仅用于图像分类，也广泛应用于目标检测、语义分割和视觉语言模型等任务。

但是，分类任务中的攻击方法不一定能够直接适用于其他任务。

例如，目标检测模型需要预测目标位置与类别，而语义分割模型需要为不同像素预测标签。

因此，研究对象从图像分类扩展到其他视觉任务时，需要重新考虑攻击目标、损失函数和评价指标。

### 12.5 从攻击研究走向系统性的鲁棒性评估

ViT Attack 的研究价值不仅在于提出新的攻击算法，也在于帮助研究人员更准确地评估模型的安全性。

通过研究不同攻击方法的有效性，可以发现模型可能存在的漏洞，并为后续防御策略的设计提供依据。

攻击效果与防御效果需要通过严谨的实验分别评价，不能仅凭模型的结构特点判断其安全性。

---

# 十三、总结：如何理解 ViT Attack 的发展历程？

将前面的内容压缩成一条主线，可以得到以下结论。

**第一阶段：2021 年，研究人员开始系统研究 ViT 的对抗鲁棒性。**

主要问题是 ViT 是否容易受到传统对抗攻击，以及 ViT 与 CNN 在鲁棒性和迁移性方面存在什么差异。

**第二阶段：2021—2022 年，研究开始利用 ViT 的结构特点设计攻击。**

PNA 和 PatchOut 分别从注意力梯度传播与 Patch 选择的角度改善对抗样本的迁移性。

**第三阶段：2023 年，研究进一步深入 Token 梯度和梯度估计机制。**

TGR 关注中间层 Token 梯度的正则化，MIG 则通过积分梯度与动量机制改善跨模型迁移能力。

**第四阶段：2023—2024 年，研究更加重视不同架构之间的特征联系，以及 ViT 内部不同层的信息利用。**

Inductive Bias Attack 从架构的归纳偏置入手，VDC 则关注层间信息传递。

**第五阶段：2024—2025 年，研究开始更加系统地优化前向与反向计算过程。**

FBTA 同时考虑前向状态多样化与反向梯度调节，FPR 则进一步研究注意力图和 Token 嵌入的前向传播优化。

因此，ViT Attack 的发展可以概括为：

$$
[ \boxed{ \begin{aligned} &\text{对抗鲁棒性评估}\\ &\quad\downarrow\\ &\text{利用 ViT 架构特点设计攻击}\\ &\quad\downarrow\\ &\text{Token 级梯度与梯度估计优化}\\ &\quad\downarrow\\ &\text{跨架构特征与归纳偏置分析}\\ &\quad\downarrow\\ &\text{前向与反向传播协同优化} \end{aligned} } ]
$$

这条发展路线体现了研究问题的不断深入，但并不意味着后一个阶段完全取代前一个阶段。不同技术路线仍然可以相互结合，也可能在不同实验条件下表现出不同的优势。

**最重要的认识是：ViT Attack 并不是单纯把传统攻击算法应用于 Transformer，而是逐渐形成了围绕注意力机制、Patch、Token、梯度传播、特征偏置和跨模型迁移性的专门研究体系。**