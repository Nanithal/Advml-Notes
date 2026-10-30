---
title: "论文 - Data-Free Universal Adversarial Perturbation and Black-Box Attack"
aliases:
  - "Data-free Universal Adversarial Perturbation and Black-box Attack论文"
tags:
  - "知识库/论文阅读"
  - "主题/通用与稀疏扰动"
笔记类型: "论文阅读"
分类: "通用与稀疏扰动"
状态: "已整理"
摘要: "梳理主导类别、Cosine-UAP、Jigsaw 图像和重复纹理构造下的无数据通用攻击。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/通用与稀疏扰动/总结 - 通用与稀疏扰动]]"
相关笔记:
  - "[[30_对抗机器学习/通用与稀疏扰动/论文 - Universal Adversarial Perturbations]]"
  - "[[30_对抗机器学习/通用与稀疏扰动/论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations]]"
  - "[[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解]]"
论文条目:
  - "[[论文来源目录#datafree]]"
论文标题:
  - "Data-Free Universal Adversarial Perturbation and Black-Box Attack"
发表载体:
  - "ICCV"
发表年份:
  - "2021"
发表类型:
  - "会议论文"
发表链接:
  - "https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.html"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/通用与稀疏扰动/总结 - 通用与稀疏扰动|通用与稀疏扰动总结]]

# 论文 - Data-Free Universal Adversarial Perturbation and Black-Box Attack

## 论文基本信息

- **论文名称**：Data-free Universal Adversarial Perturbation and Black-box Attack（无数据通用对抗扰动与黑盒攻击）
- **作者**：Chaoning Zhang、Philipp Benz、Adil Karjauv、In So Kweon
- **会议**：ICCV 2021
- **研究方向**：对抗攻击、通用对抗扰动（UAP）、无数据攻击、黑盒攻击
- **原文**：用户上传的论文 PDF（共 10 页）

## 一、核心思想

这篇论文主要研究一个问题：

> **在没有原始训练数据，甚至无法访问目标模型的情况下，能否生成有效的通用对抗扰动（Universal Adversarial Perturbation，UAP）？**

通用对抗扰动是指只需生成一个固定扰动 $\nu$，就可以使大量不同图像被神经网络错误分类，而不需要针对每张图片单独优化。

论文围绕三个方向展开：

1. **机制解释**：为什么非定向 UAP 会让大量不同图片被预测为同一个类别（Dominant Label）？
2. **无数据 UAP 生成**：如何从少量训练样本逐步过渡到完全不使用真实数据，仍生成高攻击成功率的 UAP？
3. **无数据黑盒攻击**：在没有训练数据、无法访问模型参数甚至不查询目标模型的情况下，能否使用简单扰动实现有效攻击？

**核心结论**：高效的通用对抗扰动不一定需要真实训练数据，也不一定需要复杂优化。具有特定频率特征的人工图像和简单的重复纹理，就能引发神经网络明显的预测错误。

## 二、研究背景与动机

传统对抗攻击通常针对每张输入图像分别生成扰动：

$$
x_{adv}=x+\delta_x
$$

其中，$\delta_x$ 是针对图像 $x$ 优化的扰动。

UAP 则使用一个与具体图像无关的扰动：

$$
x_{adv}=x+\nu
$$

其优化目标为：

$$
\max_{\nu}\Pr_{x\sim X}\left[\hat{k}(x+\nu)\neq\hat{k}(x)\right]
$$

$$
\text{s.t.}\quad\|\nu\|_p\leq\epsilon
$$

其中，$\nu$ 为通用扰动，$\epsilon$ 为扰动幅度限制。

UAP 的优势是**只需提前生成一次扰动，就可以用于攻击大量不同图像**，具有较高攻击效率。但传统方法往往依赖大量真实训练图像、样本标签或目标模型信息。作者希望逐步降低这些依赖。

## 三、论文的三个核心贡献

### 3.1 解释 Dominant Label（主导类别）现象

在非定向 UAP 攻击中，虽然没有预先指定目标类别，但大量图片会被错误分类到同一类别。例如，在 VGG16 上，作者报告：

- UAP 的总体 **Fooling Ratio 达到 97.4%**；
- **90.1% 的图像**被分类为 **“brain coral”（脑珊瑚）**。

先前一种解释认为，主导类别可能占据较大的输入空间区域。但作者发现：

- 相同条件下的随机噪声不会使大量图像转变为这个类别；
- 更换用于训练 UAP 的样本类别，主导类别也可能随之改变。

因此，仅凭“某类的决策区域更大”难以充分解释这个现象。

作者提出的替代解释是：**UAP 对网络的深层特征和最终输出有很强的主导作用，可能盖过原始图像自身的作用。**

具体比较以下三种输入的模型响应：

$$
k(x),\quad k(\nu),\quad k(x+\nu)
$$

其中 $k(\cdot)$ 表示网络输出的 Logit 向量。实验发现，单独输入 UAP 时的 Logit 分布，与加入 UAP 后对抗图像的平均 Logit 分布非常相似。

逐层和逐步分析显示：

- 浅层网络更容易保留原始图像的响应；
- 在多数层，特别是深层网络，扰动与对抗图像的特征响应具有更高相似性；
- 随着 UAP 优化，扰动响应与对抗图像响应的相似度上升，并与 Fooling Ratio 呈正相关。

**理解**：非定向 UAP 虽不具有预先指定的目标类别，但优化过程中可能自然产生一种“伪目标类别”，从而造成 Dominant Label 现象。这属于基于实验观察的解释，并非严格的因果证明。

*原文对应：第 4 节，Figure 1、Figure 2，论文第 4–5 页。*

### 3.2 Cosine-UAP 与 Jigsaw 无真实数据攻击

这是论文最核心的方法贡献。作者将减少数据依赖分为三个阶段。

#### 阶段一：不需要真实标签的 Cosine-UAP

传统方法常使用交叉熵损失，需要真实标签。作者使用自监督的余弦相似度目标：

$$
\boxed{\mathcal L=\operatorname{CosSim}\left(k(x),k(x+\nu)\right)}
$$

通过**最小化**这个目标，让原始图像和对抗图像的 Logit 向量尽量不相似，从而增加模型预测变化的概率。

方法特点：

- 不需要真实类别标签；
- 直接优化 UAP 扰动 $\nu$，无需训练额外生成器；
- 在论文设置下，具有较好的计算效率与 Fooling Ratio。

#### 阶段二：使用极少量真实训练图像

少样本条件下 UAP 容易过拟合。作者使用随机裁剪、旋转和翻转等增强方式，提高扰动的泛化能力。

只使用 **32 张训练图像**时，结果为：

| 模型 | Fooling Ratio |
|---|---:|
| VGG16 | 93.5% |
| VGG19 | 93.5% |
| ResNet50 | 91.8% |

*原文对应：Table 4，论文第 6 页。*

#### 阶段三：完全不使用真实训练数据（Jigsaw Images）

作者提出以人工构造的 **Jigsaw Images（拼图式图像）**代替真实训练图像来优化 UAP。

其设计遵循两个原则：

1. **局部平滑性（Locally Smooth）**：模拟自然图像局部区域的平滑特征；
2. **混合频率（Mixed Frequency）**：使用多种大小和变化频率的图像块，增加训练输入的多样性。

实验表明，可变频率 Jigsaw 比固定频率 Jigsaw、均匀随机噪声、高斯噪声等人工输入更有效。

**关键区别**：Jigsaw 图像是用于**训练 UAP 的人工样本**，而不是直接添加到测试图像上的扰动。此方法虽然不依赖真实训练数据，但**仍需访问目标模型以计算梯度**，属于数据自由的白盒 UAP 生成。

*原文对应：第 5 节、Figure 3、Table 5 和 Table 6，论文第 5–6 页。*

### 3.3 简单重复纹理实现无数据黑盒攻击

作者进一步研究更严格的攻击条件：

- 不需要目标模型的训练数据；
- 不需要目标模型参数或梯度；
- 不需要查询目标模型；
- 不训练替代模型；
- 不对扰动进行迭代优化。

作者测试了三类简单重复纹理：

1. 水平条纹（Horizontal Pattern）；
2. 垂直条纹（Vertical Pattern）；
3. 棋盘格（Checkerboard Pattern）。

实验显示，**棋盘格纹理在这些简单模式中效果最好**。作者还尝试使用傅里叶变换（FT）或奇异值分解（SVD）削弱原图像的部分高频内容，再施加通用纹理，从而进一步提升攻击效果。

与上一节不同，这里**重复纹理本身就是攻击扰动**，且生成过程不依赖数据或模型优化。

*原文对应：第 6 节、Table 9 和 Table 10，论文第 7 页。*

## 四、主要实验结果

### 4.1 无数据 UAP 的 Fooling Ratio 对比

论文 Table 5 报告了 ImageNet 上五种分类模型的平均 Fooling Ratio，数值越高表示攻击越有效。

| 方法 | 平均 Fooling Ratio | 是否使用真实训练数据 |
|---|---:|---|
| FFF | 51.27% | 否 |
| GD-UAP（无 Prior） | 51.59% | 否 |
| GD-UAP（有 Prior） | 64.40% | 否（使用先验） |
| AAA | 73.59% | 否 |
| **本文 Jigsaw Images** | **84.08%** | **否** |
| DeepFool-UAP | 82.46% | 是 |

本文 Jigsaw 方法的平均 Fooling Ratio 为 **84.08%**，比 AAA 高 **10.49 个百分点**，并略高于作为对照的有数据 DeepFool-UAP。注意不同论文方法设置不一定完全一致，这些数字反映的是原文报告的基准结果。

### 4.2 黑盒攻击后的分类准确率

论文 Table 9 采用攻击后分类准确率评价攻击，**越低越好**。对应实验使用 $\epsilon=0.1$。

| 方法 | 攻击后平均分类准确率 |
|---|---:|
| Beyonder（使用完整训练集） | 36.2% |
| Prototypical* | 31.2% |
| 本文 Checkerboard | 25.3% |
| **本文 Checkerboard + FT** | **20.8%** |
| 本文 Checkerboard + SVD | 22.7% |

Checkerboard + FT 将八种分类模型的平均分类准确率降至 **20.8%**，优于论文比较的 Prototypical* 方法。

### 4.3 目标检测和语义分割的攻击效果

| 任务 | 模型 | 攻击前 | 攻击后 | 指标 |
|---|---|---:|---:|---|
| 目标检测 | Faster R-CNN | 37.0 | 18.3 | AP |
| 目标检测 | Mask R-CNN | 37.9 | 18.0 | AP |
| 语义分割 | FCN ResNet101 | 63.7 | 26.9 | mIoU |
| 语义分割 | DeepLabV3 ResNet101 | 67.4 | 20.3 | mIoU |

这些结果表明，简单的周期性纹理不仅能破坏分类预测，也能使目标检测和语义分割模型的性能显著下降。

*原文对应：Table 12、Table 13 及 Figure 5、Figure 6，论文第 8 页。*

## 五、论文的创新性与不足

### 创新性

1. **机制角度**：通过模型特征响应研究 Dominant Label 现象，而不只从输入空间决策区域几何结构解释。
2. **方法角度**：提出无需真实标签的 Cosine-UAP，并使用人工 Jigsaw 图像实现严格无真实训练数据的 UAP 优化。
3. **攻击条件角度**：展示在无需数据、模型查询和优化的条件下，简单重复纹理也可以形成有效的通用攻击。

### 局限性

1. 主导类别的解释主要依赖特征相似性和相关性实验，缺少严格的理论证明。
2. Jigsaw 图像的设计较为简单，未系统探索最优人工图像生成方式。
3. Jigsaw 方案仍然需要模型梯度；真正无模型、无查询的是后面的重复纹理攻击，二者不能混为一谈。
4. 不同部分采用不同扰动预算。例如，前文 UAP 默认使用 $\ell_\infty=10/255$，Table 9 黑盒攻击采用 $\epsilon=0.1$，不宜直接进行数值比较。
5. 论文验证了多种网络、输入尺寸和部分防御设置，但不能由此推断对所有部署模型都具备同样效果。

## 六、对后续研究的启发

这篇论文值得关注的方向是**频率特征、重复纹理与通用对抗扰动之间的关系**。

- **人工数据设计**：当真实训练数据不可用时，可以尝试模拟自然图像的局部统计特征和频率分布，构建用于优化攻击的代理输入。
- **特征主导性分析**：除统计攻击成功率外，还可以分析扰动与原图在不同网络层的相对影响。
- **扰动泛化**：进一步从频率、空间周期性、输入尺寸和跨模型迁移角度研究 UAP。
- **鲁棒性评估**：评测模型时可以考虑周期性纹理等不经过梯度优化的简单模式，而不仅仅测试传统优化型对抗样本。

## 七、一句话总结

**这篇 ICCV 2021 论文通过分析通用扰动对深层特征的主导作用，提出利用余弦相似度和人工 Jigsaw 图像生成无真实数据的通用对抗扰动，并进一步证明简单的棋盘格等重复纹理也能够在无需数据、模型查询或优化的情况下，对分类、检测和分割网络产生显著攻击效果。**

从研究价值来看，最值得借鉴的不仅是具体攻击算法，更是其指出的方向：**神经网络对某些空间频率和重复纹理的敏感性，可以成为通用对抗攻击的重要来源。**

---

*本文基于用户提供的 ICCV 2021 原论文整理；数字来自对应论文表格，部分“启发”和“局限性”属于对原文结果的归纳与评价。*

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/通用与稀疏扰动/论文 - Universal Adversarial Perturbations|论文 - Universal Adversarial Perturbations]]：先回顾共享扰动与逐样本攻击的区别。
- [[30_对抗机器学习/通用与稀疏扰动/论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations|论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations]]：比较两篇无数据通用扰动工作的目标和数据假设。
- [[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解|Threat Model - 威胁模型详解]]：明确不使用真实数据与不可访问目标模型的不同限制。

## 论文出处

- [Data-Free Universal Adversarial Perturbation and Black-Box Attack](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Data-Free_Universal_Adversarial_Perturbation_and_Black-Box_Attack_ICCV_2021_paper.html)：ICCV，2021，会议论文。

<!-- knowledge-links:end -->
