---
title: "ViT Attack - 文献资料与研究问题"
aliases:
  - "ViT Atttack领域文献"
分类: "ViT与基础模型攻击"
笔记类型: "研究资料"
整理日期: 2026-10-27
---

[[首页|首页]] · [[30_对抗机器学习/ViT与基础模型攻击/索引 - ViT与基础模型攻击|ViT与基础模型攻击索引]]

# ViT Attack - 文献资料与研究问题

# Vision Transformer 对抗攻击（ViT Attack）领域文献综述与研究脉络

文献梳理范围：2020—2026 年 10 月｜重点：对抗样本生成、攻击迁移性、ViT 结构特性与关键开创性工作

ViT Attack 领域的研究可以沿着一条比较清晰的主线理解：

从“ViT 是否容易受到对抗攻击”，发展到“如何利用 ViT 特有结构进行攻击”，再发展到“如何提高对抗样本在不同 ViT、CNN 乃至其他视觉模型之间的迁移能力”。

如果准备以此为研究方向，我建议不要仅按照论文发表年份梳理，而是同时建立两套分类体系：

1. 攻击范式分类：白盒攻击、迁移式黑盒攻击、查询式黑盒攻击。
    
2. 技术创新分类：Patch/Token 攻击、Attention 攻击、梯度优化、特征攻击、模型结构优化、输入变换、跨架构迁移、下游任务攻击等。
    

这样既能说明领域发展的历史，也能更清楚地找到尚未解决的问题。

## 一、首先明确：这个领域的奠基性工作有哪些？

下面是我认为首先应该掌握的代表性论文。这里的“开创性”主要根据研究问题、首次提出的机制及后续技术路线来判断，并不简单等同于引用量排名。

|论文|年份 / 会议|关键地位|
|---|---|---|
|An Image is Worth 16x16 Words|ICLR 2021|ViT 基础架构，整个方向的模型起点|
|On the Robustness of Vision Transformers to Adversarial Examples|ICCV 2021|早期 ViT 对抗鲁棒性系统研究，重要奠基论文|
|Towards Transferable Adversarial Attacks on Vision Transformers|AAAI 2022|ViT 专用迁移攻击的重要开创性工作，提出 PNA、PatchOut|
|Adversarial Token Attacks on Vision Transformers|2021 年预印本|早期 Token/块稀疏攻击代表作|
|Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?|ICLR 2022|Patch 级攻击的重要代表作|
|Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization|CVPR 2023|ViT 内部梯度正则化方向的关键工作，提出 TGR|
|Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients|ICCV 2023|跨 ViT/CNN 迁移与积分梯度路线的重要工作|
|Improving the Adversarial Transferability of Vision Transformers with Virtual Dense Connection|AAAI 2024|通过反向传播路径改造提高迁移性的重要工作|
|Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement|CVPR 2025|从反向梯度优化扩展到前向传播优化的关键工作|

前三篇尤其值得优先阅读。其中，Mahmood 等人的 ICCV 2021 工作在 2021 年 3 月已有预印本；Wei 等人的 PNA/PatchOut 工作在 2021 年 9 月已有预印本，2022 年正式发表于 AAAI。它们研究的问题不同，不能简单把两者都称为“第一篇 ViT 攻击论文”。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

## 二、ViT Attack 领域的完整分类框架

## ViT Adversarial Attacks

以攻击者掌握的信息为一级分类，以攻击机制为二级分类

白盒攻击

像素梯度优化

Patch / Token

Attention

位置编码

特征与损失函数

迁移式黑盒

梯度正则化

反向传播改造

前向传播改造

输入变换

特征 / 模型对齐

查询式黑盒

Score-based

Decision-based

Patch 搜索

查询效率优化

交叉研究维度

ViT → ViT

ViT → CNN

CNN → ViT

ViT → 检测/分割

ViT → CLIP/VLM

数字 → 物理攻击

需要特别指出：这不是说每篇论文只能属于一个类别。 例如 PNA 属于迁移式黑盒攻击，技术上又属于反向传播梯度改造；PatchOut 同样属于迁移攻击，但它利用的是输入 Patch 子集随机化。

另外，Patch 攻击与传统小扰动攻击需要区别对待。前者可能集中修改一个局部区域，后者一般要求所有像素的扰动满足 \(L_p\) 范数约束，二者不能只根据攻击成功率直接比较。

## 三、ViT Attack 的研究发展史：从 2021 到 2025

在进一步核对早期论文的预印本时间和正式发表信息后，有一个非常重要的补充：ICLR 2022 的 Naseer 等人工作同样是 ViT 迁移攻击的早期开创性论文，而且其预印本比 PNA/PatchOut 更早。

因此，研究 ViT 迁移攻击，不能只从 AAAI 2022 的 PNA 开始。

### 第一阶段：2020—2021 年，ViT 对抗鲁棒性问题的建立

这一阶段研究者主要关心三个问题：

- ViT 在对抗样本面前是否比 CNN 更鲁棒？
    
- CNN 和 ViT 之间为什么存在较弱的对抗迁移性？
    
- ViT 的 Patch 划分和 Self-Attention 是否带来了新的攻击面？
    

|论文|最早版本 / 正式发表|贡献|
|---|---|---|
|[An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929?utm_source=chatgpt.com)|2020.10 / ICLR 2021|引入 ViT 架构|
|[Understanding Robustness of Transformers for Image Classification](https://openaccess.thecvf.com/content/ICCV2021/html/Bhojanapalli_Understanding_Robustness_of_Transformers_for_Image_Classification_ICCV_2021_paper.html?utm_source=chatgpt.com)|2021.03 / ICCV 2021|早期系统性鲁棒性研究，包括输入与模型扰动|
|[On the Robustness of Vision Transformers to Adversarial Examples](https://arxiv.org/abs/2104.02610?utm_source=chatgpt.com)|2021.03 / ICCV 2021|研究白盒攻击、黑盒攻击及 CNN–ViT 迁移性|
|[Reveal of Vision Transformers Robustness against Adversarial Attacks](https://arxiv.org/abs/2106.03734?utm_source=chatgpt.com)|2021.06 / 预印本|对多种 ViT 及混合架构进行攻击评估|
|[Are Transformers More Robust Than CNNs?](https://proceedings.neurips.cc/paper/2021/hash/e19347e1c3ca0c0b97de5fb3b690855a-Abstract.html?utm_source=chatgpt.com)|NeurIPS 2021|通过公平训练设置重新比较 CNN/ViT 鲁棒性|

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+3

这里特别推荐 Are Transformers More Robust Than CNNs?。

早期的一些实验发现 ViT 比 CNN 更鲁棒，但这篇论文指出，结论受到模型规模和训练策略等因素影响。在尽量统一训练设置后，CNN 与 ViT 的对抗鲁棒性差距并没有先前认为的那么明显。

这一结论非常重要：模型结构不同，不意味着对抗鲁棒性天然更强。

### 第二阶段：2021—2022 年，ViT 专用对抗攻击方法出现

这是最需要认真研究的阶段，因为它产生了好几条后来不断被扩展的技术路线。

2021.06

ICLR 2022

Self-Ensemble + Token Refinement（Naseer et al.）

[On Improving Adversarial Transferability of Vision Transformers](https://arxiv.org/abs/2106.04169?utm_source=chatgpt.com)

发现 ViT 不同深度的 Class Token 都包含判别信息，不应只利用最后一层分类输出生成扰动。通过层级自集成与 Token 精炼增强迁移攻击。

开创性定位：中间层 Class Token 利用、单模型内部自集成攻击的早期代表。

2021.09

AAAI 2022

PNA + PatchOut（Wei et al.）

[Towards Transferable Adversarial Attacks on Vision Transformers](https://arxiv.org/abs/2109.04176?utm_source=chatgpt.com)

利用注意力梯度处理和随机 Patch 子集优化，提高攻击跨 ViT、跨 CNN 的迁移能力。

开创性定位：面向 ViT 结构设计迁移攻击、Attention 梯度改造的重要代表。

2021.10

预印本

Adversarial Token Attack（Joshi et al.）

[Adversarial Token Attacks on Vision Transformers](https://arxiv.org/abs/2110.04337?utm_source=chatgpt.com)

利用块稀疏性设计 Token 攻击，分析 ViT 对局部 Token 扰动的敏感性。

开创性定位：早期 Token 级、块稀疏攻击探索。

2021.11

ECCV 2022

Patch Perturbation（Gu et al.）

[Are Vision Transformers Robust to Patch Perturbations?](https://arxiv.org/abs/2111.10659?utm_source=chatgpt.com)

研究注意力机制在自然 Patch 损坏与对抗 Patch 扰动下的不同表现，揭示注意力可能被恶意 Patch 误导。

关键定位：Patch 扰动机制与鲁棒性解释研究。

2021.12

NeurIPS 2022

PAR（Shi et al.）

[Decision-Based Black-Box Attack Against Vision Transformers via Patch-Wise Adversarial Removal](https://papers.neurips.cc/paper_files/paper/2022/hash/544696ef4847c903376ed6ec58f3a703-Abstract-Conference.html?utm_source=chatgpt.com)

通过粗到细的 Patch 搜索压缩扰动，在只获得分类决策的条件下提高查询效率。

开创性定位：专门针对 ViT 的决策式黑盒 Patch 搜索攻击的重要早期代表。

2022.03

ICLR 2022

Patch-Fool（Fu et al.）

[Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?](https://arxiv.org/abs/2203.08392?utm_source=chatgpt.com)

通过集中扰动少量 Patch 并影响自注意力行为，说明 ViT 在局部对抗扰动下可以非常脆弱。

关键定位：ViT 对抗 Patch 攻击路线中的经典论文。

这些工作的时间线已通过预印本或会议记录进行核对。这里的“第一”指特定研究问题的早期代表，不能理解为已证明该论文具有绝对的文献首创优先权。

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

+6

### 第三阶段：2023—2024 年，迁移攻击精细化

这一阶段的研究重点开始明显转移到：

- 如何减少对代理模型的过拟合？
    
- 如何从中间层获取更稳定的对抗梯度？
    
- 如何让 ViT 生成的扰动对 CNN 同样有效？
    
- 能否通过频域、特征和传播路径改善迁移性？
    

这一时期主要形成了 TGR、MIG、VDC、GNS-HFA、ATT 等技术路线。

### 第四阶段：2025 年以后，前向结构、集成、自监督与跨任务迁移

比较有代表性的变化是：

攻击优化对象从输入梯度，逐渐扩展到了注意力矩阵、Token 表征、代理模型前向传播过程，以及预训练视觉基础模型的共享特征。

这使得 ViT Attack 与表示学习、自监督学习、模型集成和下游任务安全的联系越来越紧密。

## 四、按技术路线梳理关键论文

这是最适合用于撰写 Related Work 和寻找研究创新点的分类方式。

### 4.1 基于 Patch / Token 的攻击

核心思想：利用 ViT 将图像拆分为 Patch Token 的特性，在局部空间或 Token 表征上制造扰动。

|方法|论文 / 年份|核心创新|
|---|---|---|
|Adversarial Token Attack|Joshi et al., 2021|块稀疏 Token 扰动|
|Patch-Fool|Fu et al., ICLR 2022|联合局部 Patch 修改与注意力干扰|
|Patch Perturbation|Gu et al., ECCV 2022|分析 Patch 对抗敏感性|
|PatchOut|Wei et al., AAAI 2022|随机选择 Patch 子集优化，增强迁移性|
|PAR|Shi et al., NeurIPS 2022|黑盒 Patch 级噪声移除与粗细搜索|
|ATT|Ming et al., NeurIPS 2024|自适应 Token 调整与渐进式 PatchOut|

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

+5

这里需要区分两类工作：

- Patch 局部攻击：真正把修改集中到一个或少数空间区域，例如 Patch-Fool。
    
- Patch 采样式迁移攻击：利用随机 Patch 子集控制每轮优化过程，例如 PatchOut。
    

两者虽然都涉及 Patch，但技术目标并不相同。

### 4.2 基于 Attention / Embedding 的攻击

这条路线更加体现 ViT 的模型特性。

|方法|论文 / 年份|核心创新|
|---|---|---|
|PNA|AAAI 2022|截断注意力矩阵相关的梯度传播|
|ATA|ACM MM 2022|联合攻击注意力不确定性与敏感 Embedding|
|GALD|Multimedia Systems, 2023|全局注意力引导与局部 Patch Drop|
|AAS|CVPR 2024|调节 Attention Softmax 前的缩放，缓解梯度掩蔽|
|FPR|CVPR 2025|注意力图多样化与 Token Embedding 动量优化|

其中两篇值得特别关注：

ATA — Generating Transferable Adversarial Examples against Vision Transformers

ACM MM 2022，Wang 等人提出 Architecture-oriented Transferable Attacking，通过扰乱不确定注意力和敏感的 Patch Embedding 增强不同 ViT 间的迁移性。

[论文原文](https://doi.org/10.1145/3503161.3547989?utm_source=chatgpt.com) · [GitHub 代码](https://github.com/nlsde-safety-team/ATA?utm_source=chatgpt.com)

AAS — Towards Understanding and Improving Adversarial Robustness of Vision Transformers

CVPR 2024，Jain 等人指出 ViT 的 Attention Softmax 可能导致数值下溢和梯度掩蔽，使常规白盒攻击低估模型的脆弱程度。AAS 通过自适应缩放改进攻击优化。

[论文原文](https://openaccess.thecvf.com/content/CVPR2024/html/Jain_Towards_Understanding_and_Improving_Adversarial_Robustness_of_Vision_Transformers_CVPR_2024_paper.html?utm_source=chatgpt.com)

后者尤其重要，因为它提醒研究者：攻击成功率低，不一定说明模型真的鲁棒，也可能是梯度质量存在问题。

![](https://www.google.com/s2/favicons?domain=https://xhplus.github.io&sz=32)

XHPlus

+2

### 4.3 基于梯度优化与反向传播改造的迁移攻击

这是我认为 ViT Attack 中已经形成较明确技术继承关系的一条主线。

ViT 梯度优化技术演进

PNA — AAAI 2022

抑制模型特定的 Attention 梯度影响

TGR — CVPR 2023

Token 粒度的中间层梯度正则化

VDC — AAAI 2024

虚拟密集连接，改善梯度传播路径

GNS-HFA — ICLR 2024

梯度归一化缩放与高频自适应

ATT — NeurIPS 2024

自适应梯度缩放、Token 截断与 Patch 调节

此图表示研究问题和机制的演进，不表示各论文之间都存在直接引用或严格继承关系。

这一方向可以分成四种具体思想。

|子方向|方法|研究对象|
|---|---|---|
|梯度截断|PNA|Attention 梯度通路|
|梯度正则化|TGR、GNS-HFA、ATT|Token、中间层 QKV/Attention/MLP 梯度|
|梯度路径重构|VDC|Transformer Block 间的反向传播路径|
|积分梯度优化|MIG|图像积分梯度及其迭代动量|

论文链接：

- [TGR — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Transferable_Adversarial_Attacks_on_Vision_Transformers_With_Token_Gradient_Regularization_CVPR_2023_paper.html?utm_source=chatgpt.com)
    
- [VDC — AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/28541?utm_source=chatgpt.com)
    
- [GNS-HFA — ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/e25198b6a75f74277ee3a2bd4165d9ef-Abstract-Conference.html?utm_source=chatgpt.com)
    
- [ATT — NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/hash/24f8dd1b8f154f1ee0d7a59e368eccf3-Abstract-Conference.html?utm_source=chatgpt.com)
    
- [MIG — ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.html?utm_source=chatgpt.com)
    

这些工作各自通过减少代理模型特有梯度的影响、改善梯度传播，或提取更加稳定的优化信息来提高迁移性。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+4

### 4.4 基于前向传播与代理模型改造的攻击

代表性论文是：

FPR — Improving Adversarial Transferability on Vision Transformers via Forward Propagation Refinement

Ren et al. · CVPR 2025

主要贡献：Attention Map Diversification（AMD）+ Momentum Token Embedding（MTE）。与直接修改反向传播梯度不同，FPR 通过调整代理 ViT 的前向计算过程来提高扰动的泛化性。

[论文](https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Improving_Adversarial_Transferability_on_Vision_Transformers_via_Forward_Propagation_Refinement_CVPR_2025_paper.html?utm_source=chatgpt.com)

·

[代码](https://github.com/RYC-98/FPR?utm_source=chatgpt.com)

这篇论文具有明显的方法论价值：

早期的 PNA、TGR 和 VDC 主要从反向传播寻找优化空间，而 FPR 进一步研究前向计算中的 Attention Map 和 Token Embedding。

不过，前向/反向改造并不是一个完全互斥的划分，因为更改前向计算通常也会改变最后的输入梯度。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

+1

### 4.5 基于中间特征与表示学习的攻击

这个类别与前面的梯度优化方向不同：不一定把分类标签损失当作主要攻击目标，而是直接扰乱模型内部表征。

|方法|年份 / 会议|核心思想|
|---|---|---|
|Self-Ensemble + Token Refinement|ICLR 2022|利用多层 Class Token 的判别信息|
|FDAP — Feature Diversity Adversarial Perturbation|AAAI 2024|无标签特征攻击，利用 ViT 特征坍缩现象|
|dSVA|ICCV 2025|利用自监督 ViT 的对比学习与掩码建模特征生成迁移扰动|
|APFD — Attention-Preserving Feature Distortion|ECAI 2025 论文集，2026 年在线发表|关注 Attention 保持条件下的特征扰乱|

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

+3

其中我尤其建议研究两篇。

FDAP — Attacking Transformers with Feature Diversity Adversarial Perturbation

AAAI 2024，研究者针对 ViT 的特征坍缩现象，设计无需分类标签的特征攻击方法。

[论文原文](https://ojs.aaai.org/index.php/AAAI/article/view/27947?utm_source=chatgpt.com)

dSVA — Boosting Generative Adversarial Transferability with Self-supervised Vision Transformer Features

ICCV 2025，将两种自监督表示结合起来：

- DINO 一类对比学习模型所提供的结构表征；
    
- MAE 一类掩码图像建模模型所提供的局部纹理表征。
    

再利用生成器生成具备跨架构迁移能力的对抗样本。

[论文原文](https://openaccess.thecvf.com/content/ICCV2025/html/Wu_Boosting_Generative_Adversarial_Transferability_with_Self-supervised_Vision_Transformer_Features_ICCV_2025_paper.html?utm_source=chatgpt.com) · [GitHub 代码](https://github.com/spencerwooo/dSVA?utm_source=chatgpt.com)

### 4.6 基于模型集成与跨架构迁移的攻击

值得单独列出的论文是：

ViT-EnsembleAttack — ICCV 2025

ViT-EnsembleAttack: Augmenting Ensemble Models for Stronger Adversarial Transferability in Vision Transformers

该工作针对 ViT 模型集成提出了代理模型增强策略，包括：

- Multi-head Dropping
    
- Attention Score Scaling
    
- MLP Feature Mixing
    

随后通过模型集成生成更具迁移性的对抗样本。作者将其定位为专门针对 ViT 的集成攻击早期工作之一。

[论文原文](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_ViT-EnsembleAttack_Augmenting_Ensemble_Models_for_Stronger_Adversarial_Transferability_in_Vision_ICCV_2025_paper.html?utm_source=chatgpt.com)

它代表着一个值得继续探索的方向：不是只设计单个更强的 ViT 代理模型，而是利用多个结构或多种代理变体，提高扰动对未知目标模型的适应性。

![](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com&sz=32)

openaccess.thecvf.com

### 4.7 查询式黑盒攻击与其他攻击目标

除了前面列出的迁移攻击，还有两类容易在综述中被遗漏的工作。

|攻击类别|代表论文|意义|
|---|---|---|
|决策式黑盒攻击|PAR，NeurIPS 2022|只依赖模型分类决策，利用 Patch 重要性优化查询|
|对抗计算开销攻击|SlowFormer，CVPR 2024|不一定改变预测类别，而是使高效 ViT 增加计算量和能耗|

SlowFormer 特别值得注意，它攻击的是具有自适应计算能力的高效 Vision Transformer，而不只是分类正确率。这意味着 ViT Attack 的目标可以扩展为模型计算资源、运行时延和能耗。

[SlowFormer — CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Navaneet_SlowFormer_Adversarial_Attack_on_Compute_and_Energy_Consumption_of_Efficient_CVPR_2024_paper.html?utm_source=chatgpt.com)

![](https://www.google.com/s2/favicons?domain=https://papers.neurips.cc&sz=32)

papers.neurips.cc

+1

## 五、技术上最重要的几种攻击思想是什么？

从算法研究角度看，建议重点理解下面这几个问题。

经典非定向攻击可以写成：

\[ \max_{\|\delta\|_\infty\leq\epsilon} \mathcal L(f_\theta(x+\delta),y) \]

其中 \(x\) 是原始图像，\(\delta\) 是对抗扰动，\(\epsilon\) 是扰动预算，\(f_\theta\) 是被攻击模型。

而迁移攻击的关键在于：使用代理模型 \(f_s\) 生成扰动，却希望它在未知目标模型 \(f_t\) 上依然有效。

### 5.1 PNA：为什么要抑制 Attention 的梯度？

ViT 自注意力的核心计算为：

\[ A=\operatorname{Softmax} \left(\frac{QK^\top}{\sqrt{d_k}}\right),\qquad Z=AV \]

常规反向传播会通过注意力矩阵 \(A\) 的变化和 Value \(V\) 的变化共同传递梯度。

PNA 的出发点是：某些注意力梯度可能高度依赖当前代理模型的具体行为，因此在攻击优化中停止梯度经过注意力矩阵的部分路径，可能得到更易迁移的扰动。

这不是说 Attention 不重要，而是说代理模型最敏感的梯度方向，不一定就是多个模型共有的脆弱方向。

### 5.2 TGR：为什么要正则化中间 Token 梯度？

设第 \(l\) 层的 Token 表示是：

\[ H_l\in\mathbb R^{N\times d} \]

其反向传播梯度为：

\[ G_l=\frac{\partial\mathcal L}{\partial H_l} \]

如果其中一些 Token 或通道具有极端梯度，它们可能主导对抗样本优化，使攻击过度依赖当前代理模型。

TGR 对 ViT 内部不同模块的极端梯度进行选择性抑制，再利用处理后的梯度更新输入扰动，以降低中间层梯度的变化和代理模型过拟合。

这里并不是简单地把所有梯度缩小，而是根据 ViT 内部结构处理特定的梯度分量。

### 5.3 FPR：为什么前向传播也值得优化？

设 ViT 第 \(l\) 层的计算为：

\[ H_{l+1}=F_l(H_l) \]

大多数早期迁移攻击主要改进反向传播，而 FPR 直接在前向计算中引入变化，比如：

- 对 Attention Map 进行多样化处理；
    
- 利用历史 Token Embedding 的动量平滑表示。
    

因此它改变的不仅是梯度传递规则，还改变了攻击迭代时代理模型所产生的内部表示。

这三篇论文分别体现了三种思路：

## PNA

梯度路径

哪些梯度不应过度依赖？

## TGR

梯度分布

哪些极端梯度应当抑制？

## FPR

前向表征

如何改善代理模型的内部表示？

三种方法表面上都是提高迁移成功率，但切入问题的层次明显不同，这对寻找自己的创新点很重要。

## 六、哪些论文可以称为某个方向的“第一篇”？

这里我建议采用严格一些的表述，不轻易使用“全领域首次”。

|方向|可作为开创性起点的工作|判断|
|---|---|---|
|ViT 对抗鲁棒性系统研究|Bhojanapalli et al.、Mahmood et al.，ICCV 2021|两篇重要早期工作，不唯一指定第一|
|ViT 内部 Class Token 的迁移攻击|Naseer et al.，ICLR 2022（2021.06 预印本）|早期代表性工作|
|ViT 专用 Attention 梯度迁移攻击|PNA，AAAI 2022|重要开创性方法|
|ViT Token 块稀疏攻击|Joshi et al.，2021|早期探索|
|ViT 对抗 Patch 机制研究|Gu et al.，ECCV 2022；Fu et al.，ICLR 2022|两条紧密相关的关键路线|
|ViT 专用决策式黑盒 Patch 搜索|PAR，NeurIPS 2022|重要早期代表|
|ViT 架构导向 Attention/Embedding 迁移|ATA，ACM MM 2022|关键代表|
|ViT 中间 Token 梯度正则化|TGR，CVPR 2023|代表性开创方法|
|ViT 梯度路径重构|VDC，AAAI 2024|关键方法|
|ViT 梯度缩放与频域联合优化|GNS-HFA，ICLR 2024|代表性方法|
|ViT 前向传播精炼式迁移攻击|FPR，CVPR 2025|明确提出该研究视角的重要论文|
|ViT 专用集成代理模型增强|ViT-EnsembleAttack，ICCV 2025|作者主张的早期专门化代表|
|自监督 ViT 双特征生成式迁移|dSVA，ICCV 2025|重要新路线代表|

注意，“某论文首次提出一个具体命名机制”和“整个研究方向第一次出现”是两回事。 正式做文献综述时，还需要对每条候选“首次”结论进行引文追溯。

## 七、如果要深入研究，我建议怎么读？

我建议将阅读工作分成四轮。

文献阅读清单

0 / 18 已完成

第一轮：建立问题认识

Mahmood — ICCV 2021

Bhojanapalli — ICCV 2021

Naseer — ICLR 2022

Wei / PNA-PatchOut — AAAI 2022

第二轮：ViT 特有攻击面

Adversarial Token Attacks — 2021

Patch-Fool — ICLR 2022

PAR — NeurIPS 2022

ATA — ACM MM 2022

第三轮：迁移攻击核心技术

TGR — CVPR 2023

MIG — ICCV 2023

VDC — AAAI 2024

GNS-HFA — ICLR 2024

ATT — NeurIPS 2024

第四轮：近年来的扩展

AAS — CVPR 2024

FDAP — AAAI 2024

FPR — CVPR 2025

ViT-EnsembleAttack — ICCV 2025

dSVA — ICCV 2025

另外，还有几篇非 ViT 专用但非常重要的基础文献：FGSM、PGD、MI-FGSM、DI²-FGSM。它们分别对应一阶梯度攻击、投影优化、动量和输入多样化等基础思想。迁移攻击的许多新方法，本质上仍以这些优化框架为基础。

![](https://www.google.com/s2/favicons?domain=https://mlanthology.org&sz=32)

ML Anthology

+2

## 八、从研究创新角度看，哪些方向值得重点关注？

根据现有研究，我更看好以下四个切入点。这里属于对文献趋势的判断，而不是声称这些方向尚无人研究。

|研究方向|已有基础|值得继续探索的问题|
|---|---|---|
|多粒度特征迁移攻击|Naseer、TGR、FDAP、dSVA|不同层级、空间尺度、语义粒度的特征如何协同优化？|
|Attention 与表示子空间联合攻击|PNA、ATA、FPR|如何识别跨模型共享的敏感方向，而非代理模型独有的梯度？|
|跨架构迁移|MIG、VDC、ATT、ViT-EnsembleAttack|如何同时提高 ViT→ViT、ViT→CNN 及 ViT→视觉基础模型的迁移率？|
|任务无关的视觉骨干攻击|FDAP、dSVA、SAM 相关攻击|无法知道下游任务或微调数据时，如何攻击共享视觉特征？|

其中跨任务方向可进一步参考：

- [Attack-SAM: Towards Attacking Segment Anything Model With Adversarial Examples](https://arxiv.org/abs/2305.00866?utm_source=chatgpt.com)（2023 年，早期 SAM 攻击探索）。
    
- [Transferable Adversarial Attacks on SAM and Its Downstream Models](https://arxiv.org/abs/2410.20197?utm_source=chatgpt.com)（2024 年，研究从基础 SAM 到下游模型的迁移）。
    
- [Attacking Attention of Foundation Models Disrupts Downstream Tasks](https://openaccess.thecvf.com/content/CVPR2025W/AdvML/html/Silva_Attacking_Attention_of_Foundation_Models_Disrupts_Downstream_Tasks_CVPRW_2025_paper.html?utm_source=chatgpt.com)（CVPR Workshop 2025，覆盖多种视觉下游任务）。
    

![](https://www.google.com/s2/favicons?domain=https://arxiv.org&sz=32)

arxiv.org

+2

如果要做出比单纯“组合两个攻击模块”更有说服力的创新，我建议重点研究：

\[ \boxed{ \text{多层级特征} +\text{跨架构共享子空间} +\text{迁移性优化} } \]

例如，可以先分析不同 ViT 层级的 Token 表示，以及 ViT 与 CNN 的特征之间是否存在共享的对抗敏感方向，再设计具有跨层一致性和跨模型一致性的攻击目标。

但这里有个重要前提：共享子空间的存在及其与迁移性的关联需要通过实验验证，不能直接假设它成立。

实验方面至少需要同时报告白盒 ASR、黑盒迁移 ASR、扰动预算、计算时间，以及不同代理模型与目标模型的迁移矩阵。尤其应使用一致的数据、模型预处理和攻击预算，防止不公平比较。2025 年的迁移攻击评估研究就指出了超参数和评测协议不一致可能造成的误导。

![](https://www.google.com/s2/favicons?domain=https://pubmed.ncbi.nlm.nih.gov&sz=32)

PubMed

## 九、推荐的综述文献

最后两篇适合作为继续扩展文献库的入口：

1. [Survey of Adversarial Sample Attacks for Vision Transformer](https://www.jsjkx.com/EN/10.11896/jsjkx.250600065?utm_source=chatgpt.com)，2026 年，《计算机科学》。直接针对 ViT 对抗攻击展开综述，按白盒、迁移黑盒、决策黑盒等方向梳理，并讨论模型结构、输入变换、积分梯度、下游任务与模型对齐。

2. [A Survey on Transferability of Adversarial Examples Across Deep Neural Networks](https://mlanthology.org/tmlr/2024/gu2024tmlr-survey/?utm_source=chatgpt.com)，TMLR 2024。研究整个深度神经网络迁移攻击领域，适合把 ViT 攻击放入更广泛的迁移攻击框架中理解。

![](https://www.google.com/s2/favicons?domain=https://www.jsjkx.com&sz=32)

jsjkx.com

+1

## 最终结论

如果只选出最值得作为 ViT 迁移对抗攻击研究主线的六篇，我的优先顺序是：

1. Naseer，ICLR 2022：Self-Ensemble 与 Token Refinement，理解中间层表征如何增强迁移性。
    
2. Wei，AAAI 2022：PNA + PatchOut，理解针对 ViT 结构设计攻击策略。
    
3. Zhang，CVPR 2023：TGR，理解中间 Token 梯度正则化。
    
4. Zhang，AAAI 2024：VDC，理解反向传播路径的作用。
    
5. Ming，NeurIPS 2024：ATT，理解自适应梯度处理与 Patch 优化。
    
6. Ren，CVPR 2025：FPR，理解前向传播精炼与表征优化。
    

如果目标是全面了解 ViT 安全，而不只是迁移攻击，再补上 Mahmood 的 ICCV 2021、Patch-Fool 的 ICLR 2022 和 PAR 的 NeurIPS 2022。

这套文献组合既能覆盖领域早期关键贡献，也能建立从 ViT 结构脆弱性 → 梯度优化 → 表征优化 → 跨模型迁移 的清晰研究脉络。

---

## 笔记联系

- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]：文献资料可借助综述页的分类框架阅读。
