---
title: "SAM Attack - 分割基础模型攻击发展历程"
aliases:
  - "SAM Attack（Segment anything model）发展历程"
分类: "ViT与基础模型攻击"
笔记类型: "研究综述"
整理日期: 2026-10-27
---

[[首页|首页]] · [[30_对抗机器学习/ViT与基础模型攻击/索引 - ViT与基础模型攻击|ViT与基础模型攻击索引]]

# SAM Attack - 分割基础模型攻击发展历程



要详细理解 SAM Attack，首先需要明确一个问题：SAM 在不同研究语境下可能指不同的技术，不能仅凭缩写就把它们当成同一种攻击。

在计算机视觉与对抗攻击研究中，SAM 常见的含义包括：

- Segment Anything Model（SAM）：由 Meta 提出的通用图像分割模型。相关攻击研究关注如何通过对抗扰动、提示操纵等方式影响模型的分割结果。
    
- Sharpness-Aware Minimization（SAM）：一种用于模型训练的优化方法。它本身不是一种对抗攻击，但可能出现在对抗鲁棒性研究中。
    

如果你所说的是前者，那么我们需要研究的是：针对 Segment Anything Model 的对抗攻击是如何发展起来的，研究者如何从传统的图像扰动攻击，逐步转向提示依赖、分割掩码操纵、跨模型迁移以及通用攻击等问题。

下面先按这一研究方向梳理发展脉络，并区分基础技术、具体攻击方法与后续研究问题，避免把所有 SAM 相关工作都误称为同一种攻击。

## 一、研究背景：为什么 SAM 会成为对抗攻击的目标？

![Prompt-SAM: A Vision-Language and SAM based Hybrid Framework for Prompt-Augmented Zero-Shot Segmentation | Human-Centric Intelligent Systems | Springer Nature Link](https://images.openai.com/static-rsc-4/DzWZWn-2bylnrnlnLSoo6ZFhlaZRbw7LaT0SWkFEaTxw7zRTOIN4HiPJk9K4Y1gGmh6RoQoUOwvH7IlVaYrRg1ZerwdVKdCCQGTnaK6Wx_8V06sgLYAXsgYflBnxU_o9AIq5wzOXRFSINnmHRvzUcZvHq6dDQ7ImcGh3rQm-W1U?purpose=inline)

Segment Anything Model

提示驱动的通用图像分割模型

![Transferable Adversarial Attacks on SAM and Its Downstream Models · NeurIPS 2024](https://images.openai.com/static-rsc-4/9pctGtTBe-XLiGtnVe9jCoerTpTYMl5WcvgTja_As-yhecFwbSbbmVqJyNj_S34gcqLebIed8rX8RARDX1GO81P5Y0PP-IahAI1XEDzFtWKXjzd7ruY4iKz39Met7wQZ0V8vE3_9b8izQclj4LMyAMf_IOJSYLr-HWIqv-wEoYQ?purpose=inline)

对抗攻击示意

输入图像轻微变化，分割结果可能明显改变

2023 年，Meta 发布了 Segment Anything Model (SAM)。与传统图像分类模型不同，SAM 的核心任务不是判断一张图片属于猫、狗还是汽车，而是**根据用户给出的提示，将图像中的目标区域分割出来**。相关基础工作是 Kirillov 等人的论文 Segment Anything。

### 1. SAM 的工作机制

SAM 的基本输入和输出可以写成：

$$
M=f_{\theta}(x,p) 
$$

其中：

- $(x)$：输入图像。
    
- $(p)$：用户提供的提示，例如点、框等。
    
- $(f_{\theta})$：SAM 模型。
    
- $(M)$：输出的分割掩码（Mask），用于标记目标区域。
    

例如，给 SAM 一张包含狗的图片，在狗身上点击一个点，模型就可能生成狗的分割掩码。

但这带来了一个关键问题：如果输入图像只发生很小的变化，模型是否仍然能够正确分割目标？

对抗攻击研究正是从这个问题切入的。

### 2. 为什么传统分类攻击不能直接照搬？

在传统分类任务中，攻击者通常希望让模型把猫识别成狗。模型输出类别概率或分类分数，攻击目标相对明确。

SAM 则具有三个不同特点：

|特点|对攻击研究的影响|
|---|---|
|输出是空间掩码|需要考虑大量像素的位置和结构，而不只是一个类别|
|依赖用户提示|同一张图像在不同提示下可能产生不同掩码|
|具有较强的零样本泛化能力|需要检验它面对未知图像、不同提示和其他模型时是否仍然脆弱|

因此，SAM 对抗攻击逐渐形成了自己的研究问题：**如何定义攻击目标、如何构造分割损失、如何降低对提示的依赖，以及如何让攻击从已知模型迁移到未知模型**。


## 二、发展历程总览

下面按代表性论文的公开时间梳理。这条时间线以针对 Segment Anything Model 的对抗攻击为主，不把名称相似的模型训练方法混为一谈。

## 2023.05

第一阶段：证明 SAM 可以被攻击

Attack-SAM

从掩码移除开始，验证传统梯度攻击能否破坏提示驱动的分割结果。

## 2023.10

第二阶段：黑盒与定向攻击

Black-box Targeted Adversarial Attack；SAM Meets UAP

分别研究如何让模型生成指定目标的掩码，以及能否用一份通用扰动攻击多张图像。

## 2024

第三阶段：提示无关与区域级攻击

Segment Shards；Practical Region-level Attack

从攻击一个固定提示，转向让一整个目标区域在不同点击位置下都难以被正确分割。

## 2025–2026

第四阶段：区域结构、通用扰动与系统化评测

Region-Guided Attack；SAM-BM；SAM Meets UAP 后续研究

进一步关注分割区域之间的**结构关系**、单一扰动的**泛化能力**，以及不同损失函数、目标尺度和攻击策略的影响。

说明：这是按代表性工作整理的研究脉络，并非 SAM 攻击领域所有论文的完整清单。各条研究路线存在时间上的交叉。

## 三、第一阶段：2023 年——从证明漏洞到建立攻击框架

### 3.1 代表性工作：Attack-SAM

论文：Attack-SAM: Towards Attacking Segment Anything Model With Adversarial Examples，2023 年 5 月公开。

这是针对 SAM 的早期代表性攻击研究之一。作者包括 Chenshuang Zhang、Chaoning Zhang 等人。论文首先系统考察了 SAM 在白盒环境与基于迁移的黑盒环境下的对抗脆弱性。

论文链接：[Attack-SAM，arXiv:2305.00866](https://arxiv.org/abs/2305.00866)

#### （1）它试图解决什么问题？

传统攻击通常把目标设定为改变模型的分类结果。但 SAM 输出的是掩码，直接将分类任务中的交叉熵损失照搬过来，并不一定符合攻击目的。

因此，Attack-SAM 先提出一个清晰的基本目标：

Mask Removal（掩码移除）：使原本能够分割出来的目标，在添加对抗扰动后无法被正常分割。

例如，原始图片中的狗能够被 SAM 完整分割。攻击之后，即使图像在人眼看来几乎没有变化，SAM 输出的狗的掩码也可能显著缩小，甚至变为空掩码。

#### （2）它如何实现攻击？

首先，定义原始图像为 $(x)$，对抗扰动为 $(\delta)$，提示为 $(p)$。攻击后的图像为：

$$
[ x_{\mathrm{adv}}=x+\delta ]
$$

模型输出可以表示为：

$$
[ y_{\mathrm{adv}}=f_{\theta}(x+\delta,p) ]
$$

这里的 $(y_{\mathrm{adv}})$ 表示掩码预测所依据的像素级输出分数。

Attack-SAM 的关键不只是对图像做扰动，而是设计一个与掩码移除目标相符的损失函数。

论文提出了 ClipMSE。它先将低于负阈值的输出截断，再通过均方误差形式推动剩余输出朝负阈值方向变化。直观地说，就是尽量让目标区域的像素不再被模型判定为前景。

其核心思路可以简化理解为：

$$
[ \mathcal{L}_{\mathrm{mask}} =\left\| \operatorname{Clip}(y_{\mathrm{adv}},t)-t \right\|_2^2 ]
$$

其中 $(t)$ 是负阈值。这里的公式用于说明损失设计的核心思想；具体截断操作与阈值设置应以论文原文为准。

为什么不直接使用普通 MSE？

如果某些像素已经被推到足够低的分数，继续要求这些像素接近阈值，未必对攻击有帮助。ClipMSE 通过截断处理，避免这部分像素继续产生不必要的梯度。

#### （3）为什么使用 FGSM 和 PGD？

Attack-SAM 将经典梯度攻击引入 SAM 的掩码预测任务中。

- FGSM：利用一次梯度更新生成扰动，计算成本较低。
    
- PGD：反复更新扰动，并将其约束在允许的扰动范围内，通常可以获得更强的攻击效果。
    

论文默认实验采用的最大扰动幅度为 ($8/255$)，PGD 使用 10 次迭代。在该论文报告的掩码移除实验中，ClipMSE 配合 PGD-10 后，攻击前后掩码的平均 IoU 降到了 0.0487；不攻击时为 1.0。

需要注意，这个数字是特定实验设置下的结果，不代表 SAM 在所有图片、所有提示下都只有如此低的分割能力。

#### （4）它的另一项重要发现：跨提示迁移

SAM 的输入不仅有图像，还有提示。如果攻击只能在某个固定点击点上生效，那么攻击的实际适用性会受到很大限制。

Attack-SAM 研究了两个问题：

- **跨提示迁移**：针对一个提示生成的扰动，能否影响另一个提示下的分割结果？
    
- **跨任务迁移**：在其他视觉任务上生成的对抗样本，能否对 SAM 的掩码预测产生影响？
    

论文发现，跨提示攻击具有一定效果，来自其他任务的对抗样本也能够在一定程度上影响 SAM。作者还进一步探索了使 SAM 生成指定掩码的定向攻击。

#### （5）这一阶段的贡献与局限

主要贡献

- 将对抗攻击正式引入提示驱动的 SAM 分割任务。
    
- 提出面向掩码移除的损失设计。
    
- 初步研究跨提示、跨任务迁移和定向掩码生成。
    

主要局限

- 白盒攻击依赖对目标模型梯度的访问。
    
- 许多实验围绕已知提示展开。
    
- 单张图像上优化的扰动，不等于能够稳定攻击其他图像的通用扰动。
    
这一阶段建立了研究起点：SAM 并不是因为具备强大的零样本能力，就天然免疫于对抗攻击。

## 四、第二阶段：2023 年下半年——黑盒定向攻击与通用扰动

第一阶段证明了 SAM 可以被攻击，但还留下两个不同的问题：

1. 如果无法直接访问目标 SAM 的内部参数和梯度，能否实现攻击？
    
2. 如果每张图像都要重新生成扰动，能否找到一份可以应用于多张图像的通用扰动？
    

这两条路线在 2023 年开始得到进一步探索。

### 4.1 黑盒定向攻击：让 SAM 输出指定的掩码

论文：Black-box Targeted Adversarial Attack on Segment Anything (SAM)，2023 年 10 月公开。

论文链接：[arXiv:2310.10010](https://arxiv.org/abs/2310.10010)

这项工作把研究重点放在黑盒定向攻击（Targeted Adversarial Attack，TAA）上。它不满足于让原来的掩码消失，而是希望攻击后的图像产生一个预先指定的目标掩码。


举个例子：

- 原始图像中，SAM 将狗分割出来。
    
- 攻击者指定一个目标掩码。
    
- 攻击后的图像输入 SAM 后，模型输出尽可能接近指定的目标掩码。
    

这与掩码移除有本质区别：

| 比较维度 | 掩码移除     | 定向掩码攻击         |
| ---- | -------- | -------------- |
| 攻击目标 | 让原有掩码失效  | 让输出接近指定掩码      |
| 成功标准 | 原始分割明显退化 | 输出与目标掩码足够相似    |
| 研究难点 | 破坏原有分割   | 同时控制输出的具体形状与位置 |

该论文提出只攻击图像编码器的思路，以减少对特定提示的依赖，并设计正则化损失以提升跨模型迁移能力。这里的“黑盒”是指攻击目标模型的信息受到限制；它并不意味着攻击过程一定完全不需要任何替代模型或先验信息。


这条路线的重要意义在于：研究者开始不仅问“能不能让 SAM 出错”，还问“能不能**控制它具体错成什么样**”。

### 4.2 通用对抗扰动：SAM Meets UAP

论文：SAM Meets UAP: Attacking Segment Anything Model With Universal Adversarial Perturbation，2023 年 10 月公开。

论文链接：[arXiv:2310.12431](https://arxiv.org/abs/2310.12431)

UAP 是 Universal Adversarial Perturbation 的缩写，即通用对抗扰动。它的目标不是为每张图像单独优化扰动，而是寻找一份固定扰动，使它对许多不同图像都能产生攻击效果。


可以把两种设定写成：

图像相关扰动

$$
[ x_i^{\mathrm{adv}}=x_i+\delta_i ]
$$

每张图像 $(x_i)$ 都有自己的扰动 $(\delta_i)$。

通用扰动

$$
[ x_i^{\mathrm{adv}}=x_i+\delta ]
$$

多张图像共享同一份扰动 ($\delta$)。

两者的困难程度并不相同。单张图像的扰动可以针对该图像的具体特征进行优化；通用扰动则必须找到**对多种图像都有效的共同影响因素**。

这项研究发现，传统的图像相关攻击框架不能直接有效地解决 SAM 的通用攻击问题。因此，作者提出基于自监督对比学习（Contrastive Learning，CL）的扰动中心框架：

- 将待优化的通用扰动作为锚点样本。
    
- 对该扰动进行增强，构造正样本。
    
- 从图像编码器提取其他自然图像的特征，形成负样本记忆库。
    
- 利用对比学习目标优化通用扰动的表示。
    

它体现了一种思路上的变化：不再只针对一张图像的分割损失优化像素，而是尝试利用特征表示层面的关系学习一份更具有跨图像影响力的扰动。


### 4.3 两条路线之间的区别

黑盒定向攻击和通用攻击虽然都试图提高实际攻击能力，但解决的是不同问题：

- 黑盒定向攻击强调攻击者对目标模型的访问限制，以及攻击结果的可控性。
    
- 通用扰动强调同一份扰动跨不同图像的复用能力。
    

因此，不能把“黑盒攻击”“定向攻击”和“通用攻击”视为同一个概念。它们可以组合，但彼此不是同义词。

## 五、第三阶段：2024 年——从固定提示攻击转向跨提示攻击

2024 年是 SAM 对抗攻击研究中一个值得单独分析的阶段。研究者逐渐发现，攻击是否能在某一个点击点上成功，并不能充分反映攻击的实用性。

真正重要的问题是：如果攻击者不知道用户将在哪里点击目标，或者用户改变了提示，攻击还能否生效？

这一问题催生了**跨提示攻击、区域级攻击，以及针对提示驱动分割模型的更一般化攻击**。

### 5.1 Segment Shards：利用批量提示增强跨提示迁移

论文：Segment Shards: Cross-Prompt Adversarial Attacks against the Segment Anything Model，2024 年 4 月发表。

论文链接：[DOI: 10.3390/app14083312](https://doi.org/10.3390/app14083312)

这项工作由 Shize Huang、Qianhui Fan 等人提出，核心方法称为 Prompt Batch Attack（PBA，提示批量攻击）。


#### （1）为什么要研究提示批量攻击？

假设攻击者只知道一个点击点：

- 在这个点上优化扰动；
    
- 用户换到目标的另一处点击；
    
- SAM 可能又能正确分割目标。
    

这说明，针对单一提示的攻击可能只破坏了模型的一种输入条件，而没有充分影响模型的整体图像表示。

作者因此研究了三种使用提示信息的策略。

|方法|提示信息的使用方式|
|---|---|
|NPA|不使用提示，主要通过图像编码器的表示构造攻击|
|PA|在攻击过程中使用提示，联合考虑图像编码器与掩码预测过程|
|PBA|将提示划分为批次，在不同迭代中使用不同的提示批次|

PBA 的重点不是简单地把更多提示一次性塞进模型，而是在优化过程中利用不同提示批次，并结合动量信息稳定扰动的更新方向。

#### （2）为什么批量提示可能提高迁移能力？

SAM 可以抽象为三个组成部分：

$$
[ z_{\mathrm{img}}=E_{\mathrm{img}}(x) ]
$$

$$
[ z_{\mathrm{prompt}}=E_{\mathrm{prompt}}(p) ]
$$

$$
[ M=D(z_{\mathrm{img}},z_{\mathrm{prompt}}) ]
$$

其中，$(E_{\mathrm{img}})$ 是图像编码器，$(E_{\mathrm{prompt}})$ 是提示编码器，$(D)$ 是掩码预测模块。

单一提示攻击只约束特定的 $(p)$。而多个提示参与优化时，攻击会受到多个不同提示条件下的输出约束。

从优化角度看，这意味着攻击不再只追求某一个输入组合上的损失，而是试图找到能在多个提示条件下产生影响的扰动方向。

#### （3）实验结果与意义

论文报告，在其多数据集实验中，与基线方法相比，PBA 的跨提示攻击成功率平均提高了 17.83%，攻击成功率平均提高了 20.84%。这些提升对应论文所采用的实验设置和比较方法。


它的主要意义在于：

- 把跨提示迁移作为一个明确的研究目标。
    
- 提出相应的跨提示攻击成功率指标。
    
- 从单一目标的分割破坏，扩展到影响整张图像中的多个分割区域。
    

标题中的 “Segment Shards” 也表达了这种结果：模型不再稳定地生成完整、连贯的目标掩码，而可能出现碎片化的分割结果。

### 5.2 Practical Region-level Attack：从一个点击点扩展到整个区域

论文：Practical Region-level Attack against Segment Anything Models，CVPR 2024 Workshop。

论文链接：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/html/Shen_Practical_Region-level_Attack_against_Segment_Anything_Models_CVPRW_2024_paper.html)

这项工作由 Yifan Shen、Zhengyuan Li 和 Gang Wang 提出，研究重点是更加实际的区域级攻击。


#### （1）核心问题：攻击者未必知道用户的准确点击位置

假设目标是一只狗。

传统点级攻击可能针对狗身上的一个特定位置进行优化。但真实使用场景中，用户可能点击狗的头部、身体或者腿部。

如果攻击只在某个点击点有效，就很难保证它能持续影响用户的操作。

区域级攻击将目标改成：在攻击者指定的某个区域内，无论用户在哪个位置点击，都尽可能使 SAM 无法正确分割目标。

#### （2）从点到区域，改变了什么？

点级攻击

区域级攻击

上图为概念示意，不是论文实验结果。

论文提出了两种相关方法：

- S-RA（Sampling-based Region Attack）：通过对指定区域内的提示位置进行采样，以有限的采样点近似表达整个区域的攻击目标。
    
- T-RA（Transferable Region Attack）：在区域攻击的基础上，利用频谱变换相关技术提升黑盒迁移能力。
    

该研究还报告了在真实 SAM 服务上的测试结果，表明区域级威胁模型不仅是理论设定，也具有实际评估价值。


#### （3）这一工作的研究意义

它推动了威胁模型的变化：

- 从攻击者知道准确提示，转向攻击者只知道目标所在区域。
    
- 从单一模型的白盒攻击，进一步考虑黑盒迁移。
    
- 从“攻击是否成功”，转向“攻击能否在用户实际交互方式变化时继续生效”。
    

这里的关键进步并不是简单地增加扰动强度，而是重新定义攻击需要满足的条件。

### 5.3 UAD：通过模拟形变攻击提示驱动分割模型

论文：Unsegment Anything by Simulating Deformation，CVPR 2024。

论文链接：[CVPR Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Unsegment_Anything_by_Simulating_Deformation_CVPR_2024_paper.html)

这项工作由 Jiahao Lu、Xingyi Yang 和 Xinchao Wang 提出。它将研究范围从单一 SAM 模型，扩展到具有不同架构和提示接口的提示驱动分割模型。


#### （1）为什么需要这种方法？

不同提示驱动分割模型的内部结构可能不同，针对一个模型生成的扰动，不一定能迁移到另一个模型。

作者观察到，提示相关的对抗噪声具有异质性，单纯针对某个提示优化的扰动往往难以稳定迁移。

因此，UAD 的目标是研究一种更不依赖特定提示、对不同提示驱动分割模型更具有迁移性的攻击。

#### （2）核心思想：破坏图像结构所依赖的表示

UAD 的名称来自 Unsegment Anything by Simulating Deformation。它通过可微的形变函数构造目标形变图像，再结合特征距离约束优化攻击。

这里的可微意味着：形变过程可以参与梯度计算，从而把图像结构变化纳入优化框架。

与直接对每个像素施加独立扰动相比，这种方法把注意力放在图像结构如何改变模型内部特征上。

论文还报告，具有目标性的特征攻击在迁移方面表现更好，这提示研究者：对抗迁移能力与图像编码器所形成的特征表示关系密切。


需要区分的是，UAD 与普通图像几何变换不是同一个概念。普通形变可以只是数据增强；UAD 则将可微形变纳入有明确攻击目标的优化过程。

### 5.4 2024 年的其他工作：红队评估与攻击面的扩展

除上述方法外，Red-Teaming Segment Anything Model 在 CVPR 2024 Workshop 上从多个角度评估了 SAM，包括风格变化、隐私相关问题，以及文本提示场景下的分割攻击。该研究还提出了 Focused Iterative Gradient Attack（FIGA），探索以较少修改像素实现攻击的可能性。


这类工作与专门设计攻击算法的论文有所不同：它们不只提出某个新算法，也尝试通过系统性红队测试发现模型在真实使用场景中的风险。

## 六、第四阶段：2025—2026 年——区域结构与系统化评测

这一阶段的代表性趋势，是更加重视分割任务本身的空间结构，以及攻击方法的可比较性。

### 6.1 Region-Guided Attack：以区域结构指导攻击

论文：Region-Guided Attack on the Segment Anything Model (SAM)，2024 年 11 月公开预印本，后发表于 Neural Networks。

论文链接：[arXiv:2411.02974](https://arxiv.org/abs/2411.02974)

作者 Xiaoliang Liu、Furao Shen 和 Jian Zhao 提出了 Region-Guided Attack（RGA）。该论文在 2024 年已经公开预印本，之后于 2026 年发表，因此应当按这个时间顺序理解，而不是把它视为 2026 年才首次出现的研究。


#### （1）为什么要关注区域结构？

SAM 的输出不仅包含前景和背景，还具有明显的空间结构：大区域、小区域、区域边界以及不同区域之间的关系。

如果攻击只把所有像素一视同仁，可能无法充分利用这些结构信息。

RGA 的核心观点是：针对分割模型的攻击应当考虑区域本身的结构，而不只是整体像素误差。

#### （2）Region-Guided Map

RGA 引入 Region-Guided Map（RGM，区域引导图），通过区域分割与膨胀策略生成攻击指导信号。

其基本思想包括：

- 使较大的分割区域发生碎片化。
    
- 使较小的区域扩张。
    
- 利用区域结构的改变，影响 SAM 最终生成的分割结果。
    

论文描述的 SAD 策略用于生成区域引导信息，并结合这些信息优化对抗扰动。


从研究思想上看，这意味着攻击目标不再局限于某个已知目标掩码的移除，而是试图从更整体的区域结构层面扰乱分割结果。

#### （3）与早期方法的区别

| 研究方向       | 主要关注点             |
| ---------- | ----------------- |
| Attack-SAM | 通过掩码移除证明模型存在对抗脆弱性 |
| PBA        | 利用多个提示提高跨提示迁移能力   |
| 区域级攻击      | 降低对准确点击位置的依赖      |
| UAD        | 通过形变与特征约束研究跨模型攻击  |
| RGA        | 利用区域结构生成攻击指导信号    |

RGA 的研究结果覆盖白盒和黑盒场景，作者报告了较好的攻击效果和迁移能力。不过，具体效果仍然取决于实验数据、扰动约束和模型配置，不能据此推断所有 SAM 变体都会以相同程度受到影响。


### 6.2 SAM-BM：从单个攻击方法走向系统化基准评测

论文：SAM-BM: An adversarial benchmark for loss functions and multi-scale objects in segment anything models，2026 年。

论文链接：[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1077314226001566)

这项工作把研究重点放在**评估框架**上，而不是仅仅提出一个新的攻击算法。它系统比较了不同损失函数、攻击策略和目标尺度对 SAM 及其变体的影响。


研究覆盖了：

- 13 种损失函数。
    
- FGSM、PGD 和 SegPGD 三种攻击策略。
    
- 6 种目标尺度。
    
- 两种梯度投影方法：tanh 与 linear。
    

论文指出，损失函数的选择会明显影响攻击效果与计算效率，目标尺度也会与损失函数、攻击策略产生交互影响。


这类工作的重要性在于，它开始回答一个更基础的问题：

如果两篇论文采用不同的损失函数、目标尺度和评估设置，它们报告的攻击成功率究竟能否直接比较？

没有统一评测框架时，一种方法看起来更强，可能部分来自实验设置不同，而不一定完全源于算法本身。

### 6.3 SAM Meets UAP：通用攻击研究继续发展

通用扰动并不是到 2026 年才出现的方向。前面介绍的 SAM Meets UAP 在 2023 年就已公开。

到 2026 年，相关工作继续发展。例如，SAM Meets UAP: Attacking Segment Anything Model With Universal Adversarial Perturbation 的期刊版本于 2026 年 8 月在线发表，进一步强调了通用扰动对 SAM 鲁棒性研究的重要性。


这条路线与区域级攻击并不相同：

- 区域级攻击强调对目标区域内不同提示的适应性。
    
- 通用扰动强调同一份扰动在不同图像上的复用能力。
    
- 跨模型攻击则强调扰动能否影响具有不同内部结构的模型。
    

这些目标可能相互关联，但需要分别进行实验验证。

## 七、把发展历程串起来：SAM Attack 到底经历了什么变化？

前面介绍了多篇论文。为了真正理解这个领域，不能只记住论文名称，还需要看清楚每一阶段的研究目标发生了什么变化。

阶段一：证明模型存在漏洞

代表工作：Attack-SAM（2023）

研究者先验证：小幅对抗扰动能否使 SAM 的掩码预测发生明显变化？由此建立面向掩码输出的攻击目标与损失设计。

阶段二：让攻击结果更可控

代表工作：黑盒定向攻击（2023）

研究目标从让掩码消失，扩展到让输出接近指定的掩码，同时探索有限模型访问条件下的攻击。

阶段三：提高跨提示迁移能力

代表工作：PBA、区域级攻击（2024）

攻击不再只针对一个点击点，而是考虑不同提示、目标区域和用户交互方式的变化。

阶段四：提高跨模型与跨图像泛化能力

代表工作：UAD、SAM Meets UAP

研究者分别探索利用图像结构和特征表示实现跨模型迁移，以及利用固定扰动攻击多张图像。

阶段五：区域结构与统一评测

代表工作：RGA、SAM-BM（2024 年预印本至 2026 年）

研究开始更加重视区域间的结构关系，以及不同损失函数、目标尺度和攻击方法之间的公平比较。

这些阶段不是严格的先后替代关系。通用扰动、跨提示攻击和区域结构攻击是相互交叉的研究方向，而不是前一种方法被后一种方法完全取代。


## 八、当前研究仍然有哪些难点？

截至 2026 年 10 月，结合上述代表性工作，可以把 SAM 对抗攻击的关键问题归纳为以下五个方面。

### 8.1 跨提示迁移能力

一个攻击在特定提示下成功，不代表用户改变点击位置后仍然成功。

因此，跨提示攻击需要明确区分：

- 生成扰动时使用的源提示。
    
- 测试攻击效果时使用的目标提示。
    
- 提示变化后，攻击成功率下降多少。
    

PBA 和区域级攻击分别从提示批次与区域覆盖的角度探索了这个问题。


### 8.2 跨模型迁移能力

在白盒设置下，可以利用目标模型的梯度优化扰动；在黑盒设置下，往往需要依赖替代模型、特征迁移或其他可用信息。

困难在于，不同模型的图像编码器、掩码解码器和提示接口可能不同。某个模型上的攻击效果，不能直接推断为其他模型也具有相同的脆弱性。

UAD 和黑盒定向攻击都研究了这一问题，但采用的技术路线不同。


### 8.3 通用扰动的稳定性

单张图像上的攻击成功，并不意味着同一扰动能在不同图像上有效。

通用扰动研究需要回答：

- 扰动对不同图像的成功率是多少？
    
- 更换数据集后效果是否保持？
    
- 更换 SAM 变体后是否仍然有效？
    
- 扰动在保持视觉质量的同时，能否产生稳定的分割退化？
    

SAM Meets UAP 直接研究了图像无关的通用扰动，但这与针对单张图像优化的攻击仍是不同的实验问题。


### 8.4 分割结构与攻击目标的匹配

分割输出是一张空间掩码。一个好的攻击目标，不应只考虑整体数值误差，还应考虑研究者到底希望破坏什么：

- 让整个目标消失。
    
- 让掩码与指定目标不一致。
    
- 让区域变得碎片化。
    
- 让边界、区域大小或空间关系发生变化。
    

RGA 的区域引导思想说明，区域结构可以成为设计攻击目标的重要依据。


### 8.5 评估结果是否公平、可复现

不同研究可能使用不同的数据集、模型版本、提示类型、扰动预算与成功标准。仅仅比较论文摘要中的攻击成功率，很容易得出不可靠的结论。

SAM-BM 的价值就在于系统研究损失函数、攻击策略与目标尺度的影响。后续研究还需要明确模型版本、数据处理方式、提示生成规则和评测协议。


## 九、如果你想系统学习 SAM Attack，建议按什么顺序阅读？

如果你的目标是深入理解这一研究方向，而不是只了解发展史，我建议按以下顺序阅读原论文。

![Chenshuang Zhang](https://images.openai.com/static-rsc-4/B10RgwbLV6GRlsfliYiLCO-e49654oXL-Aw0Fi2coQ-KzcqHr38v85UhD55eEKo-NOTgMtq-YMiD2ABb6Sz_Ji4SXIowpRYEUOY31ldjoQSXfX8KPHvfYhoYixToVqyTF0uwUPg2Ug-1pUk0esyofDs_4F57YIhQprPugIG9Zy4?purpose=inline)

1. Attack-SAM

2023 · 基础攻击框架

先理解 SAM 的输入、掩码输出、ClipMSE、FGSM、PGD，以及跨提示和跨任务迁移的基本概念。

[阅读原论文](https://arxiv.org/abs/2305.00866)

![SAM Meets UAP: Attacking Segment Anything Model With Universal Adversarial Perturbation | AI Research Paper Details](https://images.openai.com/static-rsc-4/i4_cnmcL2xjBeQv2c1VaOT4eOfYOwtA1MxOGD198xg3MGsFvCBC_amuT6iGkrME3ZDcvs59TRntGOGbW62-Zb3sLWTvNRGZooh3BaaGrIGYk5Aqdgqm_vadz-OEuMp_5DLTclqEJYGetjJqddq5vUI4oqfkEUmVx6Yx0SoiA6oM?purpose=inline)

2. SAM Meets UAP

2023 · 通用扰动

理解为什么逐图优化不等于通用攻击，以及作者如何将自监督对比学习用于扰动生成。

[阅读原论文](https://arxiv.org/abs/2310.12431)

![NeurIPS Poster Transferable Adversarial Attacks on SAM and Its Downstream Models](https://images.openai.com/static-rsc-4/XAj0p7cpXbBWS5KUnVlhENmD2Z06cBjcuVhyKqyNOA1sTPeXEchSC5RoeiPcxDGMuUOAHLXHyrA0yto_rmVsTNQpY5g-feb3SCfT99sePOTK1tgzEuDK8yrwU1rZaDwYqnQKMGqB9UEax_ZTa-lhiA81DTIrhBedcKcRRvUYUHA?purpose=inline)

3. Segment Shards

2024 · 跨提示攻击

重点阅读 NPA、PA、PBA 的区别，以及跨提示攻击成功率的定义。

[阅读原论文](https://doi.org/10.3390/app14083312)

![Region-Guided Attack on the Segment Anything Model (SAM) | AI Research Paper Details](https://images.openai.com/static-rsc-4/KmNlPidhzntqVx5UVgnQYgwvsiHrF_pCBM8WoxNwF5zP1X0YTNV5GkXMwGn2DFRmwrtEzsGSfA-D4nwoeGHXsa5ELR0Zo5NibfFSTfhRv8dswZN6UP7hWjj3xvyWlBwKQshpFrnK2Px0PTL57ASd_13Gixv50b3moHBrEywDTR4?purpose=inline)

4. Practical Region-level Attack

2024 · 区域级攻击

重点理解点级威胁模型为什么不够实际，以及 S-RA、T-RA 如何将攻击目标扩展到一个区域。

[阅读原论文](https://openaccess.thecvf.com/content/CVPR2024W/TCV2024/html/Shen_Practical_Region-level_Attack_against_Segment_Anything_Models_CVPRW_2024_paper.html)

![CVPR Poster Unsegment Anything by Simulating Deformation](https://images.openai.com/static-rsc-4/rgjSPgkburslJfBbhQaZbm6OoQBz6D_WQzuq1RsS8gPCD8FG3NjVztESNLHXj8yZ3-jaEPo5d-A6Kb27UyOHTldtyUWQIzJ2dwTdkMdNpnoYvjfM5HbN-NAo05rYg1eisktRggUztbBL5Z5nk2mSWkP-yCgpl5H6fcUUi89be1s?purpose=inline)

5. Unsegment Anything by Simulating Deformation

2024 · 形变与跨模型迁移

重点理解图像编码器特征、可微形变，以及提示无关攻击的设计思路。

[阅读原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Lu_Unsegment_Anything_by_Simulating_Deformation_CVPR_2024_paper.html)

![Red-Teaming Segment Anything Model | AI Research Paper Details](https://images.openai.com/static-rsc-4/nO5oj6N65E-5n40gXksuIJF8VP09VU8-96cyb5Ksa--lKZEJTtg6udWqDSJdwLeO1f7IAPmqyOgN7yCv04qVm8DO242rcm5Vv4LT40ChH6qjLyKfHHLou9U2sy9VL901k46xJKNqEooAq7rSIqk-YkBgHv7divpXRs5Dm8lHUcM?purpose=inline)

6. Region-Guided Attack

2024 年预印本 · 2026 年期刊发表

最后研究区域引导图、区域结构与攻击目标之间的关系，并思考怎样设计更符合分割任务特点的评估指标。

[阅读原论文](https://arxiv.org/abs/2411.02974)

---

## 笔记联系

- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]：对照分类模型与分割基础模型的攻击目标。
- [[20_深度学习/网络结构/U-Net - 编码器与解码器分割网络|U-Net - 编码器与解码器分割网络]]：结合分割网络背景比较攻击目标与输出结构。
