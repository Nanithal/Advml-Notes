---
title: "论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations"
aliases:
  - "Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations论文"
分类: "通用与稀疏扰动"
笔记类型: "论文阅读"
整理日期: "2026-10-27"
time: "2026-08-26T11:17:00"
tags:
  - "通用攻击"
  - "GD-UAP"
  - "无数据"
  - "知识库/论文阅读"
  - "主题/通用与稀疏扰动"
状态: "已整理"
摘要: "围绕“论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations”，整理研究背景与问题、核心方法：数据无关的激活最大化目标、利用先验信息增强扰动（三种变体）。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/通用与稀疏扰动/总结 - 通用与稀疏扰动]]"
相关笔记:
  - "[[30_对抗机器学习/通用与稀疏扰动/论文 - Universal Adversarial Perturbations]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples]]"
  - "[[30_对抗机器学习/通用与稀疏扰动/论文 - Data-Free Universal Adversarial Perturbation and Black-Box Attack]]"
论文条目:
  - "[[论文来源目录#gduap]]"
  - "[[论文来源目录#uap]]"
论文标题:
  - "Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations"
  - "Universal Adversarial Perturbations"
发表载体:
  - "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)"
  - "CVPR"
发表年份:
  - "2020"
  - "2017"
发表类型:
  - "期刊论文"
  - "会议论文"
发表链接:
  - "https://doi.org/10.1109/TPAMI.2018.2861800"
  - "https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html"
预印本链接:
  - "https://arxiv.org/abs/1801.08092"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/通用与稀疏扰动/总结 - 通用与稀疏扰动|通用与稀疏扰动总结]]

# 论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations

这篇论文《Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations》（TPAMI 2020 扩展版）提出了一种**无需数据（data-free）**、**任务通用（task-generalizable）**的通用对抗扰动（UAP）生成方法 GD-UAP**，是对作者之前 BMVC 2017 会议工作的完整扩展。以下是纯内容总结：

---

### 1. 研究背景与问题

- **通用对抗扰动（UAP）**：一个固定的图像无关扰动，添加到任意自然图像上可使分类器高概率出错（Moosavi-Dezfooli et al., CVPR 2017）。
    
- **现有 UAP 方法存在三大缺陷**：
    
    1. **数据依赖**：需要大量训练数据样本，且欺骗性能与数据量成正比。
        
    2. **黑盒性能弱**：在完全黑盒（目标模型架构+训练数据均未知）场景下表现显著下降。
        
    3. **任务特定**：针对分类、分割、回归等不同任务需设计不同目标函数，难以迁移。
        
- **本文目标**：提出一个**无需任何训练数据、可跨任务通用**的 UAP 生成框架。
    

---

### 2. 核心方法：数据无关的激活最大化目标

- **核心思想**：不直接优化"改变标签/输出"，而是**让扰动本身在通过网络时引发最大的神经元激活**（over-fire），从而污染中间特征表示，最终导致错误输出。
    
- **损失函数**：
    $$ Loss = -\log\left(\prod_{i=1}^{K}\|l_i(\delta)\|_2\right),\quad \text{s.t.}\ \|\delta\|_\infty \le \xi $$
    其中 $l_i(\delta)$ 是将纯扰动 $\delta$ 输入网络后第 $i$ 层的激活张量，$K$ 为参与优化的层数。
    
    - **为什么用乘积**：乘积形式迫使扰动在所有选定层同时产生强激活，比求和更有效。
        
    - **为什么取对数**：防止数值下溢。
        
- **优化过程**：
    
    - 随机初始化扰动 $\delta$。
        
    - 通过反向传播更新 $\delta$，使其最大化所有选定层输出的 $l_2$ 范数乘积。
        
    - 每次更新后投影至 $l_\infty$ 球内（像素值限制在$[-\xi,\xi]$)。
        
- **改进的自适应缩放**（相比会议版本）：
    
    - 当扰动像素饱和率（达到 $\pm \xi$ 的比例）增速低于阈值时，将扰动动态缩放到一半范围，避免梯度无效更新，提升优化效率。
        

---

### 3. 利用先验信息增强扰动（三种变体）

虽然核心目标是**完全数据无关**，但作者还探索了利用先验信息增强扰动的变体：

| 变体                    | 输入到网络的内容                                       | 说明                                  |
| --------------------- | ---------------------------------------------- | ----------------------------------- |
| **No Data (PNP)**     | 仅 $\delta$                                     | 纯数据无关，扰动本身作为网络输入                    |
| **Range Prior (PRP)** | $d+\delta$，其中 $d \sim \mathcal{N}(\mu,\sigma)$ | 使用高斯噪声模拟数据分布（均值为训练集均值，方差覆盖 [0,255]） |
| **Data Prior (PDP)**  | $x+\delta$，其中 $x$ 为真实样本                        | 利用少量真实训练样本，但不优化标签/输出，仍以激活最大化为目标     |

- **关键发现**：即使不优化标签，**仅提供数据分布的均值和动态范围（Range Prior）** 就能显著提升欺骗率（平均提升约 10%）；提供真实样本后提升更大（平均约 22%）。
    

---

### 4. 任务泛化性验证（三大任务）

#### （1）物体识别（分类）

- **目标模型**：CaffeNet、VGG-F/VGG-16/VGG-19、GoogLeNet、ResNet-152（ILSVRC 数据集）。
    
- **结果（表2）**：
    
    - **No Data**：GoogLeNet 欺骗率 58.6%，VGG-16 45.5%。
        
    - **Range Prior**：分别提升至 71.4% 和 63.1%。
        
    - **Data Prior**：分别提升至 83.5% 和 77.8%，**接近甚至超过**数据依赖方法 UAP [8]（GoogLeNet 78.5%，VGG-16 77.8%）。
        
- **与 UAP [8] 的对比（表8、图8）**：
    
    - UAP 依赖目标数据，当用 Places-205 数据替代 ILSVRC 数据时，CaffeNet 欺骗率从 93.1% 暴跌至 73.1%（本文方法不受影响）。
        
    - UAP 的欺骗率随可用样本量减少而急剧下降，而本文方法无需数据即可达到相当水平。
        
#### （2）语义分割

- **目标模型**：FCN-Alex、FCN-8s-VGG、DL-VGG、DL-RN101（Pascal VOC 数据集）。
    
- **评估指标**：广义欺骗率（GFR）基于 mean IOU。
    
- **结果（表3、表4）**：
    
    - **No Data**：FCN-Alex 的 GFR 达 80.2%，mean IOU 从 46.2 降至 15.35。
        
    - **Data w/ less BG**（筛选背景像素少于50%的样本）：GFR 达 89.6%，mean IOU 降至 8.03。
        
    - 即使与图像特定攻击 [20] 相比，本文的**图像无关**扰动也表现出强竞争力。
        

#### （3）单目深度估计（回归任务）

- **目标模型**：Monodepth-VGG 和 Monodepth-ResNet50（KITTI 数据集）。
    
- **评估**：使用 Abs Rel、Sq Rel、RMSE、δ<1.25δ<1.25 等多个指标，并引入 GFR 统一度量。
    
- **结果（表5、表6）**：
    
    - **Range Prior** 在 $\delta < 1.25$ 指标上 GFR 达 47.6%（ResNet50），**超过** Data Prior（24.3%）。
        
    - 但 Data Prior 在 Abs Rel 等误差指标上更严重（如 Sq Rel 从 1.148 升至 10.278，接近 10 倍）。
        
    - 表明不同指标下"欺骗"的定义不同，GFR 能更全面反映扰动效果。
        

---

### 5. 防御机制下的鲁棒性分析

#### （1）输入变换防御（表10、表11）

- 测试了高斯/中值/双边滤波、JPEG压缩、位深缩减、TV最小化、Image Quilting 等。
    
- **结果**：
    
    - 多数防御能降低欺骗率，但**以大幅牺牲正常精度为代价**（如双边滤波使 ResNet-152 精度下降 9.7%，欺骗率仍达 25.4%）。
        
    - JPEG(50%) 使 GoogLeNet 欺骗率从 71.4% 降至 41.8%，但精度仅降 5.2%，是一种较有效的折中。
        

#### （2）专用 UAP 防御（PRN，表10）

- PRN 是针对 UAP [8] 训练的去噪网络。
    
- **结果**：PRN 对 UAP [8] 的欺骗率降至 21.4%，但对本文 **Data Prior** 扰动仍高达 52.5%，说明 PRN **缺乏跨方法泛化能力**。
    

#### （3）鲁棒架构（散射网络 Hybrid-Net，表12）

- 散射网络在前端固定 Lipschitz 约束，理论上更稳定。
    
- **结果**：黑盒攻击下 Hybrid-Net 的欺骗率比 VGG-13/ResNet-18 **平均低约 13%**，但**仍被显著欺骗**（如 Range Prior 扰动从 ResNet-18 迁移到 Hybrid-Net 达 27.8%）。
    

---

### 6. 方法机理分析

- **特征污染程度（图9）**：随着网络加深，扰动引起的特征相对变化**单调递增**，最终积累到输出层导致误判。
    
- **非局部线性（图10）**：
    
    - 观察发现 $\| f(x+\delta)-f(x) \|_2$ 与 $\| f(\delta) \|_2$​ 存在**强相关性**，但并非相等。
        
    - 说明方法利用的是特征空间的**相关性**而非局部线性假设。
        
- **分类层输入偏移与欺骗率正相关（表13）**：
    
    - 相对偏移越大，欺骗率越高（No Data: 0.867 → 45.5%；Data Prior: 3.169 → 77.8%），验证了目标函数的有效性。
        

---

### 7. 论文贡献总结

|贡献|具体说明|
|---|---|
|**数据无关目标**|首次提出完全无需训练数据的 UAP 生成方法，仅凭随机初始化的扰动最大化中间层激活即可。|
|**任务通用性**|在分类、分割、深度估计三大任务（含分类与回归）上验证，无需修改目标函数。|
|**先验利用能力**|即使数据不可用，仅凭均值和动态范围（Range Prior）即可显著提升欺骗率；数据可用时效果逼近甚至超越数据依赖方法。|
|**广义欺骗率（GFR）**|提出与任务/指标无关的欺骗率定义，统一评估不同任务的扰动效果。|
|**防御评估**|系统评估了多种防御机制，发现现有防御对 GD-UAP 扰动效果有限或代价高昂。|
|**机理揭示**|通过特征偏移分析解释了为何"激活最大化"能跨任务、跨数据泛化。|

---

### 8. 总体评价

本文是**通用对抗扰动（UAP）领域的重要扩展**，将生成过程从"数据依赖+任务特定"推进到"数据无关+任务通用"，并系统分析了先验信息、防御鲁棒性和内在机理。其结果表明：**即使没有训练数据，深度模型仍能被高效欺骗**，对模型部署安全性提出了更高要求。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/通用与稀疏扰动/论文 - Universal Adversarial Perturbations|论文 - Universal Adversarial Perturbations]]：回顾图像无关的通用扰动问题。
- [[30_对抗机器学习/迁移与特征攻击/论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples|论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples]]：比较图像无关的共享扰动与跨模型迁移两个维度。
- [[30_对抗机器学习/通用与稀疏扰动/论文 - Data-Free Universal Adversarial Perturbation and Black-Box Attack|论文 - Data-Free Universal Adversarial Perturbation and Black-Box Attack]]：比较两篇无数据通用扰动工作的目标和数据假设。

## 论文出处

- [Generalizable Data-free Objective for Crafting Universal Adversarial Perturbations](https://doi.org/10.1109/TPAMI.2018.2861800)：IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)，2020，期刊论文。 2018 年已有在线版本，2020 年为正式卷期年份。
  [预印本](https://arxiv.org/abs/1801.08092)。
- [Universal Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html)：CVPR，2017，会议论文。

<!-- knowledge-links:end -->
