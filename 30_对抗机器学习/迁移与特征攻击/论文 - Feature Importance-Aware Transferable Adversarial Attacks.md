---
title: "论文 - Feature Importance-Aware Transferable Adversarial Attacks"
aliases:
  - "Feature Importance-aware Transferable Adversarial Attack论文"
分类: "迁移与特征攻击"
笔记类型: "论文阅读"
整理日期: "2026-10-27"
time: "2026-08-25T17:58:00"
tags:
  - "Transferability"
  - "FIA"
  - "特征攻击"
  - "知识库/论文阅读"
  - "主题/迁移与特征攻击"
状态: "已整理"
摘要: "围绕“论文 - Feature Importance-Aware Transferable Adversarial Attacks”，整理研究背景与核心问题、核心思想与创新点、关键技术细节。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击]]"
相关笔记:
  - "[[30_对抗机器学习/迁移与特征攻击/Feature Attack - 特征攻击发展历程]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Improving Adversarial Transferability via Neuron Attribution-Based Attacks]]"
  - "[[20_深度学习/网络结构/论文 - Learning Deep Features for Discriminative Localization]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima]]"
论文条目:
  - "[[论文来源目录#fia]]"
论文标题:
  - "Feature Importance-Aware Transferable Adversarial Attacks"
发表载体:
  - "ICCV"
发表年份:
  - "2021"
发表类型:
  - "会议论文"
发表链接:
  - "https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Feature_Importance-Aware_Transferable_Adversarial_Attacks_ICCV_2021_paper.html"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击|迁移与特征攻击总结]]

# 论文 - Feature Importance-Aware Transferable Adversarial Attacks




这篇论文《Feature Importance-Aware Transferable Adversarial Attacks》（ICCV 2021）针对**黑盒对抗攻击中的迁移性（Transferability）**问题，提出了一种名为**FIA（特征重要性感知攻击）**的新方法。以下是详细总结：

---

### 1. 研究背景与核心问题

- **对抗攻击**：在原始图像上添加微小扰动，使深度神经网络（DNN）产生错误分类。
    
- **黑盒攻击的两种主流方式**：
    
    - **查询攻击（Query-based）**：需反复访问目标模型，实际场景中代价高昂。
        
    - **迁移攻击（Transfer-based）**：在源模型上生成对抗样本，利用其跨模型迁移性攻击目标模型，更具实用性。
        
- **现有迁移攻击的痛点**：
    
    - 传统方法（如FGSM、BIM、MIM）和目标特征攻击（如NRDM、FDA）往往**无差别地扰动特征（Indiscriminate Distortion）**。
        
    - 由于不同模型会提取**特有的“噪声特征”（模型特异性特征）**来提高自身精度，现有方法会过度拟合源模型的这些特异性特征，导致生成的对抗样本**迁移性差**，陷入局部最优。
        

---

### 2. 核心思想与创新点

- **核心洞察**：要提高迁移性，应该攻击“**物体感知关键特征**”（Object-aware Features）——**这类特征对决策起主导作用，且在不同模型间是**通用的；同时抑制**模型特有的噪声特征**。
    
- **提出方法：特征重要性感知攻击（FIA）**：
    
    - 通过引入**聚合梯度（Aggregate Gradient）**，有效抑制模型特定噪声，同时保留物体感知的关键特征重要性。
        
    - 利用聚合梯度作为指导权重，**有区别地扰动特征**：抑制具有正权重的重要特征，增强具有负权重的次要特征，从而制造更大的跨模型决策混乱。
        

---

### 3. 关键技术细节

- **聚合梯度的计算**（抑制噪声，提取通用重要性）：
    
    - 对原始输入图像进行**多次随机像素丢弃（Random Pixel Dropping，伯努利掩码）**，得到多个变换版本。
        
    - 变换保留了空间结构和纹理，但破坏了非语义细节。
        
    - 计算这些变换版本相对于中间层特征图的梯度，并进行**平均**。
        
    - **效果**：物体感知特征的梯度在各变换中保持稳定（被保留），而模型特有的噪声梯度因变换而波动（被平均抵消）。可视化显示，聚合梯度比原始梯度更干净，且更聚焦于物体本身（见图3）。
        
- **损失函数设计**：
    
    - $\mathcal{L}(x^{adv}) = \sum \big(\Delta \odot f_k(x^{adv})\big)$
        
    - 其中 Δ是聚合梯度（作为特征重要性权重），$f_k$​ 是中间层特征图。
        
    - **优化目标**：最小化该损失，即**减少正权重区域的特征强度**（关键物体部分），**增加负权重区域的特征强度**（非关键/背景部分）。
        
- **优化求解**：采用带动量的迭代方法（MIM）来稳定更新并求解扰动 $\epsilon$。
    

---

### 4. 实验设置

- **数据集**：NIPS 2017 对抗竞赛使用的 ImageNet 兼容数据集（1000张图）。
    
- **目标模型**：涵盖12个SOTA模型。
    
    - **正常训练模型**：Inception-v3/v4、Inception-ResNet-v2、ResNet-50/152、VGG-16/19。
        
    - **防御模型（对抗训练）**：Adv-Inc-v3、Adv-IncRes-v2、Ens3/Ens4-Inc-v3、Ens-IncRes-v2。
        
- **对比基线**：包含MIM、DIM（输入多样性）、TIM（平移不变）、PIM（补丁攻击）及其组合，以及特征级攻击NRDM和FDA。
    

---

### 5. 主要实验结果

- **攻击正常训练模型（表1）**：
    
    - FIA在所有黑盒目标上均大幅领先。例如，以VGG-16为源模型时，FIA对Inc-v3的攻击成功率高达**95.7%**，而其他方法多在60%~80%之间。
        
    - 相比最优基线，FIA平均提升**9.5%**的成功率；结合PIDIM后，可再提升1%~10%。
        
    - 观察到模型复杂度规律：**较小的模型（如VGG-16）生成的对抗样本迁移性更强**，因为大模型搜索空间更大，更难避免局部最优。
        
- **攻击防御模型（表2）**：
    
    - FIA同样显著优于其他方法。例如以Inc-v3为源模型攻击Adv-Inc-v3时，FIA成功率为54.5%，而TIDIM/PITIDIM仅为40%左右。
        
    - 结合PITIDIM后，平均成功率再提升约**12.8%**。
        
- **集成模型攻击（表3）**：
    
    - 使用多模型集成生成对抗样本时，FIA依然最优，对Adv-Inc-v3成功率达90.9%。
        
- **消融研究与参数分析**：
    
    - **丢弃概率 pdpd​**：正常模型最优在0.2~0.3，防御模型在0.1；过大（如0.5）会破坏结构信息，降低攻击效果。
        
    - **集成数量 NN**：越大越好，但30时基本饱和。
        
    - **攻击层选择（图6）**：**中间层（Middle Layers）** 是最佳选择——浅层未学到语义，深层过于模型特定，而中间层既有良好的类别区分性又具备跨模型通用性。
        
- **聚合梯度有效性验证（图7与公式10）**：
    
    - 对比三种损失：无梯度引导（L1L1​）、原始梯度（L2L2​）、聚合梯度（L3L3​）。结果表明**聚合梯度（L3L3​）显著优于前两者**，证明了其对模型特定噪声的抑制和对通用重要特征的提取能力。
        

---

### 6. 论文贡献总结

|贡献点|具体说明|
|---|---|
|**问题分析**|指出现有方法迁移性差的根源是过度依赖**模型特有的噪声特征**，而非泛化的物体感知特征。|
|**方法创新**|提出**聚合梯度**提取特征重要性，并设计**感知特征重要性的损失函数**，引导扰动集中在决策关键区域。|
|**性能突破**|在正常模型和防御模型上均大幅超越SOTA，平均提升近10%~13%，可与其他方法（如PIDIM）轻松结合进一步提升。|
|**实用价值**|可作为评估模型鲁棒性的新基准，为后续防御研究提供更强攻击基线。|

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/迁移与特征攻击/Feature Attack - 特征攻击发展历程|Feature Attack - 特征攻击发展历程]]：回到特征攻击的研究脉络。
- [[30_对抗机器学习/迁移与特征攻击/论文 - Improving Adversarial Transferability via Neuron Attribution-Based Attacks|论文 - Improving Adversarial Transferability via Neuron Attribution-Based Attacks]]：比较特征重要性与神经元归因两条迁移攻击路线。
- [[20_深度学习/网络结构/论文 - Learning Deep Features for Discriminative Localization|论文 - Learning Deep Features for Discriminative Localization]]：联系特征定位、特征重要性与攻击目标的选择。
- [[30_对抗机器学习/迁移与特征攻击/论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima|论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima]]：比较损失几何优化与重要特征攻击。

## 论文出处

- [Feature Importance-Aware Transferable Adversarial Attacks](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Feature_Importance-Aware_Transferable_Adversarial_Attacks_ICCV_2021_paper.html)：ICCV，2021，会议论文。

<!-- knowledge-links:end -->
