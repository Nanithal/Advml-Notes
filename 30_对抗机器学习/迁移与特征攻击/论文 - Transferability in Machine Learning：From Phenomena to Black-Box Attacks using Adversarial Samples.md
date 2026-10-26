---
title: "论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples"
aliases:
  - "Transferability in Machine Learning：from Phenomena to  Black-Box Attacks using Adversarial Samples论文"
分类: "迁移与特征攻击"
笔记类型: "论文阅读"
整理日期: 2026-10-27
time: 2026-10-03T17:04:00
tags:
  - transferability
---

[[首页|首页]] · [[30_对抗机器学习/迁移与特征攻击/索引 - 迁移与特征攻击|迁移与特征攻击索引]]

# 论文 - Transferability in Machine Learning：From Phenomena to Black-Box Attacks using Adversarial Samples



这篇论文《Transferability in Machine Learning: from Phenomena to Black-Box Attacks using Adversarial Samples》（2016）由Papernot、McDaniel和Goodfellow撰写，系统研究了**对抗样本的迁移性（transferability）**，并将其应用于**黑盒攻击**。论文的核心贡献是：证明对抗样本不仅在同种机器学习技术训练的模型之间迁移，还能跨不同技术迁移；并提出利用**替代模型（substitute model）** 进行黑盒攻击的通用框架，成功攻击了Amazon和Google的商用机器学习服务。以下是全文总结。

---

## 1. 研究背景与核心问题

- **对抗样本**：对输入施加微小扰动，使机器学习模型产生错误输出。
    
- **迁移性（Transferability）**：为一个模型 $f$ 生成的对抗样本，往往也能欺骗另一个模型 $f′$，即使两者架构或训练集不同。
    
- **黑盒攻击**：攻击者不知道目标模型的结构、参数和训练数据，只能将其作为**Oracle**查询输入输出标签。
    
- **核心思路**：攻击者训练一个**替代模型**来近似目标模型，然后在替代模型上生成对抗样本，利用迁移性攻击目标模型。
    
- **本文目标**：
    
    1. 系统研究对抗样本在多种机器学习技术之间的迁移性。
        
    2. 改进替代模型的学习算法，使其更高效、更准确。
        
    3. 在实际商用平台上验证黑盒攻击的可行性。
        

---

## 2. 威胁模型与迁移性定义

### 2.1 对抗样本生成

给定模型 $f$ 和输入 $\vec{x}$ ，寻找最小扰动 $δ_{\vec{x}}$​，使 $f(\vec{x}+δ_{\vec{x}})≠f(\vec{x})$。

### 2.2 迁移性定义

$$\Omega_X(f,f') = \big|\{\,f'(\vec{x}) \neq f'(\vec{x}+\delta_{\vec{x}}): \vec{x}\in X\,\}\big|$$



即：为模型 ff 生成的对抗样本中，有多少也被模型 f′f′ 误分类。

- **同技术迁移性（Intra-technique）**：$f$ 和 $f′$ 用同种技术训练（如都是DNN）。
    
- **跨技术迁移性（Cross-technique）**：$f$ 和 $f′$ 用不同技术训练（如一个是DNN，一个是DT）。
    

### 2.3 两种假设

- **假设1**：同技术和跨技术迁移性在机器学习空间中普遍存在。
    
- **假设2**：黑盒攻击在实际场景中对任何未知机器学习分类器都可行。
    

---

## 3. 迁移性实验研究

### 3.1 实验设置

- **数据集**：MNIST手写数字（50,000训练，10,000验证，10,000测试）。
    
- **模型**：DNN、逻辑回归（LR）、支持向量机（SVM）、决策树（DT）、k近邻（kNN）、集成模型（Ens）。
    
- **模型特性**（表1）：
    
    - DNN：可微、非线性、非懒惰
        
    - LR：可微、线性、非懒惰
        
    - SVM：不可微、非线性、非懒惰
        
    - DT：不可微、非线性、非懒惰
        
    - kNN：不可微、非线性、懒惰
        

### 3.2 同技术迁移性

- 将训练集分成5份，每份训练一个模型（A-E），共25个模型。
    
- 对每个模型生成10,000个对抗样本，测试其他同技术模型的误分类率。
    
- **结果**（图2）：
    
    - **LR最脆弱**：迁移率 > 94%。
        
    - **DNN也较高**：迁移率 ≥ 49%。
        
    - **SVM、DT、kNN相对更鲁棒**：对角线突出，说明对抗样本在同技术内迁移性较弱。SVM的鲁棒性可能来自最大化间隔的约束；DT和kNN的鲁棒性可能来自不可微性。
        

### 3.3 跨技术迁移性

- 每种技术训练一个模型，另加一个5专家集成模型。
    
- 统一扰动幅度（L1范数约11.5%），生成对抗样本，测试跨技术迁移。
    
- **结果**（图3）：
    
    - **最脆弱**：决策树（DT），误分类率 47.20% ~ 89.29%。
        
    - **最鲁棒**：DNN，误分类率 0.82% ~ 38.27%。
        
    - **LR、SVM、kNN、集成模型**都表现出较强的跨技术迁移性。例如，为LR生成的对抗样本，SVM误分类91.43%，DT误分类87.42%。
        
    - **集成模型也不安全**：为LR生成的样本，集成模型误分类44.14%。
        

**结论**：对抗样本迁移性在机器学习空间中普遍存在，跨技术迁移性使得攻击者只需较少的模型知识即可发动攻击。

---

## 4. 替代模型学习（知识迁移）

### 4.1 核心思想

攻击者只有Oracle访问权限（能查询标签）。通过**Jacobian-based dataset augmentation**训练替代模型，使其逼近Oracle的决策边界。

### 4.2 Jacobian-based Dataset Augmentation

- 初始替代训练集 $S_0$​（100个样本），查询Oracle获取标签，训练初始替代模型 $f$。
    
- 迭代扩充数据集：
    
    $$S_{\rho+1} = \big\{\vec{x} + \lambda_\rho \cdot \operatorname{sgn}\big(J_f\big[\tilde O(\vec{x})\big]\big): \vec{x} \in S_\rho\big\} \cup S_\rho$$​
    
    其中 $J_f$​ 是替代模型的Jacobian，$\tilde{O}(\vec{x})$)是Oracle标签，$\lambda_p$ 是步长。
    
- 用扩充后的数据集训练新的替代模型，重复多轮。
    

### 4.3 两项改进

1. **周期性步长（Periodic Step Size, PSS）**：
    
    $$\lambda_\rho = \lambda \cdot (-1)^{\lfloor \rho/\tau \rfloor}$$
    
    每 $τ$ 轮翻转步长符号，提升替代模型质量。
    
2. **蓄水池采样（Reservoir Sampling, RS）**：
    
    - 每轮只随机选择 $κ$ 个样本进行扩充，避免查询次数指数增长。
        
    - 查询次数从 $n⋅2^ρ$ 降至 $n⋅2^σ+κ⋅(ρ−σ)$。
        

### 4.4 替代模型类型

- **DNN替代模型**：能近似DNN、LR、SVM、kNN等Oracle，匹配率约77%~83%；对DT Oracle匹配率仅48%（因DT不可微）。
    
- **LR替代模型**：对LR和SVM Oracle表现好，收敛更快，计算更轻量。
    
- **SVM替代模型**：只能近似SVM Oracle（匹配率79.80%），无法近似DNN/LR Oracle（匹配率仅约12%）。
    

### 4.5 实验结果（表2，图4）

- 周期性步长显著提升替代模型质量。例如DNN Oracle：vanilla 78.01% → PSS 89.28%。
    
- 蓄水池采样在减少查询次数的同时保持较好质量。例如DNN Oracle：PSS 89.28% → PSS+RS 82.90%，但查询次数从102,400降至3,600。
    

---

## 5. 对商用平台的黑盒攻击

### 5.1 攻击方法

1. 用少量查询（800次）训练替代模型（DNN或LR）。
    
2. 在替代模型上生成对抗样本（FGSM，$ϵ=0.3$）。
    
3. 将对抗样本提交给目标Oracle，测量误分类率。
    

### 5.2 Amazon Machine Learning

- 训练一个多分类模型，测试准确率92.17%。
    
- **攻击结果**（表3）：
    
    - LR替代模型，800次查询（$ρ=3$）：误分类率 **96.19%**。
        
    - DNN替代模型，800次查询：误分类率 87.44%。
        
    - 使用PSS+RS，2,000次查询：DNN替代模型达95.68%。
        
- **分析**：Amazon文档表明其使用“工业标准”多项逻辑回归，LR对对抗样本极度脆弱。
    

### 5.3 Google Cloud Prediction

- 训练模型测试准确率约92%。
    
- **攻击结果**（表4）：
    
    - LR替代模型，800次查询：误分类率 **88.94%**。
        
    - 使用PSS+RS，2,000次查询：LR替代模型达97.72%，DNN替代模型达91.57%。
        
- **防御尝试**：将对抗样本加入训练集重新训练，新Oracle仍被94.2%~100%的对抗样本欺骗，防御无效。
    

---

## 6. 对抗样本生成算法

### 6.1 DNN

- **FGSM**：
    
	$$\delta_{\vec{x}} = \epsilon \cdot \operatorname{sgn}\big(\nabla_{\vec{x}} c(f,\vec{x},y)\big)$$

### 6.2 多项逻辑回归（LR）

- 同样使用FGSM，但由于LR是线性的，FGSM是精确解。
    

### 6.3 k近邻（kNN）

- 使用平滑的soft-min替代argmin，使模型可微，再用FGSM。
    

### 6.4 多类SVM（新方法）

- 对每个二分类SVM $f_k$​，沿权重向量 $w[k]$ 的正交方向扰动：
    $$\vec{x}^* = \vec{x} - \epsilon \cdot \frac{w[k]}{\|w[k]\|}$$​
- 无需优化，计算高效。
    

### 6.5 决策树（新方法）

- **算法2**：从原始叶子出发，向上搜索祖先节点，找到兄弟子树中的不同类别叶子，修改样本特征以满足路径条件，使样本被分类到对抗叶子。
    
- 扰动通常只涉及极少数特征。
    

---

## 7. 结论与贡献

1. **迁移性普遍存在**：同技术和跨技术迁移性在机器学习空间中广泛存在，LR、SVM、DT、集成模型都易受攻击。
    
2. **替代模型学习**：DNN和LR均可有效学习替代模型，PSS和RS显著提升效率和质量。
    
3. **黑盒攻击可行**：仅用800次查询，即可攻击Amazon（96.19%）和Google（88.94%）的商用分类器。
    
4. **防御困难**：将对抗样本加入训练集无法有效防御。
    
5. **新攻击算法**：首次提出针对SVM和决策树的对抗样本生成方法。
    
6. **安全启示**：机器学习系统需要输入验证，类似SQL注入防御。
    

---

## 核心贡献总结

| 贡献         | 说明                                         |
| ---------- | ------------------------------------------ |
| **迁移性研究**  | 系统测量同技术和跨技术迁移性，覆盖DNN、LR、SVM、DT、kNN、集成模型。   |
| **替代模型学习** | 提出周期性步长和蓄水池采样，提升替代模型质量并降低查询次数。             |
| **黑盒攻击**   | 成功攻击Amazon和Google商用ML服务，误分类率96.19%和88.94%。 |
| **新攻击算法**  | 首次提出针对SVM和决策树的对抗样本生成方法。                    |
| **防御评估**   | 对抗训练对该攻击无效。                                |

---

## 笔记联系

- [[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum|论文 - Boosting Adversarial Attacks with Momentum]]：结合动量攻击论文理解跨模型迁移。
- [[30_对抗机器学习/迁移与特征攻击/Feature Attack - 特征攻击发展历程|Feature Attack - 特征攻击发展历程]]：从迁移现象延伸到特征层攻击路线。
- [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 文献综述与分类|ViT Attack - 文献综述与分类]]：将迁移问题延伸到 ViT 场景。
- [[50_汇报与写作/项目汇报/对抗妆容项目 - 汇报稿与答辩素材|对抗妆容项目 - 汇报稿与答辩素材]]：联系项目汇报中的黑盒迁移性评估。
- [[30_对抗机器学习/通用与稀疏扰动/论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations|论文 - Generalizable Data-Free Objective for Crafting Universal Adversarial Perturbations]]：比较图像无关的共享扰动与跨模型迁移两个维度。
