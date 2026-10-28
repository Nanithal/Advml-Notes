
# 论文总结：On the Robustness of Vision Transformers to Adversarial Examples (ICCV 2021)

**作者**：Kaleel Mahmood, Rigel Mahmood, Marten van Dijk  
**发表**：ICCV 2021

---

## 1. 研究背景与核心问题

- **Vision Transformer (ViT)** 在图像分类任务上取得了与 CNN 相当甚至更好的性能。
- CNN 的对抗鲁棒性已被广泛研究，但 **ViT 在对抗环境下的安全性尚未被系统研究**。
- 本文围绕三个核心问题展开：
  1. **白盒攻击下，ViT 是否比 CNN 更安全？**
  2. **对抗样本在 ViT 与其他模型（如 CNN、BiT）之间的迁移性如何？**
  3. **能否利用迁移性现象设计安全的防御？**

---

## 2. 白盒攻击分析

### 2.1 实验设置
- **数据集**：CIFAR-10、CIFAR-100、ImageNet
- **扰动约束**：\( \ell_\infty \) 范数，CIFAR 上 \( \epsilon = 0.031 \)，ImageNet 上 \( \epsilon = 0.062 \)
- **攻击方法（6种白盒攻击）**：
  - FGSM
  - PGD
  - BPDA（用于检测梯度掩码）
  - MIM
  - C&W
  - APGD
- **模型（8个 CIFAR 模型 + 7个 ImageNet 模型）**：
  - ViT-B-32、ViT-B-16、ViT-L-16、ViT-R50
  - BiT-M-R50x1、BiT-M-R101x3（ImageNet 用 BiT-M-R152x4）
  - ResNet-56、ResNet-164（ImageNet 用 ResNet-50、ResNet-152）

### 2.2 主要结论
- **ViT 在白盒攻击下并不比 CNN 更安全**。
- 在 PGD、MIM、C&W、APGD 攻击下，所有模型的鲁棒准确率几乎为 **0%**。
- FGSM 下略有差异，但整体趋势一致：**ViT 没有提供额外的白盒安全性**。
- 排除了梯度掩码的可能性（BPDA 攻击成功）。

---

## 3. 迁移性研究

### 3.1 迁移性定义
\[
t_{i,j} = \frac{1}{m}\sum_{k=1}^m \begin{cases} 
1 & \text{if } C_j(A_{C_i}(x_k, y_k)) \neq y_k \\
0 & \text{otherwise}
\end{cases}
\]
- \( C_i \)：生成对抗样本的源模型
- \( C_j \)：评估对抗样本的目标模型
- 对每对模型，测试 FGSM、PGD、MIM 三种攻击，取最高迁移率。

### 3.2 核心发现：跨模型属（Cross-genus）迁移性极低
- **模型属（Genus）** 分为三类：ViT、BiT、ResNet。
- **同属内迁移性高**（如 ViT→ViT、BiT→BiT、ResNet→ResNet）。
- **跨属迁移性显著低**：
  - 例：BiT-50x1 生成的对抗样本，ViT-L-16 仅误分类 **5.7%**（CIFAR-10）。
  - 例：ViT-L-16 生成的对抗样本，BiT-50x1 仅误分类 **42.5%**（CIFAR-10）。
  - 例：ViT-L-16 生成的对抗样本，ResNet-56 仅误分类 **13.0%**（CIFAR-10）。

### 3.3 可视化（图1）
- 3D 柱状图显示：**粉色/红色/橙色柱（同属）高，绿色/蓝色/浅蓝色柱（跨属）低**。
- 趋势在 CIFAR-10、CIFAR-100、ImageNet 上一致。

---

## 4. 白盒安全与 SAGA 攻击

### 4.1 集成防御的动机
- 基于低跨属迁移性，一个自然的防御是**集成不同属的模型**（如 ViT + BiT）。
- 采用**随机选择**输出（随机选一个模型预测），避免多数投票的“递减回报”和绝对共识的“干净准确率损失”。

### 4.2 SAGA（Self-Attention Gradient Attack）
- **动机**：朴素分析认为集成能提升鲁棒性，但攻击者可以同时攻击多个模型。
- **攻击公式**：
$$
  [
  x_{adv}^{(i+1)} = x_{adv}^{(i)} + \epsilon_s \cdot \text{sign}(G_{\text{blend}}(x_{adv}^{(i)}))
  ]
$$
$$
  [
  G_{\text{blend}} = \sum_{k \in K} \alpha_k \frac{\partial L_k}{\partial x_{adv}} + \sum_{v \in V} \alpha_v \phi_v \odot \frac{\partial L_v}{\partial x_{adv}}
  ]
$$
  - \( K \)：CNN 集合；\( V \)：ViT 集合
  - $( \phi_v )$：**自注意力图**（attention rollout），用于加权 ViT 的梯度
- **注意力图计算**：
$$
  [
  \phi_v = \left( \prod_{l=1}^{n_l} \left[ \sum_{i=1}^{n_h} (0.5 W_{l,i}^{(att)} + 0.5 I) \right] \right) \odot x
  ]
$$

### 4.3 实验结果（图2）
- SAGA 在集成模型上的攻击成功率：
  - CIFAR-10：**74.0%**
  - CIFAR-100：**84.4%**
  - ImageNet：**91.8%**
- **SAGA 优于单模型 PGD/MIM 和 Basic 攻击**。
- **结论**：集成防御在**白盒攻击下不安全**。

---

## 5. 黑盒安全与迁移性防御

### 5.1 黑盒攻击设置
- **查询型攻击**：RayS（10,000 次查询/样本）
- **迁移型攻击**：Adaptive Black-Box Attack（使用 100% 训练数据，训练替代模型 ViT-B-32）

### 5.2 实验结果（图3）
- **集成模型（ViT + BiT）在黑盒攻击下显著提升鲁棒性**：
  - RayS CIFAR-10：鲁棒准确率 **81.0%**（单 ViT 仅 14.5%，单 ResNet 仅 0.8%）
  - RayS CIFAR-100：**82.0%**（单 ViT 9.0%）
  - RayS ImageNet：**58.0%**（单 ViT 25.9%）
  - Adaptive CIFAR-10（100% 数据）：**57.0%**（单 ViT 30.8%）
- **干净准确率未受损**：CIFAR-10 98.2%，CIFAR-100 92.83%，ImageNet 85.37%。

### 5.3 核心结论
- **跨属低迁移性可以被利用来构建黑盒鲁棒的集成防御**。
- 在白盒下不安全，但在黑盒下能实现**前所未有的鲁棒性**。

---

## 6. 贡献总结

| 贡献 | 说明 |
| :--- | :--- |
| **白盒鲁棒性分析** | ViT 在白盒攻击下与 CNN 一样脆弱，无额外安全性。 |
| **迁移性研究** | 首次大规模研究 ViT 与 CNN/BiT 之间的迁移性，发现跨属迁移性极低。 |
| **SAGA 攻击** | 提出结合自注意力图和多模型梯度的白盒攻击，证明集成防御在白盒下不安全。 |
| **黑盒防御** | 利用低跨属迁移性，ViT+BiT 集成在黑盒攻击下实现高鲁棒性且不损失干净准确率。 |
| **安全启示** | ViT 的引入为对抗鲁棒性提供了新思路：跨架构集成可用于黑盒防御。 |

---

## 7. 结论

- Vision Transformer 本身并不比 CNN 更鲁棒。
- 但 ViT 与 CNN/BiT 之间的**低对抗迁移性**是一个可利用的安全特性。
- 在白盒攻击下，集成防御可被 SAGA 攻破。
- 在黑盒攻击下，ViT+BiT 集成能实现**近乎 SOTA 的干净准确率 + 显著的鲁棒性提升**。
- 本文为对抗机器学习领域提供了关于 Vision Transformer 安全性的首个系统性研究。