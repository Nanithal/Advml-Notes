如果你这里说的“**特征对抗攻击领域的特征攻击**”主要是指 **Feature-level Adversarial Attack / Feature Space Attack**——也就是不再局限于直接修改原始输入像素，而是**直接在模型学习到的特征空间、隐空间或语义空间中进行攻击**——那么这个方向的发展其实非常有代表性。

它可以概括成一条主线：

> **输入/像素空间攻击 → 特征空间攻击 → 深层语义特征攻击 → 特征重要性驱动攻击 → 多次特征扰动与迁移攻击 → 高层语义/结构化特征攻击 → 面向真实场景约束的特征攻击**

下面我按照**时间线 + 核心思想 + 数学形式 + 代表工作 + 存在问题 + 后续发展**给你系统梳理。

---

# 一、先明确：什么叫“特征攻击”？

传统对抗攻击的基本形式是：

$$
x' = x+\delta
$$

其中：

- $x$：原始输入
- $\delta$：攻击扰动
- $x'$：对抗样本
- $f(x)$：模型预测结果

攻击者通常要求：

$$
f(x')\neq f(x)
$$

同时限制：

$$
\|\delta\|_p\leq \epsilon
$$

例如图像领域最经典的思想是：

> **修改像素，但是尽可能不让人眼看出来。**

这形成了典型的：

$$
\boxed{\text{Pixel Space Attack}}
$$

但随着研究深入，研究者逐渐发现一个问题：

**像素并不是神经网络真正用于决策的核心变量。**

深度网络通常可以写成：

$$
x \overset{\phi_1}{\longrightarrow} z_1 \overset{\phi_2}{\longrightarrow} z_2 \overset{\phi_3}{\longrightarrow} \cdots \overset{\phi_L}{\longrightarrow} z_L \overset{g}{\longrightarrow} y
$$

其中：

$$
z_l=\phi_l(x)
$$

就是第 $l$ 层的**特征表示**。

因此产生了一个非常重要的研究问题：

> **如果直接攻击网络真正用于分类的 feature，而不是攻击 pixel，会发生什么？**

于是：

$$
\boxed{\text{Feature Space Adversarial Attack}}
$$

逐渐成为独立研究方向。

---

# 二、整个发展历程可以分成六个阶段

我建议你把这个领域记成下面这张“技术演进图”：

```
2013—2016
像素空间对抗攻击
      │
      ↓
2017—2019
发现深层特征具有跨模型一致性
      │
      ↓
2019
Feature Space Attack 正式成为重要攻击范式
      │
      ↓
2020—2021
从“攻击特征”进一步发展到“攻击语义/风格”
      │
      ↓
2021—2023
特征重要性、可解释性、迁移性成为核心
      │
      ↓
2024—2025
多次特征扰动、Feature→Pixel反演
      │
      ↓
2025—2026
高层语义、结构化特征、图特征、真实约束
```

下面详细展开。

---

# 三、第一阶段：像素空间攻击是特征攻击的起点

## 3.1 2013：Szegedy 等人发现 adversarial examples

现代对抗样本研究通常从 Szegedy 等人的工作开始。

核心发现是：

> 神经网络可以被非常小的输入扰动欺骗。

即：

$$
x' = x+\delta
$$

虽然：

$$
\|\delta\|\ll 1
$$

但是：

$$
f(x')\neq f(x)
$$

甚至：

$$
f(x')=y_{\mathrm{target}}
$$

这一阶段的攻击对象仍然是：

$$
\boxed{x}
$$

也就是原始输入。

---

# 四、第二阶段：FGSM、PGD——从“输入攻击”走向“梯度攻击”

随后出现 FGSM、PGD 等经典方法。

例如 FGSM：

$$
x'=x+\epsilon\operatorname{sign} \left( \nabla_x L(x,y) \right)
$$

PGD 则进行多次迭代：

$$
x_{t+1} = \Pi_{B_\epsilon(x)} \left[ x_t+ \alpha\operatorname{sign} (\nabla_xL(x_t,y)) \right]
$$

这类方法建立了后来特征攻击最重要的思想：

> **如果知道哪个方向能够最大程度改变模型的判别结果，就沿着这个方向攻击。**

只是早期的变量是：

$$
x
$$

后来研究者开始思考：

$$
\boxed{\text{为什么一定要攻击 }x？}
$$

---

# 五、第三阶段：2019——Feature Space Attack 真正形成

这是非常关键的转折点。

2019 年 CVPR 的工作：

**Feature Space Perturbations Yield More Transferable Adversarial Examples**

直接研究了：

> **Feature Space Perturbation**

作者 Inkawhich 等人发现，不同 CNN 虽然像素级决策边界可能差异很大，但是它们学习到的**高级特征表示存在明显的共同结构**。因此，在 feature space 中进行攻击，有可能得到更加具有迁移性的 adversarial examples。

---

# 六、Feature Space Attack 的核心数学思想

假设：

$$
z_l=f_l(x)
$$

是第 $l$ 层的 feature。

传统攻击：

$$
x' = x+\delta
$$

而 Feature Attack 思考的是：

$$
z_l'=z_l+\Delta z
$$

目标变成：

$$
f_l(x') \approx z_l+\Delta z
$$

然后通过优化找到一个输入 $x'$，使得：

$$
f_l(x')\rightarrow z_l^{target}
$$

例如：

$$
\min_{x'} D\left( f_l(x'), f_l(x_t) \right)
$$

其中：

- $x$：source image
- $x_t$：target image
- $f_l(x)$：source 的 feature
- $f_l(x_t)$：target 的 feature

这样攻击就从：

$$
\boxed{\text{pixel similarity}}
$$

转向：

$$
\boxed{\text{feature similarity}}
$$

---

# 七、为什么 Feature Space Attack 更有吸引力？

这是理解整个领域的关键。

传统像素攻击：

$$
\min \|x'-x\|
$$

关注的是：

> “两张图片像不像？”

Feature attack 更关注：

$$
\min \|f_l(x')-f_l(x_t)\|
$$

关注的是：

> “两个样本在神经网络眼里是不是变得像了？”

这两个概念完全不同。

---

# 八、一个非常重要的发现：特征比像素更“接近模型决策”

假设：

```
原图
 ↓
Pixel
 ↓
Edge
 ↓
Texture
 ↓
Shape
 ↓
Object
 ↓
Semantic
 ↓
Classification
```

越往后：

- 越抽象
- 越接近模型决策
- 越具有语义意义
- 越可能跨模型共享

所以：

$$
\boxed{ \text{Feature Space} \rightarrow \text{Decision Space} }
$$

这就是 Feature Attack 最核心的理论直觉。

---

# 九、第四阶段：从普通 Feature Attack 发展到 Style / Semantic Attack

2019 年之后，一个非常重要的问题出现：

> 如果我们攻击 feature，那么到底应该攻击什么 feature？

因为 feature 并不是一个简单的变量。

例如图像 CNN 中可能存在：

$$
z= [ z_{\text{color}}, z_{\text{texture}}, z_{\text{shape}}, z_{\text{object}}, z_{\text{semantic}} ]
$$

不同 feature 对分类的重要程度完全不同。

于是研究开始从：

> **Attack Feature**

逐渐走向：

> **Attack Specific Feature**

---

# 十、2021：Style Perturbation 是一个非常重要的节点

2021 AAAI 的：

**Towards Feature Space Adversarial Attack by Style Perturbation**

进一步提出：

> 不直接攻击像素，而是攻击图像的 style-related features。

例如：

- vivid colors
- sharp outlines
- texture/style-related representations

作者明确指出，这类攻击可以通过对抽象 feature 进行扰动来诱导模型错误分类，同时生成的样本可能比一些传统 pixel-space 攻击更加自然。

这说明研究重点已经发生变化：

```
Pixel Attack
      ↓
Feature Attack
      ↓
Specific Feature Attack
      ↓
Style Feature Attack
```

也就是从：

$$
x+\delta
$$

逐渐变成：

$$
z+\Delta z
$$

甚至：

$$
z_{\text{style}} + \Delta z_{\text{style}}
$$

---

# 十一、为什么 Style Attack 很重要？

因为它触及了一个很重要的问题：

## “不可感知性”到底应该怎么定义？

传统攻击：

$$
\|x'-x\|_p<\epsilon
$$

意味着：

> 像素变化小。

但是：

$$
\boxed{ \text{Pixel distance small} \neq \text{Semantic difference small} }
$$

例如：

```
原图：
一只猫

攻击后：
还是猫
但是颜色、纹理、轮廓发生变化
```

从：

$$
L_p(x,x')
$$

来看可能变化不大。

但从 feature space 来看：

$$
D(f(x),f(x'))
$$

可能发生非常大的变化。

所以 feature attack 开始重新定义：

$$
\boxed{\text{Imperceptibility}}
$$

---

# 十二、第五阶段：Feature Importance 成为核心问题

进一步的问题是：

> **不是所有 feature 都值得攻击。**

假设：

$$
z=[z_1,z_2,\cdots,z_d]
$$

模型输出：

$$
y=f(z)
$$

不同 $z_i$ 对结果影响不同：

$$
\left| \frac{\partial y}{\partial z_i} \right|
$$

可能差异巨大。

因此自然出现：

$$
I_i= \left| \frac{\partial L}{\partial z_i} \right|
$$

这样的 feature importance。

于是攻击策略变成：

$$
\Delta z_i \propto I_i
$$

即：

> **优先攻击重要特征。**

这和后来很多 feature attack、transfer attack、explainable attack 的思想直接相关。

---

# 十三、为什么“重要特征攻击”比随机特征攻击更合理？

举个简单例子。

假设一个分类器识别：

$$
\text{Dog vs Cat}
$$

特征：

|Feature|作用|
|---|---|
|毛发纹理|高|
|耳朵形状|高|
|眼睛|中|
|背景颜色|低|
|背景亮度|低|

随机攻击：

$$
z+\Delta z
$$

可能大量修改：

$$
z_{\text{background}}
$$

结果：

> 改了很多东西，但模型没被骗。

Feature importance attack 则选择：

$$
z_{\text{ear}} ,\quad z_{\text{fur}}
$$

因此：

$$
\boxed{ \text{Attack efficiency}\uparrow }
$$

---

# 十四、第六阶段：Feature Attack 和 Transferability 开始深度结合

这是整个方向非常重要的一个发展。

传统 adversarial examples 最大的问题之一：

> 在 source model 上成功，不一定能攻击 target model。

即：

$$
f_s(x')\neq y
$$

但是：

$$
f_t(x')=y
$$

于是：

$$
\boxed{\text{Transferability}}
$$

成为核心问题。

2019 CVPR 的 Feature Space Perturbations 工作已经明确把 feature-space attack 与 transferability 联系起来，并报告了很强的 targeted transfer performance。

---

# 十五、为什么 Feature Attack 可能更容易迁移？

假设两个模型：

$$
f_A(x) 
$$
$$
f_B(x)
$$
虽然：

$$
f_A^{pixel} \neq f_B^{pixel}
$$

但是它们可能学习了类似的高层表示：

$$
z_A\approx z_B
$$

比如两个 ImageNet 分类器都必须学习：

- edge
- texture
- shape
- object parts
- semantic object representation

因此：

$$
\boxed{ \text{Shared Feature} \rightarrow \text{Shared Vulnerability} }
$$

这成为 Feature Attack 的重要理论基础。

---

# 十六、但是这里出现了一个新的问题

如果 feature space 真这么好：

> 为什么不一直在 feature space 里攻击？

因为最终我们需要一个：

$$
x'
$$

而不是一个：

$$
z'
$$

也就是说：

$$
z' \rightarrow x'
$$

并不是简单的。

这实际上是 Feature Attack 长期以来的核心难题：

$$
\boxed{ \text{Feature Manipulation} \rightarrow \text{Adversarial Input Reconstruction} }
$$

---

# 十七、因此出现了 Feature Inversion 问题

假设：

$$
z'=f_l(x)+\Delta z
$$

我们希望找到：

$$
x'
$$

满足：

$$
f_l(x')\approx z'
$$

于是可以写成：

$$
\min_{x'} \left\| f_l(x')-z' \right\|_2^2
$$

这实际上就是一个 feature inversion / feature matching 问题。

因此：

```
Feature Attack
     ↓
Feature Perturbation
     ↓
Feature Inversion
     ↓
Adversarial Image
```

逐渐成为一个完整 pipeline。

---

# 十八、2025：P2FA 是一个非常值得关注的阶段

2025 ICML 的：

**Pixel2Feature Attack (P2FA): Rethinking the Perturbed Space to Enhance Adversarial Transferability**

进一步重新思考：

> 到底应该在哪里进行多次扰动？

论文指出，过去一些 feature-level attacks 从整体效果来看，实际上可能等价于在 feature importance 指引下进行“一次性的” feature-space 攻击，而没有充分利用多次 feature-space perturbation 的能力。

P2FA 的核心思路就是：

$$
\boxed{ Pixel \rightarrow Feature \rightarrow Repeated Feature Perturbation \rightarrow Pixel }
$$

即：

1. 从 pixel 得到 feature
2. 进入 feature space
3. **多次**扰动 feature
4. 再将 feature 反演回 pixel
5. 得到 adversarial example

这实际上是对早期 Feature Attack 的一个重要升级。

---

# 十九、P2FA 为什么重要？

早期：

$$
z'=z+\Delta z
$$

现在变成：

$$
z_0=z
$$
$$
z_1=z_0+\Delta z_1
$$
$$
z_2=z_1+\Delta z_2
$$
$$\cdots
$$
$$
z_T=z_{T-1}+\Delta z_T
$$

即：

$$
\boxed{ z_T=z_0+\sum_{t=1}^{T}\Delta z_t }
$$

这样攻击者可以不断：

> 找到最重要的 feature → 修改 → 重新计算 importance → 再攻击。

这比一次性修改更灵活。

---

# 二十、Feature Attack 现在已经开始从图像走向其他领域

这是你如果做论文调研特别需要注意的。

“Feature Attack”现在已经不是一个单一的计算机视觉概念。

它已经进入：

### 1. 图像

$$
\text{Image Feature Attack}
$$

---

### 2. 图神经网络

$$
\text{Node Feature Attack}
$$

例如：

$$
X' = X+\Delta X
$$

同时还可能攻击：

$$
A' = A+\Delta A
$$

因此：

$$
\boxed{ \text{Structure Attack} + \text{Feature Attack} }
$$

成为 GNN 对抗攻击的重要组合。

已有工作明确研究了同时修改 graph structure 和 node features 的攻击框架。

---

# 二十一、GNN 中的 Feature Attack 有什么不同？

图数据：

$$
G=(V,E,X)
$$

其中：

- $V$：节点
- $E$：边
- $X$：节点特征

传统结构攻击：

$$
E\rightarrow E'
$$

而 feature attack：

$$
X\rightarrow X'
$$

例如：

```
Node A
Feature:
[0.2, 0.5, 0.8, 0.1]

        ↓ attack

[0.2, 0.9, 0.8, 0.1]
```

但是 GNN 有一个特殊性质：

$$
h_i^{(l+1)} = AGG \left( h_i^{(l)}, \{h_j^{(l)}:j\in N(i)\} \right)
$$

因此攻击一个节点 feature：

$$
x_i
$$

可能通过 message passing 影响：

$$
h_i \rightarrow h_j \rightarrow h_k
$$

于是 feature attack 具有：

$$
\boxed{\text{Local perturbation → Global representation change}}
$$

这一特点。

---

# 二十二、2024：Feature-level GNN Attack 开始系统化

例如 2024 年的 Two-level Adversarial Attacks for GNNs，把：

$$
\text{task-level attack}
$$

和：

$$
\text{feature-level attack}
$$

结合起来，并通过 contrastive learning 对 learned features 进行攻击。论文认为，这种 feature-level attack 可以破坏 GNN 学到的 representation 和 discriminative ability。

这说明：

> Feature Attack 已经从“修改输入 feature”进一步发展到“破坏模型内部 representation”。

---

# 二十三、这是一个非常重要的概念升级

早期：

$$
\boxed{\text{Feature Attack}}
$$

更多是：

> 改变 feature value。

现在逐渐变成：

$$
\boxed{\text{Representation Attack}}
$$

也就是：

> **直接破坏模型内部形成的表示结构。**

这是两个不同层次。

---

# 二十四、从 Feature Attack 到 Representation Attack

可以画成：

```
Input
  │
  ↓
Raw Feature
  │
  ↓
Low-level Feature
  │
  ↓
Mid-level Feature
  │
  ↓
High-level Feature
  │
  ↓
Semantic Representation
  │
  ↓
Decision
```

攻击层次也越来越深：

```
Input Attack
      ↓
Feature Attack
      ↓
Deep Feature Attack
      ↓
Representation Attack
      ↓
Semantic Attack
      ↓
Decision Boundary Attack
```

---

# 二十五、2025—2026：开始攻击“高层语义”

这是目前非常明显的发展趋势。

例如图数据领域近期已经开始研究：

**High-level Semantics Attack**

其基本思想不是简单改变：

$$
x_i
$$

而是改变：

$$
\boxed{\text{High-level semantic representation}}
$$

同时尽量保持原始数据的主要语义不被破坏。近期工作 AHSG 就沿着这个方向研究 GNN 中的高层语义攻击。

这意味着：

> Feature Attack 正在从“数值扰动”向“语义操纵”发展。

---

# 二十六、Feature Attack 的另一个重要分支：Tabular Feature Attack

如果你研究的是“特征攻击”而不是狭义 CV Feature Space Attack，那么**表格数据（tabular data）**必须单独拿出来。

因为：

$$
X= [ x_1,x_2,\cdots,x_d ]
$$

每个 $x_i$ 本身就是一个明确的 feature。

例如：

```
Age
Income
Debt
Credit score
Employment
Loan amount
```

攻击者可能希望：

$$
x_{\mathrm{income}} \rightarrow x_{\mathrm{income}}+\delta
$$

使：

$$
f(X)=\text{Reject}
$$

变成：

$$
f(X')=\text{Approve}
$$

---

# 二十七、Tabular Feature Attack 比图像 Feature Attack 更难

因为图像：

$$
x_i\in [0,255]
$$

很多时候可以连续扰动。

但表格数据：

$$
X= [ \text{Age}, \text{Gender}, \text{Income}, \text{Occupation} ]
$$

存在：

### 连续变量

$$
Income=50000
$$

### 离散变量

$$
Gender\in\{M,F\}
$$

### 类别变量

$$
Occupation\in\{Engineer,Teacher,\cdots\}
$$

### 不可修改变量

例如：

$$
ID,\ DateOfBirth
$$

因此攻击必须满足：

$$
\boxed{ \text{Validity} + \text{Feasibility} + \text{Consistency} + \text{Attack Success} }
$$

近期综述和研究特别强调了这些约束，以及特征之间相关性和数据一致性的问题。

---

# 二十八、因此 Feature Attack 开始从“数学扰动”走向“现实约束”

早期：

$$
\min_\delta \|\delta\|
$$

现在则越来越接近：

$$
\min_{x'} D(x,x')
$$

subject to：

$$
f(x')\neq f(x)
$$

同时：

$$
x'\in\mathcal X_{\mathrm{valid}}
$$

并且：

$$
C(x')=1
$$

其中 CC 表示：

> feature consistency / domain constraint。

这实际上是一个很大的范式变化。

---

# 二十九、所以整个领域可以总结成三个阶段

如果你以后写论文 Related Work，我建议用这个框架。

## 第一阶段：Pixel-level

核心：

$$
x'=x+\delta
$$

目标：

$$
\min \|\delta\|
$$

关注：

- attack success
- $L_2$
- $L_\infty$
- imperceptibility

---

## 第二阶段：Feature-level

核心：

$$
z'=z+\Delta z
$$

其中：

$$
z=f_l(x)
$$

目标：

$$
\min D(f_l(x'),z')
$$

关注：

- feature importance
- transferability
- semantic representation
- feature inversion

---

## 第三阶段：Semantic / Representation-level

核心：

$$
h'=h+\Delta h
$$

其中：

$$
h=\Phi(x)
$$

目标不再只是：

> 修改一个 feature。

而是：

> **改变模型内部 representation，同时保持输入的自然性和语义一致性。**

关注：

- high-level semantics
- representation geometry
- transferability
- naturalness
- interpretability
- robustness

---

# 三十、整个发展历程可以浓缩成一张表

| 时期        | 攻击空间                | 核心问题                        | 代表思想                              |
| --------- | ------------------- | --------------------------- | --------------------------------- |
| 2013–2016 | Pixel               | 小扰动能否欺骗模型                   | $x+\delta$                        |
| 2017–2018 | Pixel/Gradient      | 如何提高攻击效率                    | FGSM/PGD 等                        |
| 2019      | Feature             | 为什么 feature 更容易迁移           | Feature Space Attack              |
| 2020–2021 | Feature/Style       | 攻击什么 feature                | Style/semantic feature            |
| 2021–2023 | Important Feature   | 哪些 feature 最重要              | Feature importance                |
| 2022–2024 | Representation      | 如何破坏 learned representation | contrastive/representation attack |
| 2023–2024 | Graph Feature       | 节点 feature 如何攻击 GNN         | $X\rightarrow X'$                 |
| 2024–2025 | Feature + Structure | feature 与 structure 联合攻击    | $X,A joint attack$                |
| 2025      | Feature iterative   | 如何充分利用 feature space        | P2FA                              |
| 2025–2026 | Semantic            | 如何攻击高层语义                    | semantic representation attack    |
| 当前        | Real-world Feature  | 如何保证自然性与约束                  | validity/coherence/feasibility    |

---

# 三十一、这个领域真正的核心矛盾是什么？

如果你准备深入研究，我认为最重要的不是记论文，而是理解下面这个矛盾：

$$
\boxed{ \text{Attack Effectiveness} \quad vs \quad \text{Naturalness} }
$$

攻击越强：

$$
\Delta z\uparrow
$$

越容易改变：

$$
f(x)
$$

但是：

$$
\Delta z\uparrow
$$

也可能导致：

$$
x'\notin \mathcal X_{\mathrm{natural}}
$$

因此真正的问题变成：

$$
\boxed{ \max \text{Attack Success} }
$$

同时：

$$
\boxed{ \min \text{Semantic / Perceptual / Feature Distortion} }
$$

这实际上就是未来 Feature Attack 研究的核心优化矛盾。

---

# 三十二、从研究角度看，未来可以分成 5 条路线

## 路线 1：更好的 Feature Importance

现在很多方法仍然依赖：

$$
\frac{\partial L}{\partial z_i}
$$

但 gradient importance 不一定真正等价于：

> causal importance。

因此未来很可能从：

$$
\text{Gradient Importance}
$$

发展到：

$$
\text{Causal Feature Importance}
$$

---

## 路线 2：从 feature attack 到 semantic attack

未来攻击目标可能从：

$$
z_i
$$

变成：

$$
\text{semantic concept}
$$

例如：

```
“dog”
 ↓
“fur”
 ↓
“shape”
 ↓
“ear structure”
```

攻击者直接控制：

$$
\text{Concept Representation}
$$

这比简单修改 feature vector 更有解释性。

---

## 路线 3：Feature Attack + Generative Model

这是一个非常值得关注的方向。

可以让：

$$
z\rightarrow z'
$$

然后通过：

$$
G(z')\rightarrow x'
$$

生成自然的 adversarial example。

于是：

$$
\boxed{ Feature Attack + Generative Model }
$$

就能够解决：

> feature perturbation 如何映射回自然输入？

---

## 路线 4：Feature Attack + Black-box

现实中攻击者往往不知道：

$$
f_l(x)
$$

只能得到：

$$
f(x)
$$

因此：

$$
\boxed{ Feature Attack + Black-box }
$$

会越来越重要。

2026 年的研究已经明显把注意力扩展到不同攻击者知识、黑盒查询和现实场景约束等问题。

---

# 三十三、我认为你最应该记住的一条“技术演进主线”

如果你后面是准备做**论文、开题或者 Feature Attack 方向研究**，我建议把整个历史理解成：

$$
\boxed{ \text{Pixel} \rightarrow \text{Feature} \rightarrow \text{Important Feature} \rightarrow \text{Representation} \rightarrow \text{Semantic} }
$$

对应的优化目标则是：

$$
\boxed{ \min \|\delta_x\| }
$$

↓

$$
\boxed{ \min D(z',z) }
$$

↓

$$
\boxed{ \min D(z',z) \quad s.t.\quad I(z)\text{ high} }
$$

↓

$$
\boxed{ \min D(h',h) \quad \text{s.t.}\quad f(h')\neq f(h) }
$$

↓

$$
\boxed{ \max \text{semantic decision shift} \quad s.t. \quad \text{naturalness} }
$$

也就是说：

> **早期研究问的是“改多少像素可以骗过模型？”**

后来变成：

> **“改哪些 feature 可以骗过模型？”**

再后来变成：

> **“怎样改变模型内部 representation？”**

现在越来越关注：

> **“能不能直接改变模型对高层语义的理解，同时让输入看起来仍然合理？”**

这就是**特征对抗攻击从 Pixel Space → Feature Space → Representation Space → Semantic Space 的核心发展逻辑**。

---

### 如果你是准备做这个方向的科研，我建议下一步这样梳理

尤其结合你之前在做 **SVRG、GDUAP、ImageNet 和对抗攻击代码**，如果你现在是在准备一个具体的 **Feature Attack 研究课题**，单纯看历史还不够。更值得做的是把 **2019–2026 年 Feature Attack 的论文按照“攻击空间—特征提取层—优化目标—白盒/黑盒—是否需要 surrogate model—迁移性—扰动约束—代码”做成一张论文谱系表**。

我可以进一步给你整理成一份类似下面这样的**科研级文献地图**：

```
Feature Attack
│
├── 1. Feature Space Attack
│   ├── Feature Matching
│   ├── Feature Disruption
│   └── Feature Inversion
│
├── 2. Feature Importance Attack
│   ├── Gradient-based
│   ├── Attention-based
│   └── Attribution-based
│
├── 3. Style / Semantic Feature Attack
│   ├── Style
│   ├── Texture
│   └── High-level semantics
│
├── 4. Transferable Feature Attack
│   ├── Black-box
│   ├── Cross-model
│   └── Cross-architecture
│
├── 5. Representation Attack
│   ├── Contrastive
│   ├── Embedding
│   └── Latent-space
│
├── 6. Structured Feature Attack
│   ├── Tabular
│   ├── Graph Node Feature
│   └── Multimodal
│
└── 7. Future
    ├── Generative Feature Attack
    ├── Semantic Control
    ├── Causal Feature Attack
    └── Black-box Feature Attack
```

如果你的目标是**写论文/开题报告**，下一步最有价值的是把这条线进一步扩成**“2013–2026 年代表论文时间轴 + 每篇论文的核心公式 + 优缺点 + 是否有 GitHub 代码 + 常用数据集（ImageNet/CIFAR-10 等）”**，这样就可以直接作为 Related Work 和研究现状的框架。