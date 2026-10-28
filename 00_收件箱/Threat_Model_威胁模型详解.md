# Threat Model（威胁模型）详解

在对抗机器学习（Adversarial Machine Learning）中，**Threat Model（威胁模型）**是理解一篇对抗攻击论文的基础。

在阅读 FGSM、PGD、C&W、SGM，以及 ViT Attack 相关论文时，经常会看到 White-box Threat Model、Black-box Threat Model、$L_p$-bounded Threat Model、Untargeted / Targeted Attack、Adaptive Attack 等概念。

> **Threat Model 不是具体的攻击算法，而是描述攻击者能力、目标和约束的整体框架。**

---

## 一、为什么需要 Threat Model？

设图像分类模型为 $f_\theta(x)$，正常预测为 $f_\theta(x)=y$。攻击者希望构造：

$$
x^{adv}=x+\delta
$$

使模型产生错误预测：

$$
f_\theta(x^{adv})\ne y
$$

但还需要回答：攻击者能否访问参数、梯度、训练数据？能够修改多少输入？是否有指定目标类别？可以查询目标模型多少次？

**威胁模型明确攻击的前提条件，帮助判断攻击是否现实，并公平比较不同方法。**

## 二、Threat Model 的基本组成

| 组成 | 关键问题 |
|---|---|
| Attacker Knowledge | 攻击者知道什么？ |
| Attacker Capability | 攻击者能做什么？ |
| Attacker Objective | 攻击者想实现什么？ |
| Attack Constraints | 攻击受到哪些限制？ |
| Attack Surface | 攻击发生在哪个环节？ |
| Success Criteria | 如何判断攻击成功？ |

## 三、Attacker Knowledge：攻击者的知识

### 1. White-box Attack（白盒攻击）

攻击者通常能够访问模型架构、参数 $\theta$、损失函数 $L$ 和输入梯度 $\nabla_x L$。

例如 FGSM：

$$
\boxed{x^{adv}=x+\epsilon\operatorname{sign}\left(\nabla_xL(f_\theta(x),y)\right)}
$$

标准 FGSM 需要直接计算目标模型的梯度，通常属于白盒梯度攻击。

**白盒不等于可以修改模型参数**：信息访问权限和修改权限是不同维度。

### 2. Black-box Attack（黑盒攻击）

攻击者无法直接获取目标模型内部参数或梯度。

| 类型 | 可获得的信息 | 代表方法 |
|---|---|---|
| Score-based | 输出概率或置信度 | NES、SPSA |
| Decision-based | 最终预测标签 | Boundary Attack、HopSkipJumpAttack |
| Transfer-based | 代理模型梯度 | MI-FGSM、SGM 等可用于迁移的方法 |
| Query-free Transfer | 不使用目标模型反馈 | 基于代理模型的迁移攻击 |

Score-based 和 Decision-based 通常依赖查询；Transfer-based 可以不查询目标模型。

### 3. 迁移攻击的特殊情况

设代理模型为 $f_s$，目标模型为 $f_t$。攻击者可获取 $\nabla_xL(f_s(x),y)$，但无法获取 $\nabla_xL(f_t(x),y)$。

因此对代理模型是**白盒**，对目标模型是**黑盒**：

$$
\boxed{\text{White-box Surrogate}\longrightarrow\text{Black-box Target}}
$$

SGM 所研究的迁移攻击通常属于这种设置。

## 四、Attacker Objective：攻击目标

### 1. Untargeted Attack（非定向攻击）

只要求预测错误：

$$
f(x^{adv})\ne y
$$

常见优化形式：

$$
\boxed{\max_\delta L(f_\theta(x+\delta),y)}
$$

其中 $\delta$ 还必须满足扰动约束。

### 2. Targeted Attack（定向攻击）

要求预测为指定目标类别 $y_{target}\ne y$：

$$
f(x^{adv})=y_{target}
$$

对交叉熵损失，常见优化形式为：

$$
\boxed{\min_\delta L(f_\theta(x+\delta),y_{target})}
$$

定向攻击要求更严格，但难度仍取决于具体模型和实验条件。

## 五、Attack Constraints：攻击约束

设 $x^{adv}=x+\delta$，常见扰动预算为：

$$
\boxed{\|\delta\|_p\le\epsilon}
$$

其中 $p$ 为范数类型，$\epsilon$ 为扰动预算。

### 1. $L_\infty$ 约束

$$
\|\delta\|_\infty=\max_i|\delta_i|\le\epsilon
$$

表示每个像素通道的扰动幅度均不超过 $\epsilon$。若图像归一化到 $[0,1]$，常见设置 $\epsilon=8/255\approx0.0314$。

在二维扰动空间中，可行域是**正方形**。

### 2. $L_2$ 约束

$$
\|\delta\|_2=\sqrt{\sum_i\delta_i^2}\le\epsilon
$$

限制整体扰动的欧氏长度，二维可行域是**圆形**。

### 3. $L_0$ 约束

严格来说 $L_0$ 不是真正的范数，通常表示非零扰动元素个数：

$$
\|\delta\|_0=|\{i:\delta_i\ne0\}|\le k
$$

即最多修改 $k$ 个元素。

### 4. $L_1$ 约束

$$
\|\delta\|_1=\sum_i|\delta_i|\le\epsilon
$$

二维可行域是**菱形**。

> 不同范数的 $\epsilon$ 不能直接比较数值大小来判断攻击强弱。

## 六、Attack Surface：攻击发生在哪个阶段？

| 阶段 | 常见攻击 | 主要内容 |
|---|---|---|
| 训练前或训练期间 | Data Poisoning | 污染训练数据 |
| 训练期间 | Backdoor Attack | 植入特定触发行为 |
| 推理阶段 | Evasion Attack | 修改测试输入误导模型 |
| 部署后 | Model Extraction | 通过接口提取模型信息 |
| 模型服务阶段 | Membership Inference | 推断样本是否属于训练集 |

这些分类并不完全互斥。FGSM、PGD、C&W、SGM 主要研究**推理阶段的逃逸攻击（Inference-time Evasion Attack）**。

## 七、如何完整描述一个 Threat Model？

以 ViT 黑盒迁移对抗攻击为例：

- **目标**：让目标 ViT 对图像错误分类。
- **知识**：可访问代理模型结构、参数和梯度，无法访问目标模型内部信息。
- **能力**：修改推理阶段输入图像像素。
- **约束**：$\|\delta\|_\infty\le8/255$。
- **类型**：非定向攻击。
- **查询**：不使用目标模型的查询反馈构造样本。

代理模型上的优化目标可写为：

$$
\boxed{
\begin{aligned}
\max_\delta\quad &L(f_s(x+\delta),y)\\
\text{s.t.}\quad &\|\delta\|_\infty\le\epsilon,\\
&x+\delta\in[0,1]^d
\end{aligned}}
$$

最终是否成功，仍需在目标模型上判断：

$$
\boxed{f_t(x+\delta)\ne y}
$$

**最大化代理模型损失不保证目标模型一定预测错误**，这正是迁移攻击要研究的问题。

## 八、Threat Model 与 Attack Method 的区别

| 维度 | Threat Model | Attack Method |
|---|---|---|
| 中文 | 威胁模型 | 攻击方法 |
| 回答的问题 | 在什么条件下攻击？ | 具体如何攻击？ |
| 内容 | 知识、权限、目标、约束 | 梯度、优化策略、算法 |
| 示例 | 黑盒、非定向、$L_\infty$ 约束 | FGSM、PGD、C&W、SGM |
| 作用 | 定义攻击场景 | 构造对抗样本 |

如果两篇论文使用不同范数、扰动预算或信息访问权限，**不能仅凭攻击成功率直接比较方法优劣**。

## 九、Adaptive Attack：自适应攻击

自适应攻击是指攻击者针对具体防御机制调整攻击策略。

例如：

- 随机化防御：考虑 EOT（Expectation Over Transformation）。
- 不可微预处理：考虑 BPDA（Backward Pass Differentiable Approximation）。
- 检测型防御：同时考虑分类错误与绕过检测。

**评估防御时，只测试未针对该防御调整的攻击，可能高估鲁棒性。**

Adaptive Attack 与 White-box Attack 不等价：前者描述是否针对防御调整策略，后者描述信息访问程度。

## 十、阅读 ViT Attack 论文时重点关注什么？

1. **攻击类型**：White-box、Black-box，还是 Transfer-based？
2. **攻击目标**：Targeted 还是 Untargeted？
3. **代理模型**：CNN、ViT，还是 Ensemble？
4. **扰动约束**：$L_\infty$、$L_2$ 或其他约束？$\epsilon$ 多大？
5. **目标模型**：CNN-to-CNN、CNN-to-ViT、ViT-to-ViT，还是其他跨架构迁移？
6. **评估方式**：数据集、成功率定义、查询预算和实验协议是什么？

例如：

$$
\text{ResNet-50}\rightarrow\text{ViT-B/16}
$$

属于跨架构迁移，而：

$$
\text{ViT-B/16}\rightarrow\text{ViT-L/16}
$$

属于同一架构家族内部的迁移，两者的研究意义和难度可能不同。

## 十一、总结

$$
\boxed{
\text{Threat Model}
=
\text{Knowledge}
+
\text{Capability}
+
\text{Objective}
+
\text{Constraints}
+
\text{Attack Surface}
}
$$

此外还必须明确**攻击成功的判定标准**。

> **Threat Model 决定攻击者能做什么、不能做什么，以及什么结果算成功；Attack Method 决定具体使用什么算法实现目标。**

对抗攻击研究中，应先理解 Threat Model，再分析 FGSM、PGD、C&W、SGM 等方法，这样才能判断论文解决的问题及其实验结论的适用范围。
