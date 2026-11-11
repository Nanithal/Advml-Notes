---
title: "Integrated Gradients - 积分梯度详解"
aliases:
  - "IG"
  - "Integrated Gradients"
  - "积分梯度"
tags:
  - "知识库/概念笔记"
  - "主题/模型可解释性"
  - "深度学习/特征归因"
笔记类型: "概念笔记"
分类: "训练基础"
状态: "已整理"
摘要: "从梯度饱和出发，推导积分梯度的路径积分、完备性与基线依赖，结合数值例子解释符号、交互、归一化坐标、Captum 实现，以及与 SIM 和对抗攻击的区别。"
更新日期: "2026-11-11"
上级目录: "[[20_深度学习/训练基础/总结 - 训练基础]]"
相关笔记:
  - "[[20_深度学习/训练基础/机器学习 - 函数拟合与激活函数]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Improving Adversarial Transferability via Neuron Attribution-Based Attacks]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Feature Importance-Aware Transferable Adversarial Attacks]]"
  - "[[30_对抗机器学习/迁移与特征攻击/SIM - 尺度不变攻击详解]]"
  - "[[30_对抗机器学习/迁移与特征攻击/VMI - 方差调节动量攻击详解]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程]]"
  - "[[20_深度学习/训练基础/Feature - 特征详解]]"
论文条目:
  - "[[论文来源目录#ig]]"
论文标题:
  - "Axiomatic Attribution for Deep Networks"
发表载体:
  - "ICML"
发表年份:
  - "2017"
发表类型:
  - "会议论文"
发表链接:
  - "https://proceedings.mlr.press/v70/sundararajan17a.html"
预印本链接:
  - "https://arxiv.org/abs/1703.01365"
来源核实日期: "2026-11-11"
---

[[首页|首页]] · [[20_深度学习/训练基础/总结 - 训练基础|训练基础总结]]

# Integrated Gradients：积分梯度详解

> 已于 2026-11-11 按用户确认归档到“训练基础”目录，并关联主题总结、NAA 笔记及 [[论文来源目录#ig]]。

## 一、IG 在回答什么问题？

**Integrated Gradients（IG，积分梯度）是一种特征归因方法：相对于选定的基线输入，当前输入的各个维度分别分担了多少输出分数变化？**

例如，模型把一张图片识别为猫。选定“猫”的输出分数后，IG 可以给每个像素、每个颜色通道分配一个数值，解释该分数从基线到这张图片的变化。

它的基本对象包括：

- 当前输入 $x$，例如图片张量。
- 基线 $b$，与 $x$ 使用相同的坐标和形状。
- 标量函数 $F$，即希望解释的模型输出。

分类任务中，可以选固定类别的 logit，也可以选该类别的概率；回归任务中可以选某个预测值。二者的数值和含义不同，因此需要明确报告解释的是哪个量。也可以令 $F$ 为损失函数，但此时解释的是损失变化。

**应固定要解释的类别。** 例如解释“猫”的分数，就沿整条路径计算“猫”的同一个分数，不能在每个中间输入上重新选预测类别。`argmax` 给出的离散类别编号也不适合作为通常的梯度归因目标。

IG 由 Mukund Sundararajan、Ankur Taly、Qiqi Yan 在 *Axiomatic Attribution for Deep Networks* 中提出，发表于 ICML 2017，PMLR 70:3319–3328。原论文介绍算法及其公理性质；下文数值例子、推导展开和与攻击方法的对照为教学说明。[论文正式页面](https://proceedings.mlr.press/v70/sundararajan17a.html)

特征值、局部梯度和归因的区别，以及 CNN 特征图与 ViT token 的表示方式，见 [[20_深度学习/训练基础/Feature - 特征详解|Feature - 特征详解]]。

## 二、为什么不能只看当前输入的梯度？

普通梯度

$$
\frac{\partial F(x)}{\partial x_i}
$$

描述的是：**在当前输入附近，小幅改变第 $i$ 个维度，分数的一阶变化率有多大。**

它反映局部敏感性。输入已经让模型进入饱和区域时，当前梯度可能很小，但这不代表该输入相对基线没有带来明显的分数变化。

考虑一维教学函数：

$$
F(z)=1-e^{-z},\qquad z\geq 0.
$$

取 $b=0$、$x=5$，则：

$$
F(0)=0,\qquad F(5)=1-e^{-5}\approx0.993262,
$$

而终点的梯度是：

$$
F'(5)=e^{-5}\approx0.006738.
$$

输出从 0 增长到约 0.9933，但在终点处已经很平，继续增加一点输入只会带来很小的变化。两件事并不矛盾。

IG 的办法是：从基线走到当前输入，记录沿途的梯度，再把它们积累起来。这样就不会只看到终点附近的平坦区域。它能处理这种终点梯度饱和问题，但并不保证所有归因都稳定、符合人类语义或具有因果意义。

## 三、从路径到公式

### 3.1 在基线与输入之间画一条直线

定义路径：

$$
\gamma(\alpha)=b+\alpha(x-b),\qquad \alpha\in[0,1].
$$

- $\alpha=0$ 时，$\gamma(0)=b$。
- $\alpha=1$ 时，$\gamma(1)=x$。
- 中间的 $\alpha$ 对应基线与当前输入之间的插值。

原始像素在 $[0,1]$ 内、基线为黑图时，中间图片看起来会逐渐变亮；一般基线下则是两幅输入的插值，并不一定是“变亮”。

### 3.2 在路径上的每个位置计算梯度

为避免混淆，记：

$$
g_i(\alpha)=
\left.\frac{\partial F(z)}{\partial z_i}\right|_{z=\gamma(\alpha)}.
$$

这里的导数是 **$F$ 对当前路径点 $z$ 的第 $i$ 个坐标的导数**。

IG 的公式为：

$$
\boxed{
\operatorname{IG}_i(x;b)
=(x_i-b_i)\int_0^1 g_i(\alpha)\,d\alpha
}
$$

可以把它拆成：

$$
\text{第 }i\text{ 维的归因}
=\text{该维度相对基线的变化量}
\times\text{沿整条路径的平均敏感性}.
$$

注意，积分区间是长度为 1 的 $\alpha$ 区间，因此这里的积分也是该路径参数下的平均梯度。前面的 $(x_i-b_i)$ 把“变化率”转换成对分数变化的分配量，不能随意省略。

一维例子中：

$$
\operatorname{IG}(5;0)
=5\int_0^1e^{-5\alpha}\,d\alpha
=1-e^{-5}\approx0.993262.
$$

也可以换元 $z=5\alpha$，写成 $\int_0^5 F'(z)\,dz$。在一维图里，它就是梯度曲线下从基线到输入的面积。

多维时，模型在整个输入向量同时插值的路径上计算梯度，各维再分别乘自己的变化量。不能把每个维度都当成其余维度固定不变的一维问题。

## 四、最重要的性质：完备性

对于满足相应路径微积分条件的函数，IG 满足：

$$
\boxed{
\sum_i\operatorname{IG}_i(x;b)=F(x)-F(b)
}
$$

通常的平滑网络，以及沿路径可应用微积分基本定理的常见分段光滑网络，都属于这一讨论范围。

推导为：

$$
\begin{aligned}
\sum_i\operatorname{IG}_i
&=\int_0^1\sum_i
\frac{\partial F(\gamma(\alpha))}{\partial z_i}
(x_i-b_i)\,d\alpha\\
&=\int_0^1\frac{dF(\gamma(\alpha))}{d\alpha}\,d\alpha\\
&=F(x)-F(b).
\end{aligned}
$$

中间一步使用链式法则：$\gamma_i'(\alpha)=x_i-b_i$。

**归因之和通常等于输出差，而不是当前输出本身。** 只有恰好 $F(b)=0$ 时，才等于 $F(x)$。

例如：

$$
F(x_1,x_2)=3x_1+2x_2+7,
\quad b=(0,0),\quad x=(2,1).
$$

梯度恒为 $(3,2)$，所以：

$$
\operatorname{IG}_1=2\times3=6,
\qquad\operatorname{IG}_2=1\times2=2.
$$

归因总和为 $8$，同时 $F(x)=15$、$F(b)=7$，正好有 $6+2=15-7$。常数项 7 已经存在于基线输出中，不需要由输入变化来解释。

## 五、怎样理解正、负和零归因？

| 归因符号 | 在给定目标、基线与路径下的解释 |
| --- | --- |
| 正数 | 分担了所选输出分数的增加 |
| 负数 | 分担了所选输出分数的减少 |
| 零或很小 | 该路径归因值为零或很小，需要结合变化量与梯度判断原因 |

正归因不等于“在终点继续增加这个像素，分数一定会上升”。后者取决于终点的局部梯度，而 IG 汇总整条路径，还包含 $(x_i-b_i)$ 的符号。

零归因也不能直接等同于“模型不使用这个特征”。可能的原因包括：

- 该维与基线相同，即 $x_i-b_i=0$。
- 路径上正负梯度抵消。
- 该路径上相应梯度为零。

因此，IG 是对一次具体输入与基线比较的解释，不能直接变成所有输入上的全局特征排名。

## 六、为什么基线会改变解释？

基线规定了“从什么状态变化到当前状态”。换了比较起点，分数变化及其分配也可能变化。

用包含交互项的函数说明：

$$
F(x_1,x_2)=x_1x_2,\qquad x=(2,3).
$$

**基线一：$b=(0,0)$。** 直线路径为 $(2\alpha,3\alpha)$，梯度为 $(3\alpha,2\alpha)$，因此：

$$
\operatorname{IG}_1=2\int_0^1 3\alpha\,d\alpha=3,
\qquad
\operatorname{IG}_2=3\int_0^1 2\alpha\,d\alpha=3.
$$

**基线二：$b=(0,3)$。** 路径为 $(2\alpha,3)$，只有第一维发生变化，因此：

$$
\operatorname{IG}_1=2\int_0^1 3\,d\alpha=6,
\qquad
\operatorname{IG}_2=0.
$$

两个基线的函数值都为 0，所以归因之和都是 6；分配却由 $(3,3)$ 变成 $(6,0)$。第二种情况下，第二维没有变化，它依然参与乘法运算，但不分担此次相对变化。

这也说明：遇到特征交互，归因需要一种分配约定。完备性保证总账对得上，不会证明某种分配就是唯一的因果解释。

### 6.1 图片的零基线要看坐标

如果输入是原始像素 $p\in[0,1]$，零张量表示黑图。如果输入先标准化：

$$
u=\frac{p-\mu}{\sigma},
$$

那么：

- 标准化空间里的 $u=0$ 对应 $p=\mu$，即各通道均值构成的图片。
- 原始黑图 $p=0$ 在该空间对应 $u=-\mu/\sigma$。

所以不能脱离预处理，说“baseline=0 就是黑图”。Captum 省略基线时使用输入坐标中的零，也不会自动判断它的视觉语义。[Captum IG API](https://captum.ai/api/integrated_gradients.html)

黑图、均值图等都需要结合任务选择，黑图并非所有任务中都天然代表“什么也没有”。实践中可比较多个合理基线，报告解释对基线的敏感性。

如果对多个基线的 IG 求平均，平均归因的总和为：

$$
F(x)-\mathbb{E}_b[F(b)].
$$

这减少对单个起点的依赖，但仍取决于选取的基线分布。

## 七、原论文的两条核心公理

### 7.1 敏感性：Sensitivity(a)

若基线与当前输入仅在一个维度上不同，且函数输出发生了变化，那么那个维度必须得到非零归因。

在这种条件下，由完备性也能直接看出：其他维度的变化量都为零，全部输出差只能归到变化的那一维。

它不能泛化为“所有我们直觉上认为重要的特征，在所有基线下都必须具有非零归因”。后者不是这条公理的准确表述。

### 7.2 实现不变性：Implementation Invariance

如果两个网络对所有输入实现完全相同的标量函数，那么在相同输入、基线和路径下，应给出相同的归因。

IG 从函数梯度和路径积分出发，满足论文讨论的实现不变性。这里要求函数等价，而不是仅在当前图片上预测类别相同。有限步积分、浮点计算，以及不可微点的自动微分约定，可能使实际程序产生数值差异。

不要把这两条公理理解成“只要满足它们，所有归因方法就只能是 IG”；原论文关于方法刻画和唯一性的讨论有额外条件。[原论文全文](https://arxiv.org/html/1703.01365)

## 八、计算时如何近似积分？

一种直观实现是中点求积。选择 $m$ 个点：

$$
\alpha_k=\frac{k+1/2}{m},\qquad k=0,\ldots,m-1,
$$

近似为：

$$
\widehat{\operatorname{IG}}_i
=(x_i-b_i)\frac1m
\sum_{k=0}^{m-1}
\left.\frac{\partial F(z)}{\partial z_i}\right|_{z=b+\alpha_k(x-b)}.
$$

它需要多次前向和梯度计算。路径点可以批量处理，以计算量和显存换取积分精度。具体耗时还受模型、批量大小和求积方式影响。

以前面的一维函数、$x=5$ 为例，直接用标准数学运算计算中点和：

| 中点数量 m | 数值 IG | 数值 IG − 精确输出差 |
| --- | ---: | ---: |
| 32 | 0.99225237 | −0.00100968 |
| 128 | 0.99319891 | −0.00006315 |
| 精确积分 | 0.99326205 | 0 |

这是教学函数的数值求积结果，不是神经网络实验。神经网络上的误差不会保证严格按此比例下降。

### 8.1 一个容易写错的链式法则

公式要的是路径点上的梯度：

$$
\nabla_zF(z)\big|_{z=b+\alpha(x-b)}.
$$

若直接对复合函数 $F(b+\alpha(x-b))$ 关于原变量 $x$ 求导，在固定 $b,\alpha$ 时得到：

$$
\nabla_x F(b+\alpha(x-b))
=\alpha\nabla_zF(z).
$$

这会多出 $\alpha$。将它直接代回标准 IG 的求积公式，会算出另一个量。

手写实现的一种清晰方式是：先构造插值点，再把它作为独立的求导输入（例如 PyTorch 中 `z = z.detach().requires_grad_(True)`），求 $F(z)$ 对 $z$ 的梯度，最后乘 $(x-b)$。若使用 Captum 等成熟实现，则按其 API 设置目标和基线即可。

## 九、Captum 示例与验证

下面代码假设：PyTorch 分类模型接受经过逐通道标准化的图片，输出形状为 `[B, C]` 的 logits；当前解释一张原始像素位于 $[0,1]$ 的图片。代码把标准化封装进 $F$，让 IG 在原始像素坐标中计算，因此原始黑图可直接写成零基线。

```python
import torch
from captum.attr import IntegratedGradients


def explain_image(model, pixels, mean, std, class_id, n_steps=128):
    # pixels: [1, C, H, W]，浮点型原始像素，范围 [0, 1]
    # class_id: 沿整条路径固定的类别编号
    model.eval()
    x = pixels.detach().clone()
    baseline = torch.zeros_like(x)
    mean_t = x.new_tensor(mean).view(1, -1, 1, 1)
    std_t = x.new_tensor(std).view(1, -1, 1, 1)

    def forward_raw(p):
        return model((p - mean_t) / std_t)

    ig = IntegratedGradients(forward_raw, multiply_by_inputs=True)
    attrs, delta = ig.attribute(
        x,
        baselines=baseline,
        target=class_id,
        n_steps=n_steps,
        method="gausslegendre",
        internal_batch_size=16,
        return_convergence_delta=True,
    )

    with torch.no_grad():
        score_gap = (
            forward_raw(x)[:, class_id]
            - forward_raw(baseline)[:, class_id]
        )
    attribution_sum = attrs.flatten(start_dim=1).sum(dim=1)
    completeness_residual = attribution_sum - score_gap
    return attrs, delta, completeness_residual
```

这段代码为按官方 API 编写的教学示例，本次未在安装 PyTorch/Captum 的实际模型环境中执行。`mean`、`std`、设备和数据类型应与具体模型一致。

几点实现说明：

1. `multiply_by_inputs=True` 对应包含 $(x-b)$ 因子的标准 IG；设为 `False` 得到路径平均梯度，不能继续把它当作具有相同完备性关系的标准 IG。
2. Captum 提供多种求积方式；此例显式选择 Gauss–Legendre，前一节为了推导直观使用中点求积。两者都在近似同一个积分。
3. `internal_batch_size` 控制路径样本的计算分块，有助于降低单次显存占用，不会把总积分点数降成该值。
4. 不要在整体归因调用外套 `torch.no_grad()`；求梯度的步骤需要自动微分。`model.eval()` 不会关闭输入梯度。
5. 检查 $\left|\sum_i\widehat{\operatorname{IG}}_i-[F(x)-F(b)]\right|$，并比较增加积分点数前后的结果。Captum 的 convergence delta 基于这种完备性残差。

**残差很小，只说明归因总和与分数差接近。** 不同维度的数值误差可能互相抵消；它不能单独保证每个维度的积分精确，也不能证明基线合理、解释稳定或具有因果意义。[Captum IG API](https://captum.ai/api/integrated_gradients.html) · [官方使用教程](https://captum.ai/docs/extension/integrated_gradients)

## 十、如何看图片热力图？

图片归因通常与输入形状相同，每个颜色通道都有一个归因值。可视化前需要明确聚合方式。

对像素位置 $(h,w)$，若将通道有符号求和：

$$
A_{h,w}=\sum_c\operatorname{IG}_{c,h,w},
$$

则所有像素再求和，仍保留原来的完备性总和。正负颜色可以分别显示支持和抑制所选分数的分配。

如果使用 $\sum_c|\operatorname{IG}_{c,h,w}|$，看到的是归因幅度，正负方向被合并。再经绝对值、截断、按图归一化等变换后，展示图的像素总和一般不再等于原始分数差。

因此，“热图很亮”需要结合图例理解：它可能代表强正归因，也可能只是绝对值较大。可视化强度也不宜脱离所解释的分数及其单位直接跨图比较。

## 十一、和 SIM 有什么关系？

当 $b=0$ 时，IG 的路径点为 $\alpha x$，外形上很像 SIM 使用的幅度缩放输入。但**相似的输入点不等于相同的计算目标**。

| 对比项 | 标准 IG | SIM |
| --- | --- | --- |
| 主要任务 | 解释相对基线的输出差 | 聚合不同幅度副本的损失梯度以更新对抗样本 |
| 计算位置 | 基线到输入的插值路径 | 选定的幅度缩放副本 |
| 梯度变量 | 对路径点 $z$ 求导 | 对攻击的原变量求导，包含变换的链式法则 |
| 变化量因子 | 显式乘 $(x-b)$ | 不能直接套用该 IG 因子 |
| 完备性 | 精确积分下总和为 $F(x)-F(b)$ | 一般不具有此归因恒等式 |

例如，SIM 中若定义复合损失 $L(sx)$，关于攻击变量 $x$ 的梯度为：

$$
\nabla_x L(sx)=s\nabla_z L(z)\big|_{z=sx}.
$$

这里的 $s$ 是正确的链式法则因子；而标准 IG 求积时需要的是 $\nabla_zF(z)$，不应无意中加入额外的 $\alpha$。两者对“对谁求导”的要求来自各自的定义。

另见 [[30_对抗机器学习/迁移与特征攻击/SIM - 尺度不变攻击详解]]。

## 十二、与 NAA、迁移攻击怎样衔接？

IG 本身输出的是归因，不会自动生成一个符合扰动预算的对抗样本。迁移攻击可以借鉴路径上的梯度，或利用归因选择需要干预的特征，但还需要定义攻击目标、更新规则、约束和迭代过程。

[[30_对抗机器学习/迁移与特征攻击/论文 - Improving Adversarial Transferability via Neuron Attribution-Based Attacks|NAA 笔记]] 中讨论的神经元归因，与“沿路径累计敏感性、再结合特征变化量”的思想相通。不过，中间层的表示路径、输出相对该表示的导数及其近似需要按具体方法定义，不能只把输入 IG 公式中的 $x$ 改成某层特征名就视为推导完成。

理解攻击论文中的 integrated gradients 或 MIG 等术语时，应逐项看清：

- 标量 $F$ 选的是类别分数、损失还是别的量。
- 基线与积分路径如何设定。
- 对路径点求导，还是对生成路径点的原变量求导。
- 是否乘输入/特征差，是否再取符号、归一化或加动量。

**不能因为某个位置的 $|\operatorname{IG}_i|$ 大，就直接断言攻击它能产生最大的一阶损失增长，或保证更强迁移性。** 前者还取决于当前攻击损失的梯度与约束，后者需要具体方法及实验支持。本笔记不对某个 MIG 实现作未经核对的算法等同。

相关背景见 [[30_对抗机器学习/ViT与基础模型攻击/ViT Attack - 发展历程]]。

## 十三、记忆要点与边界

可以记成三句话：

1. **普通梯度看当前位置的变化率；IG 将基线到输入沿途的梯度累计成归因。**
2. **IG 等于“输入差 × 路径平均梯度”，各维之和等于 $F(x)-F(b)$。**
3. **解释依赖目标、基线和路径；完备性不是因果正确性的证明。**

直线路径可能经过训练分布之外的输入，相关特征之间的交互也会影响分配。选择合理基线、报告输出定义、检查求积精度和解释稳定性，是应用时需要一起考虑的事。

## 参考来源

- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep Networks*. ICML, PMLR 70:3319–3328。[正式发表页面](https://proceedings.mlr.press/v70/sundararajan17a.html) · [arXiv](https://arxiv.org/abs/1703.01365) · [HTML 全文](https://arxiv.org/html/1703.01365)。
- Captum 官方：[IntegratedGradients API](https://captum.ai/api/integrated_gradients.html) · [Integrated Gradients 使用教程](https://captum.ai/docs/extension/integrated_gradients)。

来源核实日期：2026-11-11。数值例子为教学构造；未运行实际神经网络归因或迁移攻击实验。
