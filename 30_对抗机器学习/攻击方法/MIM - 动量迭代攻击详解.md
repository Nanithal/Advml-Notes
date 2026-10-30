---
title: "MIM - 动量迭代攻击详解"
aliases:
  - "MIM（动量迭代攻击）详解"
分类: "攻击方法"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
created: "2026-10-23"
tags:
  - "对抗攻击"
  - "MIM"
  - "MI-FGSM"
  - "动量"
  - "迁移攻击"
  - "知识库/笔记"
  - "主题/攻击方法"
状态: "已整理"
摘要: "围绕“MIM - 动量迭代攻击详解”，整理MIM（动量迭代攻击）详解、MIM 到底做了什么、先把攻击目标写清楚。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/攻击方法/总结 - 攻击方法]]"
相关笔记:
  - "[[30_对抗机器学习/攻击方法/MI-FGSM - 动量迭代攻击（早期阅读笔记）]]"
  - "[[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum]]"
  - "[[10_基础与算法/优化方法/Momentum - 带动量的梯度下降]]"
  - "[[30_对抗机器学习/迁移与特征攻击/论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima]]"
论文条目:
  - "[[论文来源目录#mifgsm]]"
论文标题:
  - "Boosting Adversarial Attacks with Momentum"
发表载体:
  - "CVPR"
发表年份:
  - "2018"
发表类型:
  - "会议论文"
发表链接:
  - "https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html"
预印本链接:
  - "https://arxiv.org/abs/1710.06081"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/攻击方法/总结 - 攻击方法|攻击方法总结]]

# MIM - 动量迭代攻击详解


# MIM（动量迭代攻击）详解

> 在对抗攻击语境下，MIM 通常指 Momentum Iterative Method。本文主要讲其中最常用的 $L_\infty$ 版本 MI-FGSM，即 Momentum Iterative Fast Gradient Sign Method。
>
> 来源：Dong 等人的 **Boosting Adversarial Attacks with Momentum，CVPR 2018**。不要将其误记为 ICLR 2018。
>
> 阅读顺序：攻击目标 → 单步与迭代攻击 → 动量公式 → 数值例子 → 算法实现。

## 1. MIM 到底做了什么

MIM 在每次攻击时，结合**当前输入梯度和历史梯度**，然后根据累积方向修改图像。

普通迭代攻击每步只问：“现在往哪里改，能让模型损失上升？”MIM 还会考虑：“过去几步持续支持哪个方向？”

它更新的是输入图像，模型参数保持固定。引入动量主要用于稳定方向，增强对抗样本从源模型向其他模型的迁移；不保证所有模型与预算下都优于其他攻击。

## 2. 先把攻击目标写清楚

设输入图像为 $x\in[0,1]^{C\times H\times W}$，真实标签为 $y$，模型为 $f_\theta$，交叉熵损失为 $J$。

无目标攻击希望增加真实标签的分类损失：

$$
\max_{\delta}J(f_\theta(x+\delta),y),
\qquad
\|\delta\|_\infty\le\epsilon,
\quad x+\delta\in[0,1]^d.
$$

$\|\delta\|_\infty=\max_i|\delta_i|$，表示任意一个像素通道的改变都不能超过 $\epsilon$。它不限制只有少量像素可变：全图都可以发生小幅变化。

无目标攻击的成功条件为：

$$
\arg\max_k f_\theta(x^{\mathrm{adv}})_k\ne y.
$$

它保留原始语义标签 $y$ 作为参照，目标是让**预测标签出错**；“非目标攻击不改变标签”容易造成误解。

| 符号 | 含义 |
|---|---|
| $x$ | 原始干净图像，整个攻击过程的固定投影中心 |
| $x_t$ | 第 $t$ 次迭代的对抗图像，$x_0=x$ |
| $r_t$ | 当前损失对输入的原始梯度 |
| $m_t$ | 累积历史信息的动量向量，$m_0=0$ |
| $\epsilon$ | 最大扰动预算 |
| $\alpha$ | 每次更新步长 |
| $T$ | 迭代次数 |
| $\mu$ | 历史动量衰减系数 |
| $\Pi_{\mathcal C}$ | 投影到合法图像和预算的交集 |

本文用 $r_t$ 表示原始梯度、$m_t$ 表示动量，避免两者都写成 $g_t$。

## 3. 从 FGSM 到 I-FGSM，再到 MIM

### 3.1 FGSM：根据一次梯度走一步

$$
x^{\mathrm{adv}}=\operatorname{clip}_{[0,1]}
\left(x+\epsilon\operatorname{sign}(\nabla_xJ(f_\theta(x),y))\right).
$$

因为要增大损失，所以使用加号。对 $L_\infty$ 约束的一阶近似，逐坐标取梯度符号给出局部最大化方向。

### 3.2 I-FGSM：每一步重新看当前梯度

$$
r_t=\nabla_{x_t}J(f_\theta(x_t),y),
\qquad
x_{t+1}=\Pi_{\mathcal C}(x_t+\alpha\operatorname{sign}(r_t)).
$$

每次重新计算梯度，通常能更充分攻击源模型；但也可能越来越依赖该模型特有的局部性质，迁移效果未必随步数增加而提高。

### 3.3 MIM：先归一化、再累积，最后取符号

原论文算法 1、式 (6)–(7) 可用本文符号写为：

$$
\widehat r_t=\frac{r_t}{\|r_t\|_1},
$$

$$
\boxed{m_{t+1}=\mu m_t+\widehat r_t},
$$

$$
\boxed{x_{t+1}=\Pi_{\mathcal C}
\left(x_t+\alpha\operatorname{sign}(m_{t+1})\right)}.
$$

其中：

$$
\mathcal C=[0,1]^d\cap B_\infty(x,\epsilon),
\qquad \alpha=\frac\epsilon T\ \text{是原文常用设置}.
$$

这里显式写出投影，便于实际实现。原文从干净图起步、固定 $T$ 步、$\alpha=\epsilon/T$ 时，即使方向变化，每坐标累计位移也不超过 $\epsilon$；自定义大步长时则尤其需要显式投影。

## 4. 为什么要用 $L_1$ 归一化

输入梯度中有 $d$ 个坐标：

$$
\|r_t\|_1=\sum_{i=1}^{d}|r_{t,i}|.
$$

不同迭代的梯度绝对大小可能差异很大。若直接累加，某一次异常大的梯度会仅凭尺度压过其他步。归一化使每步贡献具有可比尺度，同时保留梯度内部的相对大小与方向。

必须区分两个操作：

$$
\frac{r_t}{\|r_t\|_1}
\quad\text{与}\quad
\operatorname{sign}(r_t).
$$

前者保留各坐标相对大小，后者只保留正负。MIM **累积归一化后的实值梯度，最后才取符号**；先把各步变成正负一再累积，不是原算法。

实际代码可将分母限制为不小于一个很小正数，避免零梯度除零。这是数值处理，不是新增攻击预算。

对批量图像 $X\in\mathbb R^{B\times C\times H\times W}$，应为每张图单独计算分母，归约轴为 $(C,H,W)$，不应把整个 batch 共用一个归一化标量。

## 5. 动量怎样保留历史信息

从 $m_0=0$ 展开：

$$
m_{t+1}=\sum_{k=0}^{t}\mu^{t-k}\widehat r_k.
$$

因此，$\mu$ 控制历史梯度保留程度：

- $\mu=0$：不保留历史。在相同初始化、步长和投影下，通常退化为 I-FGSM；零梯度按实现约定处理。
- $0<\mu<1$：较早的梯度指数衰减。
- $\mu=1$：原论文常用设置，各历史归一化梯度直接累加，不发生衰减。

它与训练优化器中常见的 $v_{t+1}=\beta v_t+(1-\beta)g_t$ 有联系，但不是完全相同的定义。特别是 MIM 中 $\mu=1$ 仍会加入新梯度，不能直接套用带 $(1-\beta)$ 的 EMA 公式。

对 $0<\mu<1$，两种递推在初始值为零且使用相同梯度序列时可相差固定正比例；MI-FGSM 最后取符号，因此这种整体比例通常不改变更新方向。$\mu=1$ 的边界情况需单独理解。

## 6. 用二维例子理解“方向稳定”

假设已经归一化的连续梯度为：

$$
\widehat r_0=(0.8,0.2),\qquad
\widehat r_1=(-0.3,0.7),\qquad
\widehat r_2=(-0.9,0.1),\qquad \mu=1.
$$

三者 $L_1$ 范数都是 1。逐步计算：

| 迭代 | 当前归一化梯度 | 更新后的动量 | I-FGSM 方向 | MIM 方向 |
|---|---|---|---|---|
| 第 1 步 | $(0.8,0.2)$ | $(0.8,0.2)$ | $(+,+)$ | $(+,+)$ |
| 第 2 步 | $(-0.3,0.7)$ | $(0.5,0.9)$ | $(-,+)$ | $(+,+)$ |
| 第 3 步 | $(-0.9,0.1)$ | $(-0.4,1.0)$ | $(-,+)$ | $(-,+)$ |

第二步第一个坐标的当前梯度虽然变负，但还不足以抵消历史正方向，MIM 因而保持向正方向更新。第三步负方向证据继续增强，动量最终也会翻转。

这说明动量并非永远固定第一次的方向，而是使方向改变需要积累更多证据。这只是教学示例，不代表所有真实梯度都具有这种模式。

## 7. 动量大了，像素步长会越来越大吗

在 MI-FGSM 中，不会因为动量绝对值变大就自动放大每步像素位移：

$$
\Delta x_t=\alpha\operatorname{sign}(m_{t+1}).
$$

对非零动量坐标，投影前步幅都是 $\alpha$；动量为零时符号为零。投影还可能缩小实际位移。

因此，MIM 中“加速、积累速度”的直觉主要体现在方向的形成和稳定，而不是像普通未取符号的梯度下降一样，让位移直接正比于动量幅度。

## 8. 投影究竟怎么做

逐像素限制在固定原图邻域：

$$
\delta_{t+1}=\operatorname{clip}(x'_{t+1}-x,-\epsilon,\epsilon),
\qquad
x_{t+1}=\operatorname{clip}(x+\delta_{t+1},0,1).
$$

这里 $x'_{t+1}$ 是暂时更新后的图像。第二次裁剪仍不会超出原图邻域，因为原图本身位于 $[0,1]$ 内。

例如 $x_i=0.40$、$\epsilon=0.03$，该坐标始终在 $[0.37,0.43]$ 中。不能每轮围绕上一轮图像重新设定 $\pm\epsilon$，否则总位移会不断扩大。

## 9. 完整算法步骤与伪代码

1. 固定模型权重，将模型设为评估模式。准备原图、真实标签、预算、步数和动量系数。
2. 设置 $x_0=x,m_0=0$，通常 $\alpha=\epsilon/T$。
3. 对当前 $x_t$ 前向，计算真实标签交叉熵。
4. 求损失对当前输入的梯度，对每张图单独做 $L_1$ 归一化。
5. 累积动量，再取动量符号。
6. 图像沿该方向更新；投影到原图预算邻域，并裁剪合法像素。
7. 重复 $T$ 步，在源模型及其他目标模型上分别评估。

```text
输入：源模型 f，原图 x，标签 y，预算 ε，步数 T，动量系数 μ
α ← ε / T
z ← x
m ← 0
重复 T 次：
    loss ← CE(f(z), y)
    r ← loss 对 z 的梯度
    r_hat ← 按图像单独对 r 做 L1 归一化
    m ← μ·m + r_hat
    z ← z + α·sign(m)
    z ← x + clip(z-x, -ε, ε)
    z ← clip(z, 0, 1)
返回 z
```

每个迭代通常约需一次前向和一次反向，主要计算成本随 $T$ 增长；动量只需额外保存一个与输入同形状的张量。

## 10. PyTorch 参考实现

下面实现原始思路的无随机起点版本，并支持定向攻击。前提是图像为 `[B,C,H,W]` 的浮点张量，值位于 `[0,1]`，模型输出 `[B,K]` 的 **logits**，标签为类别索引。代码适用于通常的独立样本分类器。

归一化通过 `preprocess` 放进模型前向中，预算保持在原始像素空间。调用者先设置 `model.eval()`；函数不更新权重，也不改变模型训练状态。

```python
import torch
import torch.nn.functional as F


@torch.enable_grad()
def mim_linf(
    model,
    images,
    labels,
    eps=8 / 255,
    steps=10,
    decay=1.0,
    alpha=None,
    targeted=False,
    preprocess=None,
):
    """MI-FGSM；targeted=True 时 labels 必须是目标类别。"""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if eps < 0 or not 0 <= decay <= 1:
        raise ValueError("require eps >= 0 and 0 <= decay <= 1")
    if alpha is None:
        alpha = eps / steps
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    x = images.detach().clone()
    adv = x.clone()
    momentum = torch.zeros_like(x)
    sample_dims = tuple(range(1, x.ndim))

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        model_input = adv if preprocess is None else preprocess(adv)
        logits = model(model_input)
        ce = F.cross_entropy(logits, labels)

        # 始终对 objective 做上升：定向时上升 -CE，等价于降低目标 CE。
        objective = -ce if targeted else ce
        grad = torch.autograd.grad(objective, adv)[0]

        # 每张图独立归一化，不混入 batch 维。
        scale = grad.abs().sum(dim=sample_dims, keepdim=True)
        normalized_grad = grad / scale.clamp_min(1e-12)
        momentum = decay * momentum + normalized_grad

        with torch.no_grad():
            adv = adv + alpha * momentum.sign()
            delta = (adv - x).clamp(-eps, eps)
            adv = (x + delta).clamp(0.0, 1.0)

    return adv.detach()
```

调用示例：

```python
model.eval()

# 仅作示例：均值和标准差必须与当前模型的预训练配置一致。
mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

adv = mim_linf(
    model,
    images,
    labels,
    eps=8 / 255,
    steps=10,
    decay=1.0,
    preprocess=lambda z: (z - mean) / std,
)

# 可以在此处用 no_grad 做最终评估。
with torch.no_grad():
    clean_pred = model((images - mean) / std).argmax(dim=1)
    adv_pred = model((adv - mean) / std).argmax(dim=1)
    clean_correct = clean_pred.eq(labels)
    attack_success = clean_correct & adv_pred.ne(labels)
    asr = attack_success.sum().float() / clean_correct.sum().clamp_min(1)
    linf_per_image = (adv - images).abs().flatten(1).amax(dim=1)
```

当没有干净正确样本时，示例用分母钳制避免除零，但该批次的 ASR 应标记为“无可评估样本”，不能把数值零当成有意义的攻击结果。

`torch.autograd.grad` 直接返回所需输入梯度，不将其累积到模型参数的 `.grad` 中。每轮 `detach()` 避免把此前攻击迭代都保留在计算图里。`cross_entropy` 接收 logits，前面不要额外做 softmax。相关 API 依据 [PyTorch autograd.grad 文档](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad.html) 和 [cross_entropy 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html)。

这是教学参考实现；未在实际 ImageNet/ViT 模型上复现原论文结果。输入和模型通常应使用 FP32，避免极小稳定常数在低精度下失效。攻击阶段必须允许梯度，不能包在 `torch.inference_mode()` 中。

## 11. 定向攻击为什么是减号

定向攻击希望预测变成指定目标 $y^*$，所以降低目标类损失：

$$
\min_{x^{\mathrm{adv}}\in\mathcal C}J(f_\theta(x^{\mathrm{adv}}),y^*).
$$

若动量累积的是该交叉熵的正梯度，图像更新应为：

$$
x_{t+1}=\Pi_{\mathcal C}
(x_t-\alpha\operatorname{sign}(m_{t+1})).
$$

上面代码采用另一种等价记法：先令 `objective = -ce`，再统一用加号更新。不能同时取负损失又把更新改为减号，否则抵消了两次负号。

定向成功率应统计预测是否等于 $y^*$，不能仅统计是否离开真实类别。

## 12. 与 FGSM、I-FGSM、PGD 和 SGD 动量的区别

| 方法 | 主要更新对象 | 使用历史梯度 | 常见起点 | 主要用途 |
|---|---|---|---|---|
| FGSM | 输入图像 | 否 | 原图 | 快速单步攻击 |
| I-FGSM / BIM | 输入图像 | 否 | 原图 | 多步增加源模型损失 |
| PGD | 输入图像 | 基础版本不使用 | 常见为预算球内随机起点 | 充分搜索约束邻域、鲁棒性评估 |
| MIM / MI-FGSM | 输入图像 | 是 | 原始版本从原图出发 | 稳定方向、改善攻击迁移 |
| 带动量 SGD | 模型参数 | 是 | 初始参数 | 通常降低训练损失 |

PGD 是更一般的投影优化框架。MIM 加上投影，可理解为一种使用动量方向的投影攻击；不能说“只要有投影就是 PGD，因此和 MIM 没区别”。是否随机初始化、如何得到更新方向、是否重启，都会改变具体算法。

训练动量通常沿下降方向改 $\theta$；无目标攻击沿上升方向改 $x$。对输入调用带动量优化器也不自动等价于 MI-FGSM，因为原算法还有逐步归一化、最终取符号以及投影。

## 13. 为什么动量可能改善迁移性

源模型和目标模型可能具有部分相似的有效特征与脆弱方向，也存在各自特有的局部变化。若每一步都紧跟当前源模型梯度，可能把扰动优化到只对源模型特别有效的位置。

动量让连续多步支持的方向更容易保留，偶然反向或短暂出现的分量可能抵消，从而在论文所测条件下减少这种局部过拟合。它是对实验证据的解释，不是证明所有稳定方向都是跨模型共有方向。

原论文主要在 CNN 模型之间展示改善迁移的证据；将它用到 ViT 上通常作为基础外层攻击，再结合 TGR、PNA 等内部梯度处理。对 ViT 是否有效仍应在指定源目标对上实际验证。

## 14. 多模型集成怎么与 MIM 组合

原文研究多模型 logits 集成。令源模型输出 logits 为 $z_k(x)$：

$$
z_{\mathrm{ens}}(x)=\sum_{k=1}^{K}w_kz_k(x),\qquad
w_k\ge0,\quad\sum_kw_k=1,
$$

$$
J_{\mathrm{ens}}(x,y)=\operatorname{CE}_{\mathrm{logits}}
(z_{\mathrm{ens}}(x),y).
$$

这里 $\operatorname{CE}_{\mathrm{logits}}$ 表示内部含 softmax/log-softmax 的交叉熵形式，与 PyTorch 接口一致。每步求联合梯度，再按同样的归一化、动量、符号更新流程攻击。

“先融合 logits，再算一次损失”与“各自算损失再平均”通常不等价。原文测试中 logits 集成表现较好，但多源模型会增加计算，不能把全部收益归因于动量。

## 15. 常见参数和容易犯的错误

| 项目 | 解释与注意事项 |
|---|---|
| $\epsilon$ | 必须与输入尺度一致；`8/255` 是 `[0,1]` 像素空间中的预算 |
| $T$ | 更多步数增加成本，不保证迁移率单调上升 |
| $\alpha$ | 原文常取 $\epsilon/T$；自定义后仍需投影 |
| $\mu$ | 原文常用 1.0；0 表示移除历史项 |
| 随机起点 | 是可选扩展，不能未经说明称为原始 MI-FGSM 配置 |
| 输入标准化 | 在前向中标准化，预算和投影仍在原像素空间执行 |
| 梯度取符号 | 对累积后的动量取符号，不要先把每步梯度取符号 |
| 更新对象 | 输入变，模型权重不变 |
| 评估模式 | 通常 `model.eval()`，保持正常推理行为 |
| 图像保存 | 每轮保持浮点数；反复量化成 uint8 可能丢失小步更新 |

例如 $\epsilon=8/255,T=10$ 时，$\alpha=0.8/255$。单步小于一个 8 位灰度级，浮点攻击仍然有效；若每步都转为整数图像，更新可能被截断。最终保存图像后，也应重新读取并评估量化影响。

还应避免将“攻击当前源模型”和“迁移到未知模型”混为一谈：迁移攻击生成阶段可以访问源模型梯度，但不读取目标模型梯度；最终测试时才观察目标性能。

## 16. 如何阅读与记录实验结果

至少记录：源模型、目标模型、干净正确样本筛选、预算与输入尺度、步数与步长、动量、随机起点/重启、预处理、计算成本。

在目标模型干净正确的集合 $\mathcal I$ 上，无目标攻击成功率可定义为：

$$
\operatorname{ASR}=\frac1{|\mathcal I|}\sum_{i\in\mathcal I}
\mathbf1\{\arg\max f_t(x_i^{\mathrm{adv}})\ne y_i\}.
$$

有些论文进一步要求源模型也正确，或只在源攻击成功的样本中统计迁移率；必须说明分母。不能直接混比来自不同定义、目标集合和预算的百分比。

比较时，在相同初始化与外层参数下令 $\mu=0$，可帮助理解动量本身的作用；再分别加入随机起点或集成，避免多个变化同时发生。

## 17. 相关笔记与原始资料

- [[30_对抗机器学习/攻击方法/MI-FGSM - 动量迭代攻击（早期阅读笔记）|已有 MI-FGSM 笔记]]
- [[10_基础与算法/优化方法/Momentum - 带动量的梯度下降|带动量的 SGD]]
- [[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum|原论文阅读笔记]]
- [原论文官方会议 PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.pdf)：算法 1，式 (6)–(7)；集成与扩展见后续小节。
- [本地原论文](</home/nanithal/Advm learing/ViTAttack/Paper/00_模型与通用攻击基础/2018_MI-FGSM_Boosting Adversarial Attacks with Momentum.pdf>)。

本文将原始梯度与动量分开命名，显式加入图像边界和预算投影；二维数字是教学示例，代码是参考实现，均不应当作原论文实验数据。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/攻击方法/MI-FGSM - 动量迭代攻击（早期阅读笔记）|MI-FGSM - 动量迭代攻击（早期阅读笔记）]]：比较早期记录与后续详细推导。
- [[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum|论文 - Boosting Adversarial Attacks with Momentum]]：回到动量迭代攻击的论文阅读。
- [[10_基础与算法/优化方法/Momentum - 带动量的梯度下降|Momentum - 带动量的梯度下降]]：回顾动量平均与历史方向累积。
- [[30_对抗机器学习/迁移与特征攻击/论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima|论文 - Boosting Adversarial Transferability by Achieving Flat Local Maxima]]：比较动量累积与梯度范数惩罚的更新思路。

## 论文出处

- [Boosting Adversarial Attacks with Momentum](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)：CVPR，2018，会议论文。
  [预印本](https://arxiv.org/abs/1710.06081)。

<!-- knowledge-links:end -->
