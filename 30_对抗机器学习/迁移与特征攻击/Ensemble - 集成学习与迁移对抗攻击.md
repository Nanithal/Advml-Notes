---
title: "Ensemble - 集成学习与迁移对抗攻击"
aliases:
  - "Ensemble_集成学习与迁移对抗攻击详解"
tags:
  - "知识库/笔记"
  - "主题/迁移与特征攻击"
笔记类型: "概念笔记"
分类: "迁移与特征攻击"
状态: "已整理"
摘要: "区分传统集成学习与代理模型集成攻击，比较 logits、损失和梯度融合以及梯度冲突。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击]]"
相关笔记:
  - "[[40_课程笔记/吴恩达机器学习/集成学习/集成学习 - 课程笔记]]"
  - "[[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum]]"
  - "[[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples]]"
论文条目:
  - "[[论文来源目录#ensemble]]"
  - "[[论文来源目录#mifgsm]]"
论文标题:
  - "Delving into Transferable Adversarial Examples and Black-box Attacks"
  - "Boosting Adversarial Attacks with Momentum"
发表载体:
  - "ICLR"
  - "CVPR"
发表年份:
  - "2017"
  - "2018"
发表类型:
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://openreview.net/pdf/978205dd6095246709f97a4ec11279e65df57035.pdf"
  - "https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html"
预印本链接:
  - "https://arxiv.org/abs/1611.02770"
  - "https://arxiv.org/abs/1710.06081"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/迁移与特征攻击/总结 - 迁移与特征攻击|迁移与特征攻击总结]]

# Ensemble - 集成学习与迁移对抗攻击

> 学习重点：集成学习的基本思想、三种经典集成方法、迁移对抗攻击中的 Ensemble Attack、梯度集成与冲突、Ensemble + MI-FGSM、ViT Attack 应用。

## 一、什么是 Ensemble？

**Ensemble（集成）**是将多个模型的信息进行组合，以提高预测可靠性或改善攻击方向的方法。

- **普通机器学习**：融合多个模型的预测，以改善泛化与预测表现。
- **迁移对抗攻击**：融合多个代理模型的损失或梯度，尝试提高对未知目标模型的迁移性。

假设有三个模型，分别输出猫、狗、鸟的概率：

| 模型      |   猫 |   狗 |   鸟 |
| ------- | --: | --: | --: |
| Model 1 | 0.8 | 0.1 | 0.1 |
| Model 2 | 0.6 | 0.3 | 0.1 |
| Model 3 | 0.4 | 0.5 | 0.1 |

对各模型输出取均值：

$$
P(y\mid x)=\frac{1}{3}\sum_{i=1}^3P_i(y\mid x)
$$

集成结果为：

$$
P(\mathrm{Cat})=0.6,\quad P(\mathrm{Dog})=0.3,\quad P(\mathrm{Bird})=0.1.
$$

所以最终预测为 **Cat**。一般地，可采用加权集成：

$$
P_{ens}(y\mid x)=\sum_{i=1}^K w_iP_i(y\mid x),\quad w_i\ge 0,\quad\sum_i w_i=1.
$$

## 二、为什么 Ensemble 可能比单模型好？

关键是 **Diversity（多样性）**：不同模型的误差具有互补性时，组合模型更有机会改善结果。

设每个模型的预测误差为 $e_i$，误差方差均为 $\sigma^2$，两两误差相关系数为 $\rho$，平均误差为 $\bar e=\frac1K\sum_i e_i$，则在这些假设下：

$$
\boxed{\operatorname{Var}(\bar e)=\frac{\sigma^2}{K}[1+(K-1)\rho]}
$$

- $\rho=0$：方差降为 $\sigma^2/K$。
- $\rho=1$：方差仍为 $\sigma^2$。

**注意**：这是关于误差方差的说明，并不保证分类准确率一定提高。

## 三、传统集成学习的三种典型方式

### 1. Bagging

对训练数据进行有放回采样，各子模型训练后投票或平均。代表方法：**Random Forest（随机森林）**。主要用于降低方差。

### 2. Boosting

模型顺序训练，后续模型侧重修正前面模型的不足。代表方法：**AdaBoost、Gradient Boosting、XGBoost、LightGBM**。

### 3. Stacking

先训练多个基模型，再训练一个元模型（Meta Model）学习如何融合它们的输出：

$$
\hat y=g(f_1(x),f_2(x),\ldots,f_K(x)).
$$

## 四、Ensemble Attack：集成迁移对抗攻击

设目标模型为 $f_t$，代理模型为 $f_s$。单代理非定向迁移攻击常优化：

$$
\max_{\|\delta\|_p\le\epsilon}L(f_s(x+\delta),y)
$$

问题在于：单个代理模型的梯度可能具有强烈的**模型特定性**，导致迁移效果不足。

因此使用多个代理模型 $f_1,\ldots,f_K$，共同构造扰动：

$$
\boxed{\text{Ensemble Attack}=\text{融合多个代理模型的攻击信息}}
$$

### 1. Logits 集成

对模型 Logits 加权：

$$
z_{ens}(x)=\sum_{i=1}^K w_i z_i(x),\quad L_{ens}=L(z_{ens}(x),y).
$$

再用 $\nabla_x L_{ens}$ 生成扰动。模型输出类别顺序应一致，Logits 尺度差异也可能影响结果。

### 2. Loss 集成

对各模型损失加权：

$$
\boxed{L_{ens}(x,y)=\sum_{i=1}^K w_iL(f_i(x),y)}
$$

若权重固定：

$$
\boxed{\nabla_xL_{ens}=\sum_{i=1}^Kw_i\nabla_xL(f_i(x),y)}
$$

### 3. Gradient 集成

直接融合梯度：

$$
\boxed{g_{ens}=\sum_{i=1}^K w_i g_i},\qquad g_i=\nabla_xL(f_i(x),y).
$$

也可以先归一化：

$$
g_{ens}=\sum_{i=1}^K w_i\frac{g_i}{\|g_i\|_1+\tau},\qquad\tau>0.
$$

对于固定权重、未归一化梯度，Loss 集成与 Gradient 集成在数学上等价；归一化或动态权重则会改变这种关系。

## 五、为什么 Ensemble 可能提升迁移性？

不同代理模型的梯度方向可能不同，集成尝试寻找对多个模型共同有效的方向，以减轻攻击只适配某一模型的风险。

但它**不是必然提升迁移性**：**模型过于相似、梯度相互抵消或融合权重不合适**，都可能使效果变差。

### 梯度冲突

例如：

$$
g_1=\begin{bmatrix}1\\0\end{bmatrix},\qquad
 g_2=\begin{bmatrix}-1\\0\end{bmatrix}.
$$

简单平均后：

$$
g_{ens}=\frac12(g_1+g_2)=\begin{bmatrix}0\\0\end{bmatrix}.
$$

梯度相互抵消，导致有效更新信号消失。常用余弦相似度分析两个梯度的方向：

$$
\boxed{\operatorname{sim}(g_i,g_j)=\frac{g_i^Tg_j}{\|g_i\|_2\|g_j\|_2}}
$$

- 接近 $1$：方向相近；
- 接近 $0$：近似正交；
- 接近 $-1$：方向相反。

梯度一致性与多样性都不是越大越好，最终要看对未知目标模型的攻击效果。

## 六、Ensemble 与 MI-FGSM 结合

Ensemble 融合**模型之间**的信息，Momentum 融合**不同迭代步骤**的信息，两者可以组合。

首先计算集成梯度：

$$
g_t^{ens}=\sum_{i=1}^K w_i\nabla_xL(f_i(x_t),y).
$$

归一化：

$$
\hat g_t^{ens}=\frac{g_t^{ens}}{\|g_t^{ens}\|_1+\tau}.
$$

更新动量：

$$
m_{t+1}=\mu m_t+\hat g_t^{ens}.
$$

在非定向 $L_\infty$ 攻击下更新样本：

$$
\boxed{x_{t+1}=\Pi_{\mathcal B_\epsilon(x)}\left(x_t+\alpha\operatorname{sign}(m_{t+1})\right)}.
$$

其中 $\Pi$ 表示投影到原图像附近的允许扰动集合，并应同时保持合法输入范围。

## 七、Ensemble 在 ViT Attack 中的应用

### 1. CNN + ViT 跨架构集成

例如将 ResNet-50、ViT-B/16、Swin-T 作为代理模型：

$$
L_{ens}=w_1L_{ResNet}+w_2L_{ViT}+w_3L_{Swin}.
$$

目的是同时利用卷积网络和视觉 Transformer 的不同表征，尝试提升跨架构迁移性，实际效果需要实验验证。

### 2. 多 ViT 模型集成

例如：ViT-B/16、DeiT-B、Swin-T。可以研究不同 Transformer 架构与训练策略下的攻击迁移。

### 3. 多层特征 / 损失集成

不仅可以集成模型，还可以集成多层目标：

$$
L_{ens}=\sum_{l=1}^L w_lL_l.
$$

这是多层特征或目标融合，和多代理模型集成相关但不完全相同。

## 八、Ensemble 与 SGM 的区别

| 对比维度 | Ensemble Attack | SGM |
|---|---|---|
| 核心思想 | 融合多个代理模型的信息 | 降低残差分支的梯度贡献 |
| 主要对象 | 模型预测、损失、梯度 | 残差网络的反向传播路径 |
| 是否需要多个代理模型 | 通常需要 | 不需要 |
| 是否改变模型反向传播 | 不一定 | 是 |
| 目标 | 改善迁移攻击效果 | 改善迁移攻击效果 |
| 能否组合 | 可以 | 可以 |

例如先对多个残差代理模型分别应用 SGM，再把它们的梯度集成，最后用 MI-FGSM 的动量更新扰动：

$$
\boxed{\text{SGM}+\text{Ensemble}+\text{MI-FGSM}}
$$

## 九、PyTorch：简单的 Ensemble I-FGSM

下面的代码演示固定等权重的 Loss Ensemble + I-FGSM：

```python
import torch
import torch.nn.functional as F


def ensemble_attack(models, x, y, epsilon=8/255, alpha=2/255, steps=10):
    """非定向 L_inf 集成攻击；输入须为 [0,1] 范围的图像。"""
    x_orig = x.detach().clone()
    x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)

        loss = 0
        for model in models:
            logits = model(x_adv)
            loss = loss + F.cross_entropy(logits, y)
        loss = loss / len(models)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        delta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
        x_adv = (x_orig + delta).clamp(0, 1).detach()

    return x_adv


# 使用示例（model1、model2、model3、images、labels 需事先定义）
models = [model1, model2, model3]
for model in models:
    model.eval()

x_adv = ensemble_attack(models, images, labels)
```

**注意**：示例假定各模型都接收相同形状、处于 $[0,1]$ 范围的图像，并使用相同类别定义。实际 CNN/ViT 组合中应在各模型内部处理不同的归一化和预处理；攻击仍应相对于原始像素空间计算扰动预算。

## 十、推荐论文

1. **Delving into Transferable Adversarial Examples and Black-box Attacks**，Yanpei Liu 等，ICLR 2017。探讨对抗样本的跨模型迁移以及模型集成攻击。论文：https://arxiv.org/abs/1611.02770
2. **Boosting Adversarial Attacks with Momentum**，Yinpeng Dong 等，CVPR 2018。提出 MI-FGSM，并讨论动量和模型集成对攻击迁移性的影响。论文：https://arxiv.org/abs/1710.06081

## 十一、总结

Ensemble 的基本思想：

$$
\boxed{\text{Ensemble}=\text{融合多个模型的信息}}
$$

对抗攻击中最值得记住的两个公式：

$$
\boxed{L_{ens}=\sum_{i=1}^K w_iL_i}
$$

$$
\boxed{g_{ens}=\sum_{i=1}^K w_i\nabla_xL_i}
$$

在 ViT Attack 中继续思考：

1. 多个代理模型的互补性为何可能改善跨架构迁移？
2. 梯度冲突如何影响集成后的更新？
3. 如何选择模型、归一化梯度或设计动态权重？

**核心结论**：集成攻击尝试找到跨模型共享的有效攻击方向，但不是模型越多越好；模型差异、梯度融合与实验威胁模型共同决定最终效果。

<!-- knowledge-links:start -->

## 笔记联系

- [[40_课程笔记/吴恩达机器学习/集成学习/集成学习 - 课程笔记|集成学习 - 课程笔记]]：先回顾传统模型集成，再区分迁移攻击中的融合对象。
- [[30_对抗机器学习/攻击方法/论文 - Boosting Adversarial Attacks with Momentum|论文 - Boosting Adversarial Attacks with Momentum]]：对照动量迭代攻击论文中的集成方法。
- [[30_对抗机器学习/迁移与特征攻击/SGM - 跳跃梯度方法|SGM - 跳跃梯度方法]]：比较跨模型融合与模型内部反向路径修改。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples|论文 - On the Robustness of Vision Transformers to Adversarial Examples]]：对照跨架构模型与 SAGA 集成攻击的评估。

## 论文出处

- [Delving into Transferable Adversarial Examples and Black-box Attacks](https://openreview.net/pdf/978205dd6095246709f97a4ec11279e65df57035.pdf)：ICLR，2017，会议论文。
  [预印本](https://arxiv.org/abs/1611.02770)。
- [Boosting Adversarial Attacks with Momentum](https://openaccess.thecvf.com/content_cvpr_2018/html/Dong_Boosting_Adversarial_Attacks_CVPR_2018_paper.html)：CVPR，2018，会议论文。
  [预印本](https://arxiv.org/abs/1710.06081)。

<!-- knowledge-links:end -->
