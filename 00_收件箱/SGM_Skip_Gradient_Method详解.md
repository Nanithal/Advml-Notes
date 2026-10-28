# SGM（Skip Gradient Method）详解

SGM（Skip Gradient Method，跳跃梯度方法）是提升**迁移对抗攻击**效果的一种梯度修改策略。它主要针对带有残差连接的网络，在**不改变前向输出**的情况下，降低反向传播时残差分支的梯度贡献，使更多梯度沿恒等映射路径传播。

> 核心公式：将残差块的反向雅可比从 $I+J_F$ 修改为 $I+\gamma J_F$，其中 $0\le\gamma\le1$。

## 1. 研究背景：为什么需要 SGM？

在迁移攻击中，攻击者通常知道代理模型 $f_s$，但不能访问目标模型 $f_t$ 的梯度。攻击者构造

$$
x^{adv}=x+\delta,\qquad \|\delta\|_\infty\le\epsilon,
$$

希望目标模型也预测错误。FGSM 使用代理模型梯度：

$$
x^{adv}=x+\epsilon\operatorname{sign}(\nabla_x L(f_s(x),y)).
$$

然而，代理模型的梯度可能包含模型特有的信息，因此白盒攻击成功不代表能迁移到其他模型。SGM 研究：**残差网络中不同反向传播路径的梯度，对迁移性是否具有不同作用？**

## 2. 残差连接与梯度传播

普通残差块为：

$$
x_{l+1}=x_l+F_l(x_l).
$$

其中 $x_l$ 通过两条路径传播：

- **恒等分支（Identity/Skip Connection）**：直接将 $x_l$ 传到输出；
- **残差分支（Residual Branch）**：计算 $F_l(x_l)$。

令 $J_l=\partial F_l(x_l)/\partial x_l$，则

$$
\frac{\partial x_{l+1}}{\partial x_l}=I+J_l.
$$

若采用列向量梯度记法，反向传播为

$$
g_l=(I+J_l)^Tg_{l+1}.
$$

其中 $g_l=\nabla_{x_l}L$。

## 3. SGM 的核心机制

SGM 将反向传播修改为

$$
\boxed{g_l=(I+\gamma J_l)^Tg_{l+1}},\qquad 0\le\gamma\le1.
$$

- $\gamma=1$：普通反向传播；
- $0<\gamma<1$：降低残差分支梯度贡献；
- $\gamma=0$：对于具有恒等捷径的残差块，仅保留恒等分支的梯度。

**重要：SGM 不直接改变前向公式** $x_{l+1}=x_l+F_l(x_l)$，而是在生成对抗样本时修改所使用的反向梯度。

## 4. 为什么可能改善迁移性？

对多个残差块，反向传播包含若干 $(I+J_l)$ 的矩阵乘积。SGM 将其替换为 $(I+\gamma J_l)$。

例如两个残差块，忽略矩阵转置与乘积顺序的记号细节，展开：

$$
(I+\gamma J_2)(I+\gamma J_1)
=I+\gamma J_1+\gamma J_2+\gamma^2J_2J_1.
$$

可以看到：

- 不经过残差分支的路径权重为 $1$；
- 经过一个残差分支的路径权重为 $\gamma$；
- 经过两个残差分支的路径权重为 $\gamma^2$。

因此，$\gamma<1$ 会相对抑制经过较多残差分支的梯度路径。原始研究通过实验表明，这样调整梯度在一些跨模型攻击设置中可以改善迁移性。可能的解释是，部分残差路径所携带的梯度更依赖代理模型特有的表示，而跳跃路径的梯度更容易跨模型共享。

**注意**：这是有实验支持的机制解释，而不是“所有残差梯度都是噪声”或“$\gamma$ 越小越好”的定理。

## 5. SGM 与 I-FGSM 的结合

普通非定向 I-FGSM：

$$
x_{t+1}^{adv}=\Pi_{\mathcal B_\epsilon(x)}\left(x_t^{adv}+\alpha\operatorname{sign}(\nabla_xL(f(x_t^{adv}),y))\right).
$$

结合 SGM 后：

$$
\boxed{x_{t+1}^{adv}=\Pi_{\mathcal B_\epsilon(x)}\left(x_t^{adv}+\alpha\operatorname{sign}(g_t^{SGM})\right)}.
$$

其中 $g_t^{SGM}$ 是经过修改的反向传播计算得到的输入梯度方向。它未必是原损失函数的真实数学梯度。SGM 同样可以与 MI-FGSM、DI-FGSM 等攻击方法结合。

## 6. PyTorch 原理示例

以下代码演示**保持前向值不变、缩放残差分支反向梯度**的思想：

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.branch = nn.Linear(dim, dim)

    def forward(self, x, gamma=1.0):
        residual = self.branch(x)
        residual_sgm = (
            gamma * residual
            + (1.0 - gamma) * residual.detach()
        )
        return x + residual_sgm
```

因为 `detach()` 不参与反向传播，所以前向值仍是

$$
x+\gamma F(x)+(1-\gamma)F(x)=x+F(x),
$$

但反向雅可比变为

$$
I+\gamma J_F.
$$

这只是解释机制的最小示例，不是可以直接套用到所有 ResNet 或 ViT 的完整攻击实现。实际模型可能包含投影捷径、下采样、归一化及其他非线性操作，需要按结构处理。

## 7. SGM 与 ViT Attack

Transformer Block 也包含残差连接，例如 Pre-LN 结构：

$$
X'=X+\operatorname{MSA}(\operatorname{LN}(X)),
$$

$$
Y=X'+\operatorname{MLP}(\operatorname{LN}(X')).
$$

因此可以研究对 MSA 和 MLP 分支分别衰减反向梯度：

$$
I+\gamma_{MSA}J_{MSA},\qquad I+\gamma_{MLP}J_{MLP}.
$$

但原始 SGM 的主要实验对象是残差卷积网络；其结论不能不经验证就直接推广到 ViT。对于 ViT 迁移攻击，进一步研究不同模块的梯度贡献、衰减系数与跨架构迁移效果是有意义的方向。

## 8. 优势与局限

| 优势 | 局限 |
|---|---|
| 不需要重新训练代理模型 | 需要能修改代理模型的反向传播 |
| 可结合已有梯度攻击 | 衰减系数 $\gamma$ 需要调参 |
| 前向预测保持不变 | 不保证所有模型和任务均提升迁移性 |
| 为结构感知型迁移攻击提供思路 | 复杂捷径结构需要专门处理 |

## 9. 推荐阅读

1. **Skip Connections Matter: On the Transferability of Adversarial Examples Generated with ResNets**（ICLR 2020）：SGM 原始论文。https://arxiv.org/abs/2002.05990
2. **SGM 代码仓库**：https://github.com/csdongxian/skip-connections-matter
3. **On the Adversarial Transferability of Generalized “Skip Connections”**：关于更一般跳跃连接与对抗迁移性的后续研究。https://arxiv.org/abs/2410.08950

## 10. 总结

$$
\boxed{I+J_F\quad\longrightarrow\quad I+\gamma J_F}
$$

SGM 的本质是：**通过调整残差网络不同反向传播路径的权重，构造更适合跨模型迁移的攻击梯度。** 它是一种梯度修改策略，而不是新的损失函数或独立的基础攻击框架。
