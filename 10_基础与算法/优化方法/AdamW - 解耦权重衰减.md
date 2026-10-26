---
title: "AdamW - 解耦权重衰减"
aliases:
  - "AdamW"
分类: "优化方法"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[10_基础与算法/优化方法/索引 - 优化方法|优化方法索引]]

# AdamW - 解耦权重衰减

可以。**Adam 和 AdamW 最核心的区别只有一句话：**

> **Adam 把 L2 正则化和梯度更新混在了一起；AdamW 把 Weight Decay（权重衰减）从梯度更新中解耦出来。**

这个区别看起来很小，但对深度学习训练效果很重要。

---

# 1. 先说为什么需要 Weight Decay

假设模型参数是：

θ\theta

我们希望参数不要变得特别大，于是在损失函数中加入 L2 正则：

$$L'(\theta) = L(\theta)+\frac{\lambda}{2}\|\theta\|^2$$

对它求梯度：

$$\theta L' = \nabla_\theta L+\lambda\theta$$

于是普通 SGD：

$$\theta_{t+1} = \theta_t-\eta(\nabla L+\lambda\theta_t)$$

展开：

$$\theta_{t+1} = (1-\eta\lambda)\theta_t-\eta\nabla L$$

你会发现：

$$(1-\eta\lambda)\theta_t$$

会让参数逐渐变小。

这就是 **Weight Decay（权重衰减）**。

---

# 2. 如果使用 SGD，L2 正则和 Weight Decay 基本等价

这是理解 Adam/AdamW 的关键。

对于 SGD：

$$\theta_{t+1} = \theta_t-\eta(\nabla L+\lambda\theta_t)$$

可以写成：

$$\theta_{t+1} = \theta_t-\eta\nabla L-\eta\lambda\theta_t$$

也就是：

```
参数更新
   │
   ├── 梯度造成的更新
   │
   └── 参数衰减
```

所以在 SGD 中：

> **L2 正则化 ≈ Weight Decay**

但是到了 Adam，这两个东西就不再等价了。

---

# 3. Adam 是怎么做的？

Adam 首先计算：

$$g_t=\nabla_\theta L$$

然后：

$$m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$$

以及：

$$v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$$

最后：

$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t}+\epsilon}$$

---

# 4. 如果把 L2 正则直接放进 Adam

假设：

$$L'= L+\frac{\lambda}{2}\|\theta\|^2$$

那么梯度变成：

$$g_t = \nabla L+\lambda\theta$$

然后 Adam 会把这个整体梯度送进：

$$m_t$$

和：

$$v_t$$

也就是说：

$$\boxed{ \text{Adam：先把 }\lambda\theta\text{ 加到梯度里，再进行 Adam 自适应缩放} }$$

这就是问题所在。

---

# 5. 为什么这会有问题？

我们看一个非常简单的情况。

假设某个参数：

$$\theta=10$$

那么 L2 正则产生的梯度：

$$\lambda\theta$$

假设：

$$\lambda=0.01$$

那么：

$$\lambda\theta=0.1$$

这个 $0.1$ 会和真正的任务梯度混合：

$$g_{\text{total}} = g_{\text{loss}}+0.1$$

然后 Adam 再对它进行：

$$\frac{m}{\sqrt v}$$

的自适应调整。

所以：

> **Weight Decay 的效果被 Adam 的自适应机制改变了。**

它已经不再是我们直觉上想要的：

$$\theta\leftarrow(1-\eta\lambda)\theta$$

这种简单的参数衰减。

---

## 笔记联系

- [[10_基础与算法/优化方法/Adam - 自适应矩估计|Adam - 自适应矩估计]]：先回顾 Adam 的矩估计与更新过程。
- [[10_基础与算法/优化方法/Lion - 优化器详解|Lion - 优化器详解]]：比较两种优化器的更新方式与状态量。
- [[40_课程笔记/吴恩达机器学习/过拟合与欠拟合 - 课程笔记|过拟合与欠拟合 - 课程笔记]]：联系权重衰减与正则化。
