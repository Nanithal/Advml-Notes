---
title: "Momentum - 带动量的梯度下降"
aliases:
  - "Momentum 带动量的SGD"
分类: "优化方法"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/优化方法"
状态: "已整理"
摘要: "围绕“Momentum - 带动量的梯度下降”，整理数学定义与标准更新公式、参数与符号详解：、几何直觉：Momentum 解决了什么问题？。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/优化方法/总结 - 优化方法]]"
相关笔记:
  - "[[10_基础与算法/数学基础/EMA - 指数加权平均]]"
  - "[[10_基础与算法/优化方法/Adam - 自适应矩估计]]"
  - "[[40_课程笔记/吴恩达机器学习/梯度下降 - 课程笔记]]"
  - "[[30_对抗机器学习/攻击方法/MIM - 动量迭代攻击详解]]"
论文条目: []
论文标题: []
发表载体: []
发表年份: []
发表类型: []
发表链接: []
预印本链接: []
---

[[首页|首页]] · [[10_基础与算法/优化方法/总结 - 优化方法|优化方法总结]]

# Momentum - 带动量的梯度下降


在神经网络和机器学习的优化算法中，**Momentum（动量法）** 是一种用于加速梯度下降（Gradient Descent）收敛、并减少参数更新过程中剧烈震荡的技术。

标准梯度下降（GD/SGD）只关注“**当前这一步的梯度**”，就像一个没有记忆盲目下山的人，极易在陡峭的峡谷两壁来回碰撞震荡，或者在平坦的区域停滞不前。而 Momentum 引入了**物理学中的“惯性”概念**，让参数更新不仅取决于当前的梯度，还取决于**历史更新的方向和速度**。

### 一、 数学定义与标准更新公式

在现代深度学习理论以及 Adam 等后续优化器的标准定义中，Momentum 采用**指数加权移动平均（Exponentially Weighted Moving Average, EMA）来对历史梯度进行平滑，得到一个**平滑后的方向向量（称为一阶动量 $m_t$）。

在第 $t$ 次迭代中，假设模型参数为 $\theta_t$，损失函数关于参数的当前梯度为 $g_t$：

1. **计算当前梯度**：
    
    $$g_t = \nabla_\theta J(\theta_t)$$
    
2. **更新一阶动量（累积历史方向与惯性）**：
    
    $$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t$$
    
3. **更新模型参数**：
    
    $$\theta_{t+1} = \theta_t - \alpha \cdot m_t$$
    

#### 参数与符号详解：

- **$\theta_t$**：当前时刻的模型参数。
    
- **$g_t$**：当前 Batch 算出的原始梯度（方向陡峭，但可能带有较多随机噪声）。
    
- **$m_t$**：结合了历史惯性后的“平滑梯度（速度）”，初始状态 $m_0 = 0$。
    
- **$\beta$（动量衰减系数）**：通常设为 **$0.9$**。表示保留 $90\%$ 的历史速度，同时融入 $10\%$ 的当前梯度信息。几何上相当于平滑了最近约 $\frac{1}{1-\beta} = 10$ 步的梯度。
    
- **$\alpha$（学习率 Learning Rate）**：控制最终迈出的步长大小。
    

### 二、 几何直觉：Momentum 解决了什么问题？

#### 1. 抑制高频震荡（峡谷地形 / 病态曲率）

在损失曲面呈现狭长“沟壑/峡谷”形状时，梯度方向会垂直于沟壑壁：

- **横向（两侧反弹方向）**：第 1 步向左下，第 2 步向右下……梯度的正负符号频繁交替。在公式 $m_t = \beta m_{t-1} + (1-\beta)g_t$ 中，正负梯度互相抵消，**震荡被大幅削弱**。
    
- **纵向（沿着谷底下山方向）**：连续多步梯度的符号完全一致。正向梯度不断累加，**前进速度越来越快**。
    

#### 2. 惯性冲过鞍点与平坦区

在**鞍点（Saddle Point）**或**梯度消失区**附近，当前真实梯度 $g_t \approx 0$：

- 纯 GD 算出的更新量接近 0，训练容易直接停滞。
    
- 动量法中，哪怕当前 $g_t = 0$，动量项 $m_t = \beta \cdot m_{t-1}$ 依然保留着上一时刻的冲劲，**凭着惯性直接冲过鞍点**。
    

### 三、 进阶变体：Nesterov 动量 (NAG)

标准 Momentum 是“在当前位置看梯度，再加上历史动量”。

而 **Nesterov Accelerated Gradient（NAG，前瞻动量）** 引入了**预判机制（Look-ahead）**：**“先顺着动量走一步，到预判的新位置再看梯度”**。

#### NAG 的标准公式：

1. **预测下一步可能到达的位置**：
    
    $$\theta_{\text{lookahead}} = \theta_t - \alpha \cdot \beta \cdot m_{t-1}$$
    
2. **在前瞻位置计算梯度**：
    
    $$g_t = \nabla_\theta J(\theta_{\text{lookahead}})$$
    
3. **更新动量与参数**：
    
    $$m_t = \beta \cdot m_{t-1} + (1 - \beta) \cdot g_t$$
    
    $$\theta_{t+1} = \theta_t - \alpha \cdot m_t$$
    

> **形象对比**：普通动量像是一个闭眼往前冲的滚石；NAG 则是在冲锋前提前往前看一眼陡峭程度，如果发现前方是上坡（梯度反向），会**提前减速刹车**，有效避免过冲（Overshoot）。

### 四、 NumPy 面向对象实现

使用 Python 和 NumPy 编写一个符合工业标准的 Momentum 优化器类：

Python

```
import numpy as np

class MomentumOptimizer:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.m = None  # 动量向量，延迟初始化

    def update(self, params, grads):
        # 首次调用时初始化 m 的形状与参数一致
        if self.m is None:
            self.m = np.zeros_like(params)
        
        # 1. 动量更新公式 (指数加权平均): m_t = beta * m_{t-1} + (1 - beta) * g_t
        self.m = self.beta * self.m + (1.0 - self.beta) * grads
        
        # 2. 参数更新公式: theta_{t+1} = theta_t - alpha * m_t
        updated_params = params - self.lr * self.m
        
        return updated_params

# --- 验证逻辑 ---
if __name__ == "__main__":
    opt = MomentumOptimizer(lr=0.1, beta=0.9)
    w = np.array([5.0, -4.0])  # 初始参数
    
    for epoch in range(3):
        grad = w * 2.0  # 假设损失函数 J(w) = w^2，梯度的真正值为 2w
        w = opt.update(w, grad)
        print(f"Step {epoch+1}: w = {np.round(w, 4)}, m = {np.round(opt.m, 4)}")
```

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/数学基础/EMA - 指数加权平均|EMA - 指数加权平均]]：动量法是指数加权平均的优化应用。
- [[10_基础与算法/优化方法/Adam - 自适应矩估计|Adam - 自适应矩估计]]：从一阶动量延伸到自适应矩估计。
- [[40_课程笔记/吴恩达机器学习/梯度下降 - 课程笔记|梯度下降 - 课程笔记]]：回顾基本梯度下降的更新步骤。
- [[30_对抗机器学习/攻击方法/MIM - 动量迭代攻击详解|MIM - 动量迭代攻击详解]]：比较训练优化中的动量与攻击优化中的动量。

<!-- knowledge-links:end -->
