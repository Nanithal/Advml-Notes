---
title: "RMSprop - 自适应学习率"
aliases:
  - "自适应学习率RMSProp"
分类: "优化方法"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/优化方法"
状态: "已整理"
摘要: "围绕“RMSprop - 自适应学习率”，整理为什么要引入 RMSprop？（解决的痛点）、RMSprop 的核心思想、标准数学公式。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/优化方法/总结 - 优化方法]]"
相关笔记:
  - "[[10_基础与算法/数学基础/EMA - 指数加权平均]]"
  - "[[10_基础与算法/优化方法/Adam - 自适应矩估计]]"
论文条目: []
论文标题: []
发表载体: []
发表年份: []
发表类型: []
发表链接: []
预印本链接: []
---

[[首页|首页]] · [[10_基础与算法/优化方法/总结 - 优化方法|优化方法总结]]

# RMSprop - 自适应学习率

在了解了基于“一阶动量（惯性方向）”的 Momentum 之后，**RMSprop（Root Mean Square Propagation）** 代表了优化算法发展的另一个核心维度：**自适应学习率（Adaptive Learning Rate）**。

Momentum 解决的是“**更新方向**”的问题，而 RMSprop 解决的是“**更新步长（学习率）**”的问题。

### 一、 为什么要引入 RMSprop？（解决的痛点）

在复杂的损失曲面上，经常会出现**病态曲率（Pathological Curvature）**：

- 在某些维度上（如峡谷的两壁），陡峭度极大，梯度 $g_t$ 非常大。
    
- 在某些维度上（如峡谷底部延伸方向），坡度极平缓，梯度 $g_t$ 非常小。
    

如果所有参数使用**同一个固定的全局学习率 $\alpha$**：

- 学习率设大了，陡峭方向会直接爆炸或剧烈震荡。
    
- 学习率设小了，平缓方向更新缓慢，导致训练停滞。
    

#### RMSprop 的核心思想

**“频繁更新/梯度大的参数，降低其学习率；稀疏更新/梯度小的参数，放大其学习率。”**

RMSprop 通过为**每个参数独立调节学习率**，实现了更新步长的自适应控制。

### 二、 标准数学公式

RMSprop 采用**指数加权移动平均（EMA）**来追踪梯度**平方**的历史累积量（通常称为**二阶动量 $v_t$**）。

在第 $t$ 次迭代中：

#### 1. 计算当前梯度

$$g_t = \nabla_\theta J(\theta_t)$$

#### 2. 更新二阶动量（梯度的平方均值）

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

_(注：$g_t^2$ 表示梯度的按元素平方，即按维度的点乘。)_

#### 3. 自适应缩放学习率并更新参数

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t} + \epsilon} \odot g_t$$

### 三、 参数与符号详细拆解

|**符号**|**名称**|**常用默认值**|**含义与作用**|
|---|---|---|---|
|**$v_t$**|**二阶动量**|初始为 $0$|记录各参数维度上历史梯度的**平方移动平均**|
|**$\beta_2$**|**二阶衰减系数**|**$0.99$ 或 $0.999$**|控制二阶动量的衰减率（通常比 Momentum 的 $\beta$ 更大，平滑范围更长）|
|**$\epsilon$**|**平滑项 (Epsilon)**|**$10^{-8}$**|一个极小的极小值，防止分母除以零|
|**$\frac{\alpha}{\sqrt{v_t} + \epsilon}$**|**自适应学习率**|—|**核心部分**：每个参数独立的动态步长|

### 四、 为什么 RMSprop 有效？（深入机制）

我们可以从以下两种场景观察公式中分母 $\sqrt{v_t}$ 的调节机制：

1. **对于梯度的陡峭/震荡方向**：
    
    - 历史梯度 $g_t$ 模长很大 $\rightarrow$ $g_t^2$ 非常大 $\rightarrow$ 累积的 $v_t$ 很大。
        
    - 分母 $\sqrt{v_t}$ 变大 $\rightarrow$ 该维度的有效步长 $\frac{\alpha}{\sqrt{v_t}}$ 被**自动缩小**，从而抑制了剧烈震荡。
        
2. **对于梯度的平缓/稀疏方向**：
    
    - 历史梯度 $g_t$ 模长很小 $\rightarrow$ $g_t^2$ 很小 $\rightarrow$ 累积的 $v_t$ 很小。
        
    - 分母 $\sqrt{v_t}$ 变小 $\rightarrow$ 该维度的有效步长 $\frac{\alpha}{\sqrt{v_t}}$ 被**自动放大**，从而加速了平缓区域的推进。
        

### 五、 历史脉络：AdaGrad 与 RMSprop 的关系

了解 RMSprop 必须要提及它的前身 **AdaGrad**：

- **AdaGrad 的公式**：$v_t = v_{t-1} + g_t^2$ （直接将历史梯度的平方简单累加）。
    
    - **缺陷**：因为 $g_t^2 \ge 0$，二阶动量 $v_t$ 会随着训练不断**单调递增**。导致分母越来越大，学习率会在训练中后期**过早衰减至 0**，使模型无法继续学习新知识。
        
- **RMSprop 的改进**：Geoffrey Hinton 提出了引入衰减系数 $\beta_2$（指数加权平均），使得 $v_t$ 更加关注**最近一段时间**的梯度大小，摆脱了 AdaGrad 学习率无限衰减停滞的缺陷。
    

### 六、 NumPy 面向对象代码实现

下面是符合工业级标准的 RMSprop 纯 Python/NumPy 实现：

Python

```
import numpy as np

class RMSpropOptimizer:
    def __init__(self, lr=0.001, beta2=0.99, epsilon=1e-8):
        self.lr = lr
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = None  # 初始化二阶动量

    def update(self, params, grads):
        # 首次迭代延迟初始化 v，形状与参数相同
        if self.v is None:
            self.v = np.zeros_like(params)

        # 1. 更新二阶动量 (梯度的平方指数移动平均)
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)

        # 2. 计算自适应步长并更新参数: theta = theta - (alpha / (sqrt(v) + eps)) * g
        adaptive_lr = self.lr / (np.sqrt(self.v) + self.epsilon)
        updated_params = params - adaptive_lr * grads

        return updated_params

# --- 验证逻辑 ---
if __name__ == "__main__":
    opt = RMSpropOptimizer(lr=0.01, beta2=0.99)
    # 假设有两个参数，w0 梯度极大(10.0)，w1 梯度极小(0.1)
    w = np.array([1.0, 1.0])
    grads = np.array([10.0, 0.1])

    for epoch in range(3):
        w = opt.update(w, grads)
        print(f"Step {epoch+1}: w = {np.round(w, 4)}, v = {np.round(opt.v, 4)}")
```

### 七、 总结与对比：Momentum vs RMSprop

| **维度**   | **Momentum（动量法）**                                  | **RMSprop（均方根传播）**      |
| -------- | -------------------------------------------------- | ----------------------- |
| **优化核心** | 改变**更新方向**（冲量/惯性）                                  | 改变**更新步长**（自适应缩放）       |
| **依靠工具** | **一阶动量** $m_t$（梯度的均值）                              | **二阶动量** $v_t$（梯度的平方均值） |
| **物理对应** | 给物体赋予“惯性速度”                                        | 给不同维度的地势设定“独立脚印步长”      |
| **经典组合** | 将两者结合即为著名的 **Adam 优化器**（$m_t$ 提供方向，$v_t$ 提供自适应步长）。 |                         |

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/数学基础/EMA - 指数加权平均|EMA - 指数加权平均]]：理解二阶统计量的平滑方式。
- [[10_基础与算法/优化方法/Adam - 自适应矩估计|Adam - 自适应矩估计]]：从自适应步长延伸到 Adam。

<!-- knowledge-links:end -->
