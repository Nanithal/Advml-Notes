---
title: "Lion - 优化器详解"
aliases:
  - "Lion优化器"
分类: "优化方法"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[10_基础与算法/优化方法/索引 - 优化方法|优化方法索引]]

# Lion - 优化器详解

## Lion讲解

**Lion（EvoLved Sign Momentum）** 是由 Google Research 团队于 2023 年利用**符号架构搜索（Symbolic Program Search）**自动发现并优化出的一种**极简、高效的深度学习优化器**。

在 Large Language Models (LLM) 和 Vision Transformer (ViT) 爆发的时代，**AdamW** 几乎成为了训练大模型的默认标配。然而，Lion 的出现刷新了人们对优化器的认知：**它不仅在多个大模型训练任务上超越了 AdamW 的性能，还将优化器的显存占用（Memory Footprint）直接削减了近半**。

为了让你全面且透彻地理解 Lion，我们将从 **“诞生背景与核心动机 $\rightarrow$ 算法推导与核心公式 $\rightarrow$ 与 AdamW 的三大本质对比 $\rightarrow$ 核心超参数调优指南 $\rightarrow$ PyTorch 原生实现 $\rightarrow$ 适用场景与局限”** 进行全景拆解。

### 一、 诞生背景：为什么要寻找 AdamW 的替代者？

#### 1. AdamW 的显存瓶颈

AdamW 在训练大模型（如 LLaMA、GPT-3）时极其消耗 GPU 显存。因为对于模型的**每一个可训练参数**，AdamW 都需要维护两个历史状态：

- **一阶矩（First Moment, $m$）**：梯度的指数移动平均（EMA），用于估计动量。
    
- **二阶矩（Second Moment, $v$）**：梯度平方的 EMA，用于自适应调整学习率。
    

如果模型有 700 亿（70B）参数，仅保存 $m$ 和 $v$ 两个优化器状态就需要消耗高达 **560 GB** 的显存（以 FP32 存储计算），这甚至超过了参数本身占用的内存！

#### 2. 算法进化：让 AI 来设计算法

Google 团队放弃了人类基于直觉推导优化器的传统方法，而是通过定义一系列基本代数算子（如加减乘除、提取符号 Sign、计算点积等），使用**符号架构搜索算法**在庞大的搜索空间中进行了数百万次试验，最终筛选并演化出了这个被称为 **Lion** 的算法。

### 二、 Lion 的核心算法原理与数学推导

Lion 的核心物理直觉可以概括为两句话：

1. **纯粹的符号方向（Sign-based Direction）**：只关心梯度的**更新方向（+1 或 -1）**，完全放弃对梯度模长/大小的估计。
    
2. **动量分离机制**：解耦“用于**计算当前步更新方向**的动量”与“用于**传给下一步历史累积**的动量”。
    

#### Lion 的更新迭代公式

在时间步 $t$，假设损失函数相对于参数 $\theta_t$ 的梯度为 $g_t$，动量状态为 $m_{t-1}$：

1. **计算当前步的更新方向向量 $c_t$**：
    
    $$c_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
    
    - 这里用超参数 $\beta_1$ 对上一步的动量 $m_{t-1}$ 和当前梯度 $g_t$ 进行插值，混合出一个即时的动量估计 $c_t$。
        
2. **更新模型参数 $\theta_t$**：
    
    $$\theta_t = \theta_{t-1} - \eta_t \cdot \left( \text{sign}(c_t) + \lambda \theta_{t-1} \right)$$
    
    - **$\text{sign}(c_t)$**：对 $c_t$ 逐元素取符号函数（元素为正取 $+1$，为负取 $-1$，为零取 $0$）。
        
    - **$\eta_t$**：第 $t$ 步的学习率。
        
    - **$\lambda \theta_{t-1}$**：解耦的权重衰减（Decoupling Weight Decay）。
        
    - **关键点**：注意！无论梯度的具体数值有多大或多小，每次更新的步骤长度**全由 $\eta_t$ 严格控制，更新量中的每一个元素模长都是 1**。
        
3. **更新并保存下一代历史动量状态 $m_t$**：
    
    $$m_t = \beta_2 m_{t-1} + (1 - \beta_2) g_t$$
    
    - 用超参数 $\beta_2$ 计算新的历史动量 $m_t$，并保存供下一轮使用。
        

### 三、 Lion 与 AdamW 的三大本质对比

|**对比维度**|**AdamW**|**Lion (EvoLved Sign Momentum)**|
|---|---|---|
|**优化器状态数量**|**2 个**（一阶矩 $m$ + 二阶矩 $v$）|**1 个**（仅需维护历史动量 $m$）|
|**显存占用**|高（$2 \times$ 参数量大小）|**超低（$1 \times$ 参数量大小，省下 50% 优化器显存）**|
|**更新量数值（Step size）**|自适应（由 $\frac{m}{\sqrt{v} + \epsilon}$ 决定）|**各维度统一固定**（由 $\text{sign}(c_t)$ 缩放）|
|**计算开销**|涉及开方、除法等开销较高的逐元素计算|**仅包含加法、乘法与 Sign 提取**（硬件加速极友好）|
|**推荐默认学习率**|基准学习率 $\eta_0$|**通常为 AdamW 的 $\frac{1}{3} \sim \frac{1}{10}$**|
|**推荐默认 Weight Decay**|通常为 $0.01 \sim 0.1$|**通常为 AdamW 的 $3 \sim 10$ 倍**|

#### 为什么 Lion 可以节省大量显存？

因为 Lion 抛弃了 AdamW 中的二阶矩 $v$（即不需要记录 $\nabla L^2$ 的历史平滑值），因此只需要维护一个与模型参数量等大的动量张量 $m$。**在训练百亿/千亿级大语言模型时，这相当于免费省出了几百 GB 的显存，能让你用更少的 GPU 跑更大的 Batch Size。**

### 四、 Lion 核心超参数调优指南

Lion 的超参数与 AdamW 有着明显不同的数量级特征，**切记不要直接套用 AdamW 的参数设置**！

1. **学习率（Learning Rate, $\eta$）**：
    
    - **调优法则**：通常设置为 AdamW 最佳学习率的 **$1/3$ 至 $1/10$**。
        
    - **原因**：因为 Lion 使用了 $\text{sign}(\cdot)$ 函数，其更新向量每个维度的元素模长都是 1，而 AdamW 的更新步长经过平方根缩放后通常远小于 1。如果使用与 AdamW 相同的学习率，Lion 会导致训练发散。
        
2. **超参数 $\beta_1$ 与 $\beta_2$**：
    
    - **默认推荐**：**$\beta_1 = 0.9, \beta_2 = 0.99$**。
        
    - **作用区隔**：
        
        - $\beta_1$ 控制当前迭代步用于计算 $\text{sign}$ 方向的短时动量权值。
            
        - $\beta_2$ 控制存入内存、传给下一步的长时历史动量权值。
            
3. **权重衰减（Weight Decay, $\lambda$）**：
    
    - **调优法则**：通常设置为 AdamW 最佳 Weight Decay 的 **$3 \sim 10$ 倍**。
        
    - **原因**：在 Lion 中，参数更新量是 $\eta_t (\text{sign}(c_t) + \lambda \theta)$。由于 $\text{sign}(c_t)$ 的元素绝对值为 1，为了让权重衰减项（$\lambda \theta$）与更新量维持相对相等的比例，$\lambda$ 需要相应调大。
        

### 五、 PyTorch 代码实现

以下是符合 PyTorch 官方 API 规范的纯原生 **Lion 优化器代码实现**：

Python

```
import torch
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        """
        Lion 优化器官方数学实现
        :param params: 待优化参数 Iterable
        :param lr: 学习率 (建议为 AdamW 的 1/3 ~ 1/10)
        :param betas: (beta1, beta2) 动量平滑系数，默认 (0.9, 0.99)
        :param weight_decay: 权重衰减系数 (建议为 AdamW 的 3 ~ 10 倍)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')

                state = self.state[p]

                # 优化器状态初始化: 仅需维护一个动量 m (exp_avg)
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                lr = group['lr']
                weight_decay = group['weight_decay']

                # 1. 逐步计算权重衰减 (Decoupled Weight Decay)
                if weight_decay != 0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # 2. 计算当前步的更新方向向量 c_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)

                # 3. 参数更新: theta_t = theta_{t-1} - lr * sign(c_t)
                p.add_(torch.sign(update), alpha=-lr)

                # 4. 更新历史动量状态: m_t = beta2 * m_{t-1} + (1 - beta2) * g_t
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss

# --- 验证 Lion 优化器可用性 ---
if __name__ == "__main__":
    # 模拟简单模型
    model = torch.nn.Linear(10, 2)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # 模拟一步前向/反向传播
    x = torch.randn(5, 10)
    loss = model(x).sum()
    loss.backward()

    # 更新参数
    optimizer.step()
    print("Lion 优化器成功完成一步更新！")
```

### 六、 Lion 的优缺点与适用场景

#### 优势：

1. **显著节省 GPU 显存**：将优化器状态显存直接减半，非常适合**超大语言模型（LLM）预训练/微调**与 **Vision Transformer（ViT）** 训练。
    
2. **计算速度加快**：不需要计算 AdamW 中较为繁重的逐元素平方根与除法，使得每一步的训练迭代速度提升大约 $2\% \sim 5\%$。
    
3. **大 Batch Size 下表现极为出色**：研究表明，当训练的 Batch Size 较大时，Lion 的性能提升更加明显。
    

#### 局限性与风险：

1. **对小 Batch Size 敏感度稍高**：如果 Batch Size 太小，梯度的噪声会急剧放大，$\text{sign}(\cdot)$ 函数可能会对噪声过于敏感，导致性能不如 AdamW。
    
2. **需要重新微调超参数**：无法无缝替换现有的 AdamW 代码（直接替换会导致训练崩溃），必须重新调整学习率与 Weight Decay。

## 为何Lion表现有时会优于Adam

对，你这个问题问到了 **Lion 最值得理解的地方**：

> **既然 Lion 放弃了 Adam 很重要的二阶动量 $v_t$，为什么它有时候反而比 Adam/AdamW 表现更好？**

关键答案是：

> **二阶动量不是“越多越好”。Adam 的 $v_t$ 虽然能自适应调整步长，但这种自适应有时候反而会干扰参数更新；Lion 用更简单的“方向 + 动量”可能产生更合适的更新。**

而且要先纠正一个容易产生的误解：

**Lion 并不是在所有任务上都比 AdamW 好。**它是在一些任务、模型和调参条件下表现更好，尤其是在大模型训练场景中有优势。

---

# 1. 先看 Adam 到底为什么需要二阶动量

Adam 有：

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$

这是**一阶动量**，主要负责：

> “往哪个方向走？”

然后还有：

$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

这是**二阶矩**，主要负责：

> “这个方向的梯度有多大？”

最后：

$$\Delta\theta = -\eta \frac{m_t}{\sqrt{v_t}+\epsilon}$$

所以 Adam 实际上是：

```
梯度
 │
 ├──────────────→ 一阶动量 m
 │                    │
 │                    │ 方向
 │                    ▼
 │
 └──────────────→ 二阶矩 v
                      │
                      │ 梯度尺度
                      ▼
                 m / √v
                      │
                      ▼
                   参数更新
```

这看起来非常合理。

但问题也恰恰出在：

$$\frac{m}{\sqrt v}$$

---

# 2. Adam 的“自适应”不一定永远是好事

举一个极端例子。

假设两个参数：

$$\theta_1,\theta_2$$

它们的梯度长期分别是：

$$g_1=10$$ $$g2=0.1$$

Adam 会认为：

> 第一个参数梯度特别大，所以我要把它的步长压小。

第二个参数梯度小：

> 那我给它更大的相对步长。

于是 Adam 实际上是在做一种：

$$\boxed{\text{不同参数，不同有效学习率}}$$

这就是 Adam 的核心优势。

但从另一个角度看：

> **Adam 在主动改变不同参数之间的更新比例。**

有些任务里，这种自适应反而不一定是最优的。

---

# 3. Lion 干脆不管梯度有多大

Lion 的核心是：

$$\boxed{ \Delta\theta = -\eta\operatorname{sign}(c_t) }$$

其中：

$$c_t= \beta_1m_{t-1}+(1-\beta_1)g_t$$

所以：

$$c_t=100$$

和：

$$c_t=0.001$$

只要都是正数：

$$\operatorname{sign}(c_t)=1$$

那么更新都是：

$$-\eta$$

这看起来非常粗暴。

但是它有一个好处：

> **不会因为某个参数过去的梯度特别大，就不断压低它未来的有效更新。**

---

# 4. Lion 真正厉害的地方：它更关注“方向”

你可以把：

### Adam

理解成：

> **方向 + 梯度尺度**

而 Lion：

> **方向 + 动量**

即：

$$\boxed{ Adam = 一阶动量 + 二阶矩 }$$ $$\boxed{ Lion = 一阶动量 + Sign }$$

所以 Lion 并不是“少了一个东西就更厉害”。

而是：

> **它认为对于某些深度学习问题，精确估计梯度尺度并不是最重要的事情。**

---

# 5. 一个非常重要的理解：神经网络的梯度尺度可能很乱

现代神经网络里面，不同参数的梯度尺度可能差别巨大。

比如：

```
参数 A：梯度 ≈ 100
参数 B：梯度 ≈ 0.01
参数 C：梯度 ≈ 0.000001
参数 D：梯度 ≈ 10
```

Adam 会通过：

$$\frac{m}{\sqrt v}$$

尽可能把这些不同尺度进行调整。

Lion 则相当于：

```
100       → +1
0.01      → +1
0.000001  → +1
-10       → -1
```

也就是说：

> **先把梯度尺度差异抹掉，再根据方向统一更新。**

这有点类似于：

> “我不管你坡有多陡，我只管现在总体上是往上还是往下。”

---

# 6. 那为什么不会因为“不看梯度大小”而训练崩掉？

因为还有一个东西：

$$\boxed{Momentum}$$

这非常关键。

Lion 不是：

$$\operatorname{sign}(g_t)$$

这么简单。

它使用：

$$\operatorname{sign} \left( \beta_1m_{t-1} + (1-\beta_1)g_t \right)$$

所以它不是只看当前梯度。

例如：

```
第1次： +1
第2次： +1
第3次： +1
第4次： -1
第5次： -1
```

Momentum 会保留历史趋势。

因此即使某一次梯度突然反向：

$$g_t<0$$

只要历史 Momentum 仍然很强：

$$m_{t-1}>0$$

最终：

$$c_t>0$$

那么 Lion 仍然可能：

$$\operatorname{sign}(c_t)=+1$$

这就减少了震荡。

---

# 7. 所以 Lion 可以理解成“过滤噪声”

这个角度特别重要。

假设真实优化方向是：

++

但是梯度里面有噪声：

```
真实趋势： + + + + + + + +

观测梯度： + + - + + - + +
```

如果直接 SGD：

```
+ → + → - → + → + → - → + → +
```

容易抖。

Lion：

```
梯度
 ↓
Momentum
 ↓
过滤短期波动
 ↓
Sign
 ↓
最终方向
```

因此：

> **Momentum 帮助 Lion 保留长期方向，而 Sign 进一步降低梯度尺度和噪声幅度对更新的影响。**

---

# 8. 还有一个非常现实的原因：省显存

这个其实是 Lion 在大模型领域非常重要的优势。

Adam：

```
参数 θ
   +
一阶矩 m
   +
二阶矩 v
```

Lion：

```
参数 θ
   +
一阶矩 m
```

因此优化器状态明显更少。

对于小模型：

> 这个优势可能没那么明显。

但是对于几十亿参数甚至更大的模型：

> **省下来的显存非常可观。**

这意味着你可能可以：

- 使用更大的 batch
- 训练更大的模型
- 减少显存压力
- 或者在相同显存下获得更高的训练效率

所以 Lion 的价值并不只是：

> “Loss 比 Adam 低。”

还有：

> **在相似资源下训练得更高效。**

---

# 9. 但为什么论文里会出现“Lion 比 Adam 好”？

这里要特别谨慎。

实验中可能出现：

$$Lion>AdamLion > Adam$$

但不能理解成：

$$\boxed{Lion永远>Adam}$$

更准确应该说：

在某些模型、数据集和超参数设置下，$$\boxed{ \text{在某些模型、数据集和超参数设置下，Lion可以优于Adam/AdamW} }$$

而且 Lion 往往需要重新调参数。

比如：

$$lr$$

和：

$$weight\ decay$$

都不能简单照搬 AdamW。

---

# 10. 一个很有意思的现象

你可以把 Adam 和 Lion 想成两个司机。

### Adam

司机说：

> “这个坡度是 10°，那个坡度是 2°，我要根据坡度调整不同的速度。”

所以：

$$\text{梯度大小} \rightarrow \text{影响步长}$$

---

### Lion

司机说：

> “我不管坡度具体是多少，我主要判断总体应该向左还是向右。”

所以：

$$\text{梯度方向} \rightarrow \text{决定更新方向}$$

然后 Momentum 告诉它：

> “别被某一个瞬间的坡度变化骗了，看看之前的趋势。”

这就是 Lion 的思想。

---

# 11. 但是这里还有一个更深的问题

你可能马上会问：

> **既然 Lion 不需要二阶矩，那为什么 Adam 当初还要设计二阶矩？**

这个问题非常好。

因为 Adam 的二阶矩解决的是一个真实问题：

$$\boxed{ 不同参数的梯度尺度差异很大 }$$

所以 Adam 的：

$$\frac{m}{\sqrt v}$$

可以自动进行尺度调整。

Lion 则选择：

$$\operatorname{sign}(m)$$

直接把尺度信息砍掉。

因此它们实际上是两种不同的哲学：

### Adam

> **“我要尽可能精细地估计每个参数应该走多远。”**

### Lion

> **“我宁愿牺牲梯度大小信息，也要获得简单、稳定、低内存的更新。”**

---

# 12. 你现在可以这样理解整个优化器发展

```
SGD
 │
 │  当前梯度
 ▼
Momentum
 │
 │  加入历史方向
 ▼
Adam
 │
 │  再加入梯度尺度自适应
 ▼
AdamW
 │
 │  把 Weight Decay 解耦
 ▼
Lion
 │
 │  不要二阶矩
 │  Momentum + Sign
 ▼
更简单、更省状态
```

所以 **Lion 并不是“因为少了二阶动量所以更好”**。

恰恰应该理解成：

> **它发现二阶动量带来的“自适应尺度”并非在所有场景都值得付出代价，于是用更简单的方向性更新换取更低的状态开销，并在一些任务上获得了很好的泛化/训练表现。**

---

## 最后给你一个最重要的对比

假设：

$$m=10,\quad v=100$$

Adam：

$$\frac{m}{\sqrt v} = \frac{10}{10} = 1$$

而 Lion：

$$\operatorname{sign}(m)=1$$

两者这里恰好一样。

但如果：

$$m=0.1,\quad v=100$$

Adam：

$$\frac{0.1}{10}=0.01$$

Lion：

$$\operatorname{sign}(0.1)=1$$

于是：

```
Adam：  更新 0.01η
Lion：  更新 1η
```

**这就是两者最本质的差别：**

> Adam 会根据“梯度有多大”调整更新幅度；Lion 基本不管梯度绝对大小，只根据 Momentum 的正负决定方向，然后用学习率统一控制更新幅度。

所以你如果现在继续深入优化器，下一步最值得搞懂的其实是 **“Adam 的二阶矩 vtv_t 到底在数学上解决了什么问题，以及为什么它有时会损害泛化”**。这个搞懂后，Lion 为什么这么设计就会非常自然。

---

## 笔记联系

- [[10_基础与算法/优化方法/AdamW - 解耦权重衰减|AdamW - 解耦权重衰减]]：对照 AdamW 的更新方式。
- [[10_基础与算法/优化方法/论文 - Symbolic Discovery of Optimization Algorithms|论文 - Symbolic Discovery of Optimization Algorithms]]：从算法讲解回到原论文阅读。
