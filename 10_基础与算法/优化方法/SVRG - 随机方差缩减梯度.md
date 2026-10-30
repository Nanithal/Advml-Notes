---
title: "SVRG - 随机方差缩减梯度"
aliases:
  - "SVRG：Stochastic Variance Reduced Gradient 随机方差梯度衰减"
分类: "优化方法"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/优化方法"
状态: "已整理"
摘要: "围绕“SVRG - 随机方差缩减梯度”，整理动机与痛点：GD vs SGD 的权衡、批量梯度下降（Batch Gradient Descent, GD）、随机梯度下降（Stochastic Gradient Descent, SGD）。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/优化方法/总结 - 优化方法]]"
相关笔记:
  - "[[10_基础与算法/优化方法/论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]]"
  - "[[40_课程笔记/吴恩达机器学习/梯度下降 - 课程笔记]]"
论文条目:
  - "[[论文来源目录#svrg]]"
论文标题:
  - "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
发表载体:
  - "NIPS"
发表年份:
  - "2013"
发表类型:
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[10_基础与算法/优化方法/总结 - 优化方法|优化方法总结]]

# SVRG - 随机方差缩减梯度


## 理论知识

**SVRG（Stochastic Variance Reduced Gradient，随机方差衰减梯度）** 算法是优化理论和机器学习领域极具里程碑意义的一个算法。它由 Rie Johnson 和 Tong Zhang（张潼）于 2013 年在 NIPS（现 NeurIPS）上提出，旨在解决**传统随机梯度下降（SGD）收敛慢**与**全梯度下降（GD）单步计算成本高**之间的矛盾。

SVRG 成功在维持与 SGD 相当的低单步计算复杂度的同时，实现了如全梯度下降一般的**线性收敛速率（Linear Convergence Rate）**。

下面我们将从 **背景痛点 $\rightarrow$ 核心思想 $\rightarrow$ 数学更新机制 $\rightarrow$ 算法推导与方差衰减原理 $\rightarrow$ 完整算法伪代码 $\rightarrow$ 与其他优化器对比 $\rightarrow$ PyTorch 代码实现** 进行全景拆解。

### 一、 动机与痛点：GD vs SGD 的权衡

在机器学习中，绝大多数优化问题都可以写成**有限和（Finite-sum）损失函数**的形式：

$$\min_{w \in \mathbb{R}^d} P(w) = \frac{1}{n} \sum_{i=1}^{n} f_i(w)$$

其中 $n$ 是训练样本总数，$f_i(w)$ 表示模型在第 $i$ 个样本上的损失。

#### 1. 批量梯度下降（Batch Gradient Descent, GD）

- **更新公式**：$w_{t+1} = w_t - \eta \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(w_t)$
    
- **优点**：梯度的估计是无偏且准确的，方差为 0。在强凸条件下具有**线性收敛速率**（即迭代 $O(\log(1/\epsilon))$ 次即可达到 $\epsilon$ 精度）。
    
- **缺点**：每一步更新都需要计算所有 $n$ 个样本的梯度，单步计算复杂度为 $O(n)$。在海量数据（如 $n = 10^7$）面前极其缓慢甚至不可行。
    

#### 2. 随机梯度下降（Stochastic Gradient Descent, SGD）

- **更新公式**：随机抽取一个样本 $i_t \in \{1, \dots, n\}$，更新 $w_{t+1} = w_t - \eta \nabla f_{i_t}(w_t)$
    
- **优点**：单步计算复杂度仅为 $O(1)$，极其高效。
    
- **致命缺点**：随机梯度 $\nabla f_{i_t}(w_t)$ 是真实梯度的**无偏估计**，但引入了极大的**方差（Variance）**：
    

$$\mathbb{E}[\nabla f_{i_t}(w_t)] = \nabla P(w_t), \quad \text{Var}(\nabla f_{i_t}(w_t)) \neq 0$$

当模型逼近极小值点 $w^*$ 时，由于单个样本的梯度 $\nabla f_{i_t}(w^*)$ 通常不为零，这种持续存在的方差会导致更新步在极小值附近不断震荡。为了保证收敛，SGD 必须使用**逐渐衰减的学习率**（如 $\eta_t = O(1/t)$），这使得 SGD 的收敛速率降为 **亚线性（Sublinear）**（$O(1/\epsilon)$ 或 $O(1/\sqrt{\epsilon})$）。

#### SVRG 的设计目标

**能否兼得两者的优点？** 即：**保持 SGD 的单步 $O(1)$ 计算成本，同时消除随机梯度的方差，实现 GD 的线性收敛速率。**

### 二、 核心思想：控制变量法与方差衰减

SVRG 的核心技巧是引入了一个**控制变量（Control Variate）**，采用“内外双层循环”的架构来消除方差。

#### 1. 锚点与快照（Snapshot）

SVRG 每隔一定的步数（例如 $m$ 步），就会固定一个当前的参数作为**锚点（Anchor Point）/ 快照 $\tilde{w}$**。在锚点处，算法会计算一次全量样本的“精确梯度”：

$$\mu = \nabla P(\tilde{w}) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(\tilde{w})$$

#### 2. 随机梯度的方差衰减修正

在随后的内层小步迭代中，当算法在点 $w_t$ 处随机抽取样本 $i_t$ 时，更新方向不再直接使用 $\nabla f_{i_t}(w_t)$，而是使用修正后的梯度 $v_t$：

$$v_t = \nabla f_{i_t}(w_t) - \nabla f_{i_t}(\tilde{w}) + \mu$$

这个修正梯度 $v_t$ 由三部分组成：

1. $\nabla f_{i_t}(w_t)$：当前迭代点在当前样本上的随机梯度。
    
2. $-\nabla f_{i_t}(\tilde{w})$：快照点在**同一样本**上的随机梯度（作为基准线扣除）。
    
3. $+\mu$：快照点在全量数据上的精确平均梯度。
    

### 三、 数学推导：为什么 $v_t$ 能做到无偏且方差衰减？

#### 1. 无偏性（Unbiased Estimator）证明

计算 $v_t$ 对随机抽取样本 $i_t$ 的数学期望（注意 $w_t, \tilde{w}, \mu$ 在当前步都是确定的）：

$$\begin{aligned} \mathbb{E}_{i_t}[v_t] &= \mathbb{E}_{i_t} \left[ \nabla f_{i_t}(w_t) - \nabla f_{i_t}(\tilde{w}) + \mu \right] \\ &= \mathbb{E}_{i_t}[\nabla f_{i_t}(w_t)] - \mathbb{E}_{i_t}[\nabla f_{i_t}(\tilde{w})] + \mu \\ &= \nabla P(w_t) - \nabla P(\tilde{w}) + \mu \end{aligned}$$

因为 $\mu = \nabla P(\tilde{w})$，上式后两项抵消：

$$\mathbb{E}_{i_t}[v_t] = \nabla P(w_t)$$

**结论**：$v_t$ 依然是当前真实全梯度 $\nabla P(w_t)$ 的**完全无偏估计**！

#### 2. 方差衰减（Variance Reduction）原理

考察 $v_t$ 的方差变化机制：

$$\text{Var}(v_t) = \text{Var}\left( \nabla f_{i_t}(w_t) - \nabla f_{i_t}(\tilde{w}) \right)$$

随着内层迭代的深入以及外层循环的轮转，$w_t$ 和快照 $\tilde{w}$ 都会逐渐收敛并靠近极小值点 $w^*$：

$$\lim_{k \to \infty} w_t = w^*, \quad \lim_{k \to \infty} \tilde{w} = w^*$$

当 $w_t$ 与 $\tilde{w}$ 非常接近时：

$$\nabla f_{i_t}(w_t) - \nabla f_{i_t}(\tilde{w}) \to 0$$

这意味着，**随着算法趋近收敛，估计量的方差自动衰减至零 ($\text{Var}(v_t) \to 0$)！**

正因为**方差会自动消失**，SVRG **不需要像 SGD 那样将学习率逐渐调小**，而是可以一直使用固定的常数学习率（Constant Learning Rate），从而直接冲向线性收敛。

### 四、 完整算法流程（伪代码）

SVRG 采用典型的**双重循环结构**：

Plaintext

```
输入: 初始权重 w_0, 学习率 η, 内层循环步数 m, 外层迭代轮数 S
对 s = 1, 2, ..., S 执行 (外层循环):
    1. 设置锚点快照: w~ = w_{s-1}
    2. 计算全量梯度: μ = (1/n) * ∑_{i=1}^n ∇f_i(w~)
    3. 设定内层起点: w_0^{(s)} = w~
    
    对 t = 0, 1, ..., m-1 执行 (内层循环):
        a. 随机从 {1, ..., n} 中无放回或有放回抽取一个样本索引 i_t
        b. 计算修正梯度: v_t^{(s)} = ∇f_{i_t}(w_t^{(s)}) - ∇f_{i_t}(w~) + μ
        c. 更新参数: w_{t+1}^{(s)} = w_t^{(s)} - η * v_t^{(s)}
        
    4. 更新外层快照 w_s (常用选项):
       - 选项 A (理论证明常用): 从 {w_0^{(s)}, ..., w_m^{(s)}} 中随机选取一个
       - 选项 B (实践工程常用): 直接取内层最后一步 w_s = w_m^{(s)}

输出: 最终参数 w_S
```

#### 复杂度分析

- **外层循环 1 次**：计算 1 次全量梯度，开销为 $n$ 次梯度计算。
    
- **内层循环 $m$ 步**：每一步需要计算 $\nabla f_{i_t}(w_t)$ 和 $\nabla f_{i_t}(\tilde{w})$ 两次单样本梯度，开销为 $2m$ 次梯度计算。
    
- **一轮外层循环总开销**：$n + 2m$ 次梯度计算。通常设置内层步数 $m = O(n)$（如 $m = 2n$），这样平摊到内层每一步的平均计算成本依然是 **$O(1)$** 级别。
    

### 五、 主流优化算法全方位对比

|**优化算法**|**每次迭代梯度计算次数**|**学习率选择**|**强凸条件下的收敛速率**|**内存/存储开销**|
|---|---|---|---|---|
|**GD (全梯度下降)**|$O(n)$|常数学习率 $\eta$|**线性收敛** $O(\log(1/\epsilon))$|$O(d)$|
|**SGD (随机梯度下降)**|$O(1)$|逐渐衰减 $O(1/t)$|**亚线性** $O(1/\epsilon)$|$O(d)$|
|**SAG / SAGA**|$O(1)$|常数学习率 $\eta$|**线性收敛** $O(\log(1/\epsilon))$|**$O(n d)$** (需存储所有样本旧梯度)|
|**SVRG**|$O(1)$ (平摊后)|常数学习率 $\eta$|**线性收敛** $O(\log(1/\epsilon))$|**$O(d)$** (只需存一个全量梯度 $\mu$)|

> **为什么 SVRG 比 SAG/SAGA 更受好评？**
> 
> SAG（Stochastic Average Gradient）也是经典的方差衰减算法，但它必须在内存中维持一个大小为 $n \times d$ 的表格来保存每个样本上一次的梯度值。当样本量 $n$ 和特征维度 $d$ 很大时，内存会爆掉。SVRG 巧妙地用外层定期计算一次全量梯度 $\mu$ 的方式，**把存储开销降回到了 $O(d)$**。

### 六、 PyTorch 实战实现：SVRG 求解逻辑回归

以下代码演示了如何在 PyTorch 中手写实现 SVRG 优化器，并在二分类数据集上求解带有 $L_2$ 正则化的逻辑回归。

Python

```
import torch
import torch.nn as nn
import numpy as np

# 1. 构造合成二分类数据集
torch.manual_seed(42)
n_samples, n_features = 2000, 10

X = torch.randn(n_samples, n_features)
true_w = torch.randn(n_features, 1)
y = (torch.mm(X, true_w) > 0).float() # 0 或 1 标签

# 2. 定义二元交叉熵损失函数 (加 L2 正则)
def compute_loss_and_grad(w, X_batch, y_batch, l2_reg=1e-4):
    """手写前向与梯度计算，方便展示 SVRG 的微观梯度操作"""
    z = torch.mm(X_batch, w)
    p = torch.sigmoid(z)
    
    # BCE 损失
    loss = nn.functional.binary_cross_entropy(p, y_batch) + 0.5 * l2_reg * torch.sum(w ** 2)
    
    # 梯度 ∇f_i(w) = (p - y) * x + l2_reg * w
    grad = torch.mm(X_batch.t(), (p - y_batch)) / X_batch.size(0) + l2_reg * w
    return loss, grad

# 3. SVRG 求解器
w = torch.zeros(n_features, 1, requires_grad=False) # 参数初始化

S = 10                  # 外层迭代轮数 (Epochs)
m = n_samples           # 内层循环步数 m = n
eta = 0.1               # 常数学习率

print("--- 开始 SVRG 优化训练 ---")

for s in range(S):
    # 【步骤 1】保存快照 w_tilde 并计算全量梯度 mu
    w_tilde = w.clone()
    _, full_grad = compute_loss_and_grad(w_tilde, X, y)
    mu = full_grad.clone()
    
    # 记录当前外层 loss
    current_loss, _ = compute_loss_and_grad(w, X, y)
    print(f"Outer Epoch [{s+1}/{S}], Loss: {current_loss.item():.6f}")
    
    # 【步骤 2】内层随机小步迭代
    w_inner = w.clone()
    
    # 打乱样本索引
    perm = torch.randperm(n_samples)
    
    for t in range(m):
        idx = perm[t]
        x_i = X[idx:idx+1] # 单个样本 Batch
        y_i = y[idx:idx+1]
        
        # a. 当前点在单样本上的梯度 ∇f_i(w_t)
        _, grad_current = compute_loss_and_grad(w_inner, x_i, y_i)
        
        # b. 快照点在同一样本上的梯度 ∇f_i(w_tilde)
        _, grad_snapshot = compute_loss_and_grad(w_tilde, x_i, y_i)
        
        # c. SVRG 方差衰减梯度更新方向: v_t = ∇f_i(w_t) - ∇f_i(w~) + μ
        v_t = grad_current - grad_snapshot + mu
        
        # d. 参数更新
        w_inner = w_inner - eta * v_t
        
    # 【步骤 3】更新快照参数
    w = w_inner.clone()

final_loss, _ = compute_loss_and_grad(w, X, y)
print(f"--- 训练结束，最终 Loss: {final_loss.item():.6f} ---")
```

### 七、 总结与适用场景

1. **核心创新**：SVRG 通过快照点（Snapshot）的全梯度 $\mu$ 配合单样本梯度差，设计出了一个**无偏且方差随收敛自动衰减**的梯度估计量 $v_t$。
    
2. **性能飞跃**：在凸与强凸问题（如逻辑回归、SVM、岭回归）上，实现了与全梯度下降一样的**线性收敛**，但计算成本却大幅降低到与 SGD 同阶。
    
3. **深度学习中的应用**：在非凸的深度神经网络（Deep Learning）中，由于存在大量局部极小值和 saddle points，SVRG 的理论线性收敛保证不再严格成立，且每隔一段时间计算一次全量梯度的开销过大，因此深度学习主流依然是 **AdamW、SGD with Momentum** 等。但在传统的机器学习大模型、凸优化以及联邦学习（Federated Learning）降低通信方差等领域，SVRG 依然是极为重要的核心算法。

## 详细流程

在实际应用中，使用 **SVRG（随机方差衰减梯度）** 算法的流程可以分为 **准备阶段、外层循环（快照与全梯度计算）、内层循环（方差衰减更新）以及收敛判定** 四个核心部分。

为了让你在工程实现或论文实验中能直接落地，下面详细梳理 SVRG 的**标准使用流程、参数设置规范以及代码落地框架**。

### 一、 SVRG 标准使用流程图

Plaintext

```
               [ 初始化参数 w_0, 超参数 (S, m, η) ]
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 外层循环 (s = 1, 2, ..., S)                                     │
│  1. 设定锚点快照: w̃ = w_{s-1}                                    │
│  2. 遍历全量数据集，计算精确全梯度: μ = (1/n) * Σ ∇f_i(w̃)       │
│  3. 重置内层起始点: w_0^(s) = w̃                                 │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ 内层循环 (t = 0, 1, ..., m-1)                           │   │
│   │  a. 从 {1, ..., n} 中随机抽取一个样本/小批量 (i_t)       │   │
│   │  b. 计算当前点梯度: g1 = ∇f_{i_t}(w_t^(s))              │   │
│   │  c. 计算快照点梯度: g2 = ∇f_{i_t}(w̃)                     │   │
│   │  d. 合成修正梯度: v_t = g1 - g2 + μ                     │   │
│   │  e. 参数更新: w_{t+1}^(s) = w_t^(s) - η * v_t           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  4. 更新外层参数 w_s (常用: w_s = w_m^(s))                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                     [ 达到最大轮数，输出 w_S ]
```

### 二、 详细步骤拆解

#### 阶段 1：准备与超参数配置

在开始训练前，需要确定以下三组关键超参数：

1. **初始权重 $w_0$**：可以随机初始化（如高斯分布）或全零初始化。
    
2. **学习率（步长）$\eta$**：
    
    - **不同于 SGD**：SVRG 无需随时间衰减学习率，**全程使用固定常数学习率**。
        
    - **取值经验**：根据理论，$\eta$ 应满足 $\eta < \frac{1}{4L}$（$L$ 为损失函数的 Lipschitz 常数）。工程上通常通过网格搜索（如 $0.1, 0.01, 0.001$）确定。
        
3. **内层循环步数 $m$**：
    
    - **理论推荐**：设为数据集样本量 $n$ 的 1~2 倍，即 **$m \approx n$ 或 $m = 2n$**。
        
    - **目的**：确保内层计算量与外层全梯度计算量相匹配，使平摊到每步的计算复杂度维持在 $O(1)$。
        
4. **外层循环总轮数 $S$**：即 Epochs 数量（通常 $10 \sim 50$ 轮即可达到极高精度）。
    

#### 阶段 2：外层循环（快照保存与全量梯度计算）

每一轮外层循环 $s$ 开始时：

1. **快照（Snapshot）打点**： 将当前模型参数保存为“锚点快照” $\tilde{w} = w_{s-1}$。
    
2. **全量梯度计算（Full Gradient Computation）**： 遍历整个数据集 $n$ 个样本，计算快照点处的精确平均梯度 $\mu$：
    
    $$\mu = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(\tilde{w})$$
    
    - _注意_：这一步是全局同步的，计算开销为 $O(n)$，它为后续内层迭代提供了消除方差的基准（Base Gradient）。
        
3. **内层初始化**： 设置内层迭代的初始点 $w_0^{(s)} = \tilde{w}$。
    

#### 阶段 3：内层循环（方差衰减与随机更新）

进行 $m$ 次内层小步迭代（$t = 0, 1, \dots, m-1$）：

1. **随机采样**： 从数据集 $\{1, 2, \dots, n\}$ 中随机抽取一个样本索引 $i_t$（或一个 Mini-batch）。
    
2. **计算双梯度**：
    
    - 计算当前点 $w_t^{(s)}$ 在该样本上的梯度：$g_1 = \nabla f_{i_t}(w_t^{(s)})$
        
    - 计算快照点 $\tilde{w}$ 在**同一个样本**上的梯度：$g_2 = \nabla f_{i_t}(\tilde{w})$
        
3. **合成方差衰减梯度 $v_t$**：
    
    $$v_t = \underbrace{\nabla f_{i_t}(w_t^{(s)})}_{g_1} - \underbrace{\nabla f_{i_t}(\tilde{w})}_{g_2} + \underbrace{\mu}_{\text{全量基准}}$$
    
4. **参数更新**： 以固定学习率 $\eta$ 沿着 $v_t$ 方向反向更新：
    
    $$w_{t+1}^{(s)} = w_t^{(s)} - \eta v_t$$
    

#### 阶段 4：外层更新与收敛判别

当内层循环走完 $m$ 步后：

1. **更新下一个外层锚点 $w_s$**：
    
    - **工程实践（Option A）**：直接取内层最后一步的结果，$w_s = w_m^{(s)}$。这是最常用且效果极佳的方式。
        
    - **理论证明（Option B）**：从内层产出的 $\{w_0^{(s)}, w_1^{(s)}, \dots, w_m^{(s)}\}$ 中**随机均匀选取一个**作为 $w_s$（在收敛性证明中更容易处理）。
        
2. **检查停止条件**：
    
    - 若达到设定轮数 $S$，或损失函数变化量小于阈值 $\Vert{}\nabla P(w_s)\Vert{} < \epsilon$，则停止训练；
        
    - 否则，进入下一个外层循环 $s+1$。
        

### 三、 实用工程优化策略

在实际代码编写或大规模数据集上使用 SVRG 时，通常会结合以下两项优化：

1. **Mini-batch SVRG（小批量变体）**：
    
    - 纯单样本（Batch Size = 1）更新在 GPU 或向量化 CPU 上效率较低。
        
    - 实践中，内层采样常改为采样大小为 $B$ 的小批量（Mini-batch $I_t$），梯度计算变为：
        
        $$v_t = \frac{1}{\vert{}I_t\vert{}} \sum_{i \in I_t} \nabla f_i(w_t^{(s)}) - \frac{1}{\vert{}I_t\vert{}} \sum_{i \in I_t} \nabla f_i(\tilde{w}) + \mu$$
        
2. **避免重复前向传播**：
    
    - 内层同时需要计算 $\nabla f_{i_t}(w_t^{(s)})$ 与 $\nabla f_{i_t}(\tilde{w})$。在计算 $g_2 = \nabla f_{i_t}(\tilde{w})$ 时，可以利用前向传播并行处理 $w_t^{(s)}$ 和 $\tilde{w}$，以充分利用硬件并行度。
        

### 四、 通用算法模版（Python 伪代码框架）

Python

```
def train_svrg(dataset, model, S, m, eta, batch_size):
    """
    通用 SVRG 训练流程
    :param dataset: 训练数据集 (N 个样本)
    :param model: 待优化的模型/参数
    :param S: 外层迭代轮数 (Epochs)
    :param m: 内层迭代次数 (通常 m = N / batch_size)
    :param eta: 固定学习率
    :param batch_size: 内层小批量大小
    """
    # 初始化
    w = model.get_params()
    
    for s in range(1, S + 1):
        # --- 步骤 1: 保存快照 ---
        w_snapshot = w.clone()
        
        # --- 步骤 2: 计算全量梯度 μ ---
        # 遍历整个数据集累加梯度并取平均
        mu = compute_full_gradient(dataset, model, w_snapshot)
        
        # --- 步骤 3: 内层循环准备 ---
        w_inner = w.clone()
        
        for t in range(m):
            # a. 随机抽取一个小批量 batch
            batch = dataset.sample_mini_batch(batch_size)
            
            # b. 计算当前参数在 batch 上的梯度 g1
            g1 = compute_batch_gradient(model, w_inner, batch)
            
            # c. 计算快照参数在同个 batch 上的梯度 g2
            g2 = compute_batch_gradient(model, w_snapshot, batch)
            
            # d. 合成 SVRG 修正梯度 v_t
            v_t = g1 - g2 + mu
            
            # e. 参数更新
            w_inner = w_inner - eta * v_t
            
        # --- 步骤 4: 更新锚点 ---
        w = w_inner # 或随机选取内层轨迹上的某一点
        
        print(f"Outer Loop {s}/{S} finished.")
        
    return w
```

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/优化方法/论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction|论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction]]：从算法推导回到提出该方法的论文。
- [[40_课程笔记/吴恩达机器学习/梯度下降 - 课程笔记|梯度下降 - 课程笔记]]：回顾全梯度下降，再比较随机梯度与方差缩减。
- [[50_汇报与写作/SVRG/总结 - SVRG汇报|SVRG汇报总结]]：对应主题的 PPT 与讲解材料。

## 论文出处

- [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)：NIPS，2013，会议论文。

<!-- knowledge-links:end -->
