---
title: "RNN - 循环神经网络"
aliases:
  - "RNN循环神经网络"
分类: "网络结构"
笔记类型: "概念笔记"
整理日期: 2026-10-27
tags:
  - model
  - RNN循环神经网络
time: 2026-08-23T13:44:00
---

[[首页|首页]] · [[20_深度学习/网络结构/索引 - 网络结构|网络结构索引]]

# RNN - 循环神经网络

**序列数据：样本里面的各个数据点之间有先后顺序，顺序本身携带信息**。
序列：**顺序不能随便打乱，时序、位置先后是关键信息**

![[attachment/Pasted image 20260823135033.png]]

![[attachment/Pasted image 20260823135232.png]]

![[attachment/Pasted image 20260823135355.png]]

![[attachment/Pasted image 20260823135441.png]]

![[attachment/Pasted image 20260823135619.png]]

![[attachment/Pasted image 20260823135742.png]]

![[attachment/Pasted image 20260823135810.png]]

![[attachment/Pasted image 20260823135908.png]]

![[attachment/Pasted image 20260823140029.png]]

![[attachment/Pasted image 20260823140057.png]]

![[attachment/Pasted image 20260823140334.png]]

![[attachment/Pasted image 20260823140748.png]]

![[attachment/Pasted image 20260823165426.png]]

![[attachment/Pasted image 20260823165454.png]]

![[attachment/Pasted image 20260823165619.png]]

![[attachment/Pasted image 20260823165808.png]]

![[attachment/Pasted image 20260823165938.png]]

![[attachment/Pasted image 20260823170151.png]]

![[attachment/Pasted image 20260823170350.png]]

![[attachment/Pasted image 20260823170526.png]]

![[attachment/Pasted image 20260823170702.png]]

![[attachment/Pasted image 20260823170731.png]]

![[attachment/Pasted image 20260823170825.png]]


![[attachment/Pasted image 20260823171248.png]]

![[attachment/Pasted image 20260823171150.png]]

![[attachment/Pasted image 20260823171420.png]]

![[attachment/Pasted image 20260823171439.png]]

![[attachment/Pasted image 20260823171616.png]]

![[attachment/Pasted image 20260823171807.png]]

![[attachment/Pasted image 20260823172131.png]]

## RNN详解


循环神经网络（Recurrent Neural Network, RNN）是专门用来处理**序列数据**（如文本、时间序列、语音、代码）的深度学习模型。

传统的全连接神经网络（MLP）或卷积神经网络（CNN）假设输入之间是相互独立的，而 RNN 的核心突破在于：**引入了“记忆”机制，让网络在处理当前时刻的输入时，能够参考上一时刻的隐状态信息。**

### 一、 核心动机：为什么需要 RNN？

假设要处理两个自然语言处理任务：

1. 句子 A：“我不喜欢吃**苹果**”
    
2. 句子 B：“**苹果**发布了新款 iPhone”
    

对于“苹果”这个词，标准的前馈神经网络（Feedforward Network）只能根据“苹果”本身的特征（词向量）进行分类，无法区分它是水果还是公司。而人类理解语言时，会依赖**上下文记忆**。RNN 正是为了模拟这种**依据时间/顺序延伸的记忆力**而设计的。

### 二、 RNN 的结构与数学原理

RNN 的结构可以通过按时间展开（Unrolling over time）来直观理解。

#### 1. 结构展开图

假设我们有一个序列 X=(x1​,x2​,…,xT​)，RNN 在每个时间步 t 进行如下计算：



```Plaintext
       y_1          y_2          y_3          y_t
        ^            ^            ^            ^
        |            |            |            |
     [  V  ]      [  V  ]      [  V  ]      [  V  ]
  -->[ h_1 ]----->[ h_2 ]----->[ h_3 ]----->[ h_t ]----> (隐状态传递)
       ^ W_hh       ^ W_hh       ^ W_hh       ^ W_hh
       |            |            |            |
      x_1          x_2          x_3          x_t
```

- $x_t$​：t 时刻的输入向量（例如第 t 个词的 Word Embedding）。
    
- $h_t$​：t 时刻的隐藏状态（Hidden State），代表网络到 t 时刻为止保留的**记忆**。
    
- $y_t$：t 时刻的输出向量（这个依据任务的不同而存在/不存在，例如预测下一个词的概率分布）。
    

#### 2. 核心数学公式

RNN 在每一个时间步 t 共享同一套参数矩阵 ($W_{xh}$​,$W_{hh}$,$W_{hy}$​,$b_h$​,$b_y$​)。

**步骤 1：更新隐藏状态 $h_t$​** 隐藏状态由“当前输入 $x_t$​”和“上一时刻的记忆 $h_{t-1}$”共同决定：

$$h_t = \tanh\big(W_{xh}x_t + W_{hh}h_{t-1} + b_h\big)$$

- $W_{xh}$​：输入到隐藏层的权重矩阵。
    
- $W_{hh}$​：隐藏层到隐藏层的权重矩阵（**核心：跨时间的循环连接**）。
    
- $b_h$​：隐藏层的偏置项。
    
- $tanh$：双曲正切激活函数，将状态值压缩到 (−1,1) 之间，防止数值爆炸。
    

**步骤 2：计算当前输出 $\hat{y_t}$​**
依据当前的隐藏状态 $h_t$​ 生成输出：

$$\hat{y}_t = \mathrm{Softmax}\big(W_{hy}h_t + b_y\big)$$

- $W_{hy}$​：隐藏层到输出层的权重矩阵。
    
- $Softmax$：用于多分类任务（如语言模型预测词汇概率）。若为回归任务，可直接使用线性激活。
    

### 三、 权重共享机制（Weight Sharing）

RNN 的关键特性是**参数共享**： 无论输入序列长度是 5 还是 5000，网络使用的权重矩阵 $W_{xh}$​,$W_{hh}$​,$W_{hy}$​ 是完全相同的。

- **优点**：大大减少了参数量，且模型可以处理任意长度的输入序列。
    
- **物理意义**：代表模型学会了一种“时间上不变”的转化规则。
    

### 四、 RNN 的训练算法：通过时间反向传播（BPTT）

RNN 的训练基于梯度下降，采用**通过时间反向传播（Backpropagation Through Time, BPTT）**。

1. **前向传播**：从 t=1 顺序计算至 t=T，得到各时刻的 $h_t$ 与 $\hat{y_t}$，并计算总损失：
    $$L = \sum_{t=1}^{T} L_t(y_t,\hat{y}_t)$$
2. **反向传播**：梯度不仅需要从输出层向隐藏层传递，还要**沿时间轴向后传递**（从 t=T 倒推回 t=1）。
    

例如，损失 L 对隐藏层转置矩阵 $W_{hh}$ 的偏导数需要累计所有时间步的贡献：

$$\frac{\partial L}{\partial W_{hh}} =\sum_{t=1}^{T}\sum_{k=1}^{t} \frac{\partial L_t}{\partial \hat{y}_t} \frac{\partial \hat{y}_t}{\partial h_t} \left( \prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}} \right) \frac{\partial h_k}{\partial W_{hh}}$$​​

### 五、 传统 RNN 的致命缺陷：梯度消失与梯度爆炸

在上述 BPTT 公式中，包含连乘项 $\prod_{j=k+1}^{t} \frac{\partial h_j}{\partial h_{j-1}}$。

- $\frac{\partial h_j}{\partial h_{j-1}} = \mathrm{diag}\big(1-\tanh^2(\dots)\big)\cdot W_{hh}^\top$
    
- 当时间步长 T 很大（如序列很长）时：
    
    - 若 Whh​ 的特征值小于 1 或 tanh′ 的导数小于 1（≤1），随着连乘次数增加，梯度会呈指数级衰减至 0 → **梯度消失（Gradient Vanishing）**。
        
    - 若 Whh​ 的特征值很大，梯度会呈指数级增长 → **梯度爆炸（Gradient Exploding）**。
        

**后果**：传统 RNN 缺乏长距离依赖（Long-term Dependency）捕捉能力。模型容易“近视”，只能记住最近几个时间步的信息，而忘记很久以前的信息。

### 六、 常见变体与进阶结构

为了解决梯度消失/爆炸问题，后续提出了更优秀的架构：

1. **LSTM（长短期记忆网络, Long Short-Term Memory）**
    
    - 引入了细胞状态（Cell State）和三个门控机制：**遗忘门（Forget Gate）**、**输入门（Input Gate）**、**输出门（Output Gate）**。
        
    - 细胞状态以加法形式更新（梯度通过加法路径传递，不易衰减），彻底解决了长距离依赖问题。
        
2. **GRU（门控循环单元, Gated Recurrent Unit）**
    
    - LSTM 的简化版，合并了细胞状态与隐藏状态，仅保留**重置门（Reset Gate）**与**更新门（Update Gate）**，计算效率更高。
        
3. **BiRNN（双向循环神经网络, Bidirectional RNN）**
    
    - 同时包含正向和反向两个 RNN，使 t 时刻的输出同时参考上下文（Past & Future）信息，常用于文本分类或序列标注。
        

### 七、 PyTorch 代码极简实现

Python

```
import torch
import torch.nn as nn

# 定义一个简单的 RNN 模型
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        # input_size: 输入特征维度 (如 word embedding 维度)
        # hidden_size: 隐藏层节点数 (记忆维度)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        # out shape: (batch_size, sequence_length, hidden_size)
        # h_n shape: (1, batch_size, hidden_size) —— 最后一个时间步的隐状态
        out, h_n = self.rnn(x)
        
        # 取最后一个时间步的隐藏状态做分类/回归
        logits = self.fc(out[:, -1, :])
        return logits

# 实例化测试
model = SimpleRNN(input_size=10, hidden_size=20, output_size=2)
dummy_input = torch.randn(32, 5, 10) # Batch大小 32，序列长度 5，特征维度 10
output = model(dummy_input)
print("Output shape:", output.shape) # 应为 (32, 2)
```

你能掌握 RNN 的核心逻辑吗？如果有关于 BPTT 的具体数学推导或 LSTM 门控机制想深入探讨的地方，可以告诉我！

## 隐藏层参数更新详解

是我粗糙了，抱歉！咱们把隐藏层梯度的推导拆得更细，一步步把矩阵维度、链式法则的展开过程彻底理顺。

针对 RNN 隐藏层参数 $W_{hh}, W_{xh}, b_h$ 的梯度推导，核心难点在于：**$h_t$ 不仅影响当前的输出 $y_t$，还会作为下一时刻的输入影响 $h_{t+1}, h_{t+2}, \dots, h_T$。**

我们采用工程和学术界通用的 **时间倒序递推（BPTT）** 来展开。

### 一、 符号定义与链式法则总览

再复习一下 $t$ 时刻的前向传播公式：

$$z_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h \quad \in \mathbb{R}^{d_h \times 1}$$

$$h_t = \tanh(z_t) \quad \in \mathbb{R}^{d_h \times 1}$$

$$o_t = W_{hy} h_t + b_y \quad \in \mathbb{R}^{d_y \times 1}$$

其中：

- $d_h$ 为隐藏层维度，$d_y$ 为输出维度。
    
- $L$ 为序列的总损失 $L = \sum_{t=1}^T L_t$。
    
- 记输出层未激活梯度为 $\delta_{o,t} = \frac{\partial L_t}{\partial o_t} = \hat{y}_t - y_t \in \mathbb{R}^{d_y \times 1}$。
    

我们的目标是求 $L$ 对隐藏层未激活输入 $z_t$ 的偏导数，记为：

$$\delta_{h,t} = \frac{\partial L}{\partial z_t} \in \mathbb{R}^{d_h \times 1}$$

### 二、 核心关键：$\delta_{h,t}$ 的倒序递推推导

根据多元函数链式法则，损失 $L$ 对 $z_t$ 的影响有**两条路径**：

1. **路径一（当前路径）**：$z_t \rightarrow h_t \rightarrow o_t \rightarrow L_t$ （通过当前时刻的输出影响当前损失）
    
2. **路径二（未来路径）**：$z_t \rightarrow h_t \rightarrow z_{t+1} \rightarrow \dots \rightarrow L_{>t}$ （通过下一时刻的隐状态影响后续所有损失）
    

因此，$\frac{\partial L}{\partial h_t}$ 等于这两条路径梯度的叠加：

$$\frac{\partial L}{\partial h_t} = \underbrace{\frac{\partial L_t}{\partial h_t}}_{\text{来自当前输出}} + \underbrace{\frac{\partial L_{>t}}{\partial h_t}}_{\text{来自未来时刻}}$$

#### 1. 计算第一项（来自当前输出）

$$\frac{\partial L_t}{\partial h_t} = \left( \frac{\partial o_t}{\partial h_t} \right)^T \frac{\partial L_t}{\partial o_t} = W_{hy}^T \delta_{o,t}$$

#### 2. 计算第二项（来自未来时刻）

$h_t$ 通过影响 $z_{t+1} = W_{xh} x_{t+1} + W_{hh} h_t + b_h$ 来传递给未来：

$$\frac{\partial L_{>t}}{\partial h_t} = \left( \frac{\partial z_{t+1}}{\partial h_t} \right)^T \frac{\partial L}{\partial z_{t+1}} = W_{hh}^T \delta_{h,t+1}$$

#### 3. 汇总合并得到 $\frac{\partial L}{\partial h_t}$

将上述两项相加：

$$\frac{\partial L}{\partial h_t} = W_{hy}^T \delta_{o,t} + W_{hh}^T \delta_{h,t+1}$$

#### 4. 通过激活函数求 $\delta_{h,t} = \frac{\partial L}{\partial z_t}$

因为 $h_t = \tanh(z_t)$，由链式法则：

$$\delta_{h,t} = \frac{\partial L}{\partial z_t} = \frac{\partial L}{\partial h_t} \odot \frac{\partial \tanh(z_t)}{\partial z_t}$$

因为 $\tanh'(z_t) = 1 - \tanh^2(z_t) = 1 - h_t^2$，代入上式得到**BPTT 核心递推公式**：

$$\delta_{h,t} = \left( W_{hy}^T \delta_{o,t} + W_{hh}^T \delta_{h,t+1} \right) \odot (1 - h_t^2)$$

> **边界条件（最后一个时间步 $T$）**：
> 
> 因为 $t=T$ 时没有未来时刻，所以不存在 $t+1$ 项：
> 
> $$\delta_{h,T} = \left( W_{hy}^T \delta_{o,T} \right) \odot (1 - h_T^2)$$

我们从 $t = T$ 开始向前倒推计算（$T \to T-1 \to \dots \to 1$），即可依次拿到所有时刻的 $\delta_{h,t}$。

### 三、 共享权重矩阵梯度的最终求解

拿到每个时刻的隐藏层梯度 $\delta_{h,t}$ 之后，求共享权重 $W_{hh}, W_{xh}, b_h$ 的梯度就非常直观了。**由于这些权重在所有时间步共享，总梯度即为所有时间步梯度的累加和**。

#### 1. 递归权重矩阵 $W_{hh}$ 的梯度

对于任意时间步 $t$，$z_t = W_{xh}x_t + W_{hh}h_{t-1} + b_h$，所以 $\frac{\partial z_t}{\partial W_{hh}} = h_{t-1}^T$。

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial z_t} \frac{\partial z_t}{\partial W_{hh}} = \sum_{t=1}^{T} \delta_{h,t} h_{t-1}^T$$

- **维度对齐检查**：$\delta_{h,t} \in \mathbb{R}^{d_h \times 1}$，$h_{t-1}^T \in \mathbb{R}^{1 \times d_h}$，外积结果为 $\mathbb{R}^{d_h \times d_h}$，与 $W_{hh}$ 的形状完全一致！
    

#### 2. 输入权重矩阵 $W_{xh}$ 的梯度

同理，$\frac{\partial z_t}{\partial W_{xh}} = x_t^T$：

$$\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^{T} \delta_{h,t} x_t^T$$

- **维度对齐检查**：$\delta_{h,t} \in \mathbb{R}^{d_h \times 1}$，$x_t^T \in \mathbb{R}^{1 \times d_x}$，结果为 $\mathbb{R}^{d_h \times d_x}$，与 $W_{xh}$ 的形状一致。
    

#### 3. 隐藏层偏置 $b_h$ 的梯度

由于 $\frac{\partial z_t}{\partial b_h} = I$（单位矩阵）：

$$\frac{\partial L}{\partial b_h} = \sum_{t=1}^{T} \delta_{h,t}$$

### 四、 参数更新示例（以 SGD 为例）

计算出梯度总和后，若计算的是平均损失（除以序列长度 $T$），则加上缩放因子 $\frac{1}{T}$，根据梯度下降公式更新参数（学习率为 $\eta$）：

$$W_{hh} \leftarrow W_{hh} - \eta \cdot \frac{1}{T} \sum_{t=1}^{T} \delta_{h,t} h_{t-1}^T$$

$$W_{xh} \leftarrow W_{xh} - \eta \cdot \frac{1}{T} \sum_{t=1}^{T} \delta_{h,t} x_t^T$$

$$b_h \leftarrow b_h - \eta \cdot \frac{1}{T} \sum_{t=1}^{T} \delta_{h,t}$$





---

## 笔记联系

- [[20_深度学习/自然语言处理/序列建模 - 机器学习与深度学习如何理解语句|序列建模 - 机器学习与深度学习如何理解语句]]：结合序列模型理解语句建模。
- [[20_深度学习/Transformer与ViT/Transformer - 入门详解|Transformer - 入门详解]]：比较循环结构与注意力结构的序列建模。
- [[50_汇报与写作/索引 - 汇报与写作|汇报与写作索引]]：查看本主题的 PPT 文件与保留版本。
