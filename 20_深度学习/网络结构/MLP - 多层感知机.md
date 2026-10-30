---
title: "MLP - 多层感知机"
aliases:
  - "MLP多层感知机"
分类: "网络结构"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/网络结构"
状态: "已整理"
摘要: "围绕“MLP - 多层感知机”，整理MLP 的网络结构、前向传播（Forward Propagation）的数学细节、符号定义：。"
更新日期: "2026-10-30"
上级目录: "[[20_深度学习/网络结构/总结 - 网络结构]]"
相关笔记:
  - "[[20_深度学习/训练基础/深度学习 - 神经网络入门]]"
  - "[[40_课程笔记/吴恩达机器学习/神经网络 - 课程笔记]]"
  - "[[20_深度学习/训练基础/神经网络 - 权重初始化]]"
  - "[[20_深度学习/网络结构/CNN - 卷积神经网络]]"
论文条目: []
论文标题: []
发表载体: []
发表年份: []
发表类型: []
发表链接: []
预印本链接: []
---

[[首页|首页]] · [[20_深度学习/网络结构/总结 - 网络结构|网络结构总结]]

# MLP - 多层感知机

多层感知机（Multi-Layer Perceptron, MLP）是最基础、最经典的前馈神经网络（Feedforward Neural Network, FNN）。

如果在学习深度学习时遇到过全连接神经网络（Fully Connected Network, Dense Network），它们其实指的是同一个东西。MLP 是计算机视觉（CNN）、序列建模（RNN）以及 Transformer 等所有复杂现代架构的基础基石。

为了彻底教会你，我们将按照 **“结构认知 $\rightarrow$ 前向传播数学推导 $\rightarrow$ 激活函数的意义 $\rightarrow$ 损失函数 $\rightarrow$ 反向传播（BP）极详细推导 $\rightarrow$ 参数更新”** 的顺序全景拆解。

### 一、 MLP 的网络结构

MLP 由多个层（Layer）堆叠而成，每一层包含若干个神经元（Neuron）。典型结构分为三部分：

1. **输入层（Input Layer）**：接收原始特征向量 $x$，节点数等于特征维度。输入层不做任何数学计算，只负责传参。
    
2. **隐藏层（Hidden Layer）**：可有一层或多层，进行特征提取与非线性变换。
    
3. **输出层（Output Layer）**：输出最终的预测结果 $\hat{y}$。
    

Plaintext

```
 (输入层 Input)        (隐藏层 Hidden)        (输出层 Output)
     x_1  --------------> ( h_1 ) --------------\
           \           /         \              \
            \         /           \              --> ( y_1 )
             \       /             \            /
     x_2  -----> ( h_2 ) --------------> ( h_2 ) 
             /       \             /            \
            /         \           /              --> ( y_2 )
           /           \         /              /
     x_3  --------------> ( h_3 ) --------------/
```

**全连接（Fully Connected）的含义**：

前一层的**每一个神经元**都与下一层的**每一个神经元**相连。每个连接上都有一个可学习的权重参数 $w$。

### 二、 前向传播（Forward Propagation）的数学细节

为了把公式推导讲透，我们构建一个具体的 **3 层 MLP 模型**（包含：1个输入层、1个隐藏层、1个输出层）。

#### 符号定义：

- **输入特征向量**：$x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_{d_{in}} \end{bmatrix} \in \mathbb{R}^{d_{in} \times 1}$
    
- **隐藏层权重矩阵**：$W^{(1)} \in \mathbb{R}^{d_{h} \times d_{in}}$ （其中 $d_h$ 为隐藏层神经元个数）
    
- **隐藏层偏置向量**：$b^{(1)} \in \mathbb{R}^{d_{h} \times 1}$
    
- **隐藏层激活函数**：$\sigma(\cdot)$
    
- **输出层权重矩阵**：$W^{(2)} \in \mathbb{R}^{d_{out} \times d_{h}}$ （其中 $d_{out}$ 为输出维度）
    
- **输出层偏置向量**：$b^{(2)} \in \mathbb{R}^{d_{out} \times 1}$
    

#### 计算步骤（分步展开）

##### 步骤 1：隐藏层线性变换（未激活值 $z^{(1)}$）

对于隐藏层的第 $i$ 个神经元，它把输入的全部特征加权求和，再加上偏置：

$$z_i^{(1)} = \sum_{j=1}^{d_{in}} W_{ij}^{(1)} x_j + b_i^{(1)}$$

写成完整的矩阵形式：

$$z^{(1)} = W^{(1)} x + b^{(1)} \quad \in \mathbb{R}^{d_h \times 1}$$

##### 步骤 2：隐藏层非线性激活（激活值 $h$）

把未激活输入 $z^{(1)}$ 送入激活函数：

$$h = \sigma(z^{(1)}) \quad \in \mathbb{R}^{d_h \times 1}$$

##### 步骤 3：输出层线性变换（未激活值 $z^{(2)}$）

将隐藏层输出 $h$ 作为输出层的输入：

$$z^{(2)} = W^{(2)} h + b^{(2)} \quad \in \mathbb{R}^{d_{out} \times 1}$$

##### 步骤 4：输出层激活（得到预测值 $\hat{y}$）

根据任务类型决定输出层的激活方式：

$$\hat{y} = g(z^{(2)})$$

- **回归任务**：$g(z) = z$ （直接输出线性结果）
    
- **二分类任务**：$g(z) = \text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$
    
- **多分类任务**：$g(z) = \text{Softmax}(z)$，其中 $\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$
    

### 三、 为什么 MLP 必须需要“激活函数”？

假设我们**不使用**激活函数（即 $\sigma(z) = z$），看看网络会变成什么样：

$$h = z^{(1)} = W^{(1)} x + b^{(1)}$$

$$\hat{y} = W^{(2)} h + b^{(2)} = W^{(2)} (W^{(1)} x + b^{(1)}) + b^{(2)}$$

$$\hat{y} = \underbrace{(W^{(2)} W^{(1)})}_{W_{new}} x + \underbrace{(W^{(2)} b^{(1)} + b^{(2)})}_{b_{new}}$$

**结论**：没有非线性激活函数，**无论堆叠多少层隐藏层，整个网络在数学上等价于单层线性回归模型 $W_{new}x + b_{new}$**。

激活函数（如 ReLU, Sigmoid, GELU）为网络引入了**非线性表达能力**，使其能够拟合极其复杂的非线性决策边界（由通用近似定理 Universal Approximation Theorem 保证）。

### 四、 损失函数（Loss Function）

模型算出了预测值 $\hat{y}$，我们需要评估它与真实标签 $y$ 的差距。

1. **均方误差损失（MSE Loss，用于回归任务）**：
    
    $$L = \frac{1}{2} \Vert{}\hat{y} - y\Vert{}^2 = \frac{1}{2} \sum_{k=1}^{d_{out}} (\hat{y}_k - y_k)^2$$
    
2. **交叉熵损失（Cross-Entropy Loss，用于分类任务）**：
    
    $$L = -\sum_{k=1}^{d_{out}} y_k \log \hat{y}_k$$
    

### 五、 极详细推导：反向传播算法（Backpropagation, BP）

反向传播是使用**链式法则（Chain Rule）**，从输出层向后求出损失 $L$ 对每一个参数矩阵 $W^{(2)}, b^{(2)}, W^{(1)}, b^{(1)}$ 的偏导数（梯度）。

这里我们以 **MSE 损失 + 任意激活函数 $\sigma$** 为例，逐步拆解梯度的矩阵求导。

#### 第一阶段：输出层参数梯度求解 ($W^{(2)}, b^{(2)}$)

我们先定义**输出层的未激活误差项 $\delta^{(2)}$**：

$$\delta^{(2)} = \frac{\partial L}{\partial z^{(2)}} \in \mathbb{R}^{d_{out} \times 1}$$

##### 1. 求 $\delta^{(2)}$

根据链式法则：

$$\delta^{(2)} = \frac{\partial L}{\partial \hat{y}} \odot \frac{\partial \hat{y}}{\partial z^{(2)}}$$

若输出层无激活函数（回归）：$\hat{y} = z^{(2)}$，则 $\frac{\partial \hat{y}}{\partial z^{(2)}} = 1$，

$$\delta^{(2)} = \hat{y} - y$$

##### 2. 求 $W^{(2)}$ 的梯度

因为 $z^{(2)} = W^{(2)} h + b^{(2)}$，利用矩阵求导法则（$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} \cdot h^T$）：

$$\frac{\partial L}{\partial W^{(2)}} = \delta^{(2)} h^T \quad \in \mathbb{R}^{d_{out} \times d_h}$$

- **维度对齐检查**：$\delta^{(2)} \in \mathbb{R}^{d_{out} \times 1}$，$h^T \in \mathbb{R}^{1 \times d_h}$，外积结果为 $d_{out} \times d_h$，与 $W^{(2)}$ 维度完全吻合！
    

##### 3. 求 $b^{(2)}$ 的梯度

$$\frac{\partial L}{\partial b^{(2)}} = \delta^{(2)} \quad \in \mathbb{R}^{d_{out} \times 1}$$

#### 第二阶段：隐藏层参数梯度求解 ($W^{(1)}, b^{(1)}$)

这是 BP 算法的最精彩之处：**把输出层的误差 $\delta^{(2)}$ 沿网络反向传播给隐藏层**。

我们定义**隐藏层的未激活误差项 $\delta^{(1)}$**：

$$\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}} \in \mathbb{R}^{d_h \times 1}$$

##### 1. 误差反向传递求 $\delta^{(1)}$

隐藏层变量 $z^{(1)}$ 影响 $h$，进而影响 $z^{(2)}$，最终影响 $L$。根据链式法则：

$$\frac{\partial L}{\partial h} = \left( \frac{\partial z^{(2)}}{\partial h} \right)^T \frac{\partial L}{\partial z^{(2)}} = (W^{(2)})^T \delta^{(2)}$$

再结合隐藏层的激活函数 $h = \sigma(z^{(1)})$：

$$\delta^{(1)} = \frac{\partial L}{\partial z^{(1)}} = \frac{\partial L}{\partial h} \odot \sigma'(z^{(1)})$$

合并得到 **隐藏层误差项公式**：

$$\delta^{(1)} = \left( (W^{(2)})^T \delta^{(2)} \right) \odot \sigma'(z^{(1)})$$

> **注**：$\odot$ 为 Hadamard 积（按元素相乘），$\sigma'(z^{(1)})$ 是激活函数对 $z^{(1)}$ 的导数。

##### 2. 求 $W^{(1)}$ 的梯度

因为 $z^{(1)} = W^{(1)} x + b^{(1)}$：

$$\frac{\partial L}{\partial W^{(1)}} = \delta^{(1)} x^T \quad \in \mathbb{R}^{d_h \times d_{in}}$$

- **维度对齐检查**：$\delta^{(1)} \in \mathbb{R}^{d_h \times 1}$，$x^T \in \mathbb{R}^{1 \times d_{in}}$，结果为 $d_h \times d_{in}$，与 $W^{(1)}$ 维度完全一致！
    

##### 3. 求 $b^{(1)}$ 的梯度

$$\frac{\partial L}{\partial b^{(1)}} = \delta^{(1)} \quad \in \mathbb{R}^{d_h \times 1}$$

### 六、 参数更新（Parameter Update）

通过反向传播求得所有梯度后，利用梯度下降法（SGD）按相反的方向更新权重，学习率为 $\eta$：

1. **更新输出层权重与偏置**：
    
    $$W^{(2)} \leftarrow W^{(2)} - \eta \cdot \frac{\partial L}{\partial W^{(2)}}$$
    
    $$b^{(2)} \leftarrow b^{(2)} - \eta \cdot \frac{\partial L}{\partial b^{(2)}}$$
    
2. **更新隐藏层权重与偏置**：
    
    $$W^{(1)} \leftarrow W^{(1)} - \eta \cdot \frac{\partial L}{\partial W^{(1)}}$$
    
    $$b^{(1)} \leftarrow b^{(1)} - \eta \cdot \frac{\partial L}{\partial b^{(1)}}$$
    

### 七、 从零实现：PyTorch 手写与验证

#### 1. 使用标准 PyTorch 高级 API 实现 MLP

Python

```
import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultiLayerPerceptron, self).__init__()
        # 第一层全连接 (隐藏层)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 激活函数
        self.relu = nn.ReLU()
        # 第二层全连接 (输出层)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 1. z1 = W1*x + b1
        # 2. h = ReLU(z1)
        h = self.relu(self.fc1(x))
        # 3. z2 = W2*h + b2
        out = self.fc2(h)
        return out

# 测试模型
model = MultiLayerPerceptron(input_dim=4, hidden_dim=8, output_dim=2)
x_dummy = torch.randn(5, 4) # Batch Size = 5, 特征数 = 4
output = model(x_dummy)
print("输出形状:", output.shape) # (5, 2)
```

### 八、 MLP 的优缺点总结

#### 优点：

1. **结构通用性强**：原理简单直观，可以拟合任意复杂的非线性函数。
    
2. **易于扩展**：可以自由调整隐藏层数量和神经元节点数。
    

#### 局限性：

1. **参数量极大**：全连接导致权重数量随着特征维度暴增（参数爆炸），容易引发过拟合。
    
2. **丢失空间/时序结构**：必须把输入拉平（Flatten）为一维向量，无法直接保持图像的二维空间关系（因此图像领域演化出了 **CNN**）或文本的时间前后关系（因此序列领域演化出了 **RNN / Transformer**）。

<!-- knowledge-links:start -->

## 笔记联系

- [[20_深度学习/训练基础/深度学习 - 神经网络入门|深度学习 - 神经网络入门]]：回顾神经网络的基础结构。
- [[40_课程笔记/吴恩达机器学习/神经网络 - 课程笔记|神经网络 - 课程笔记]]：对照课程记录与结构、公式详解。
- [[20_深度学习/训练基础/神经网络 - 权重初始化|神经网络 - 权重初始化]]：延伸到训练前的权重初始化。
- [[20_深度学习/网络结构/CNN - 卷积神经网络|CNN - 卷积神经网络]]：比较卷积连接与全连接结构。

<!-- knowledge-links:end -->
