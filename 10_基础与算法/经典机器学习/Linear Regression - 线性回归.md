---
title: "Linear Regression - 线性回归"
aliases:
  - "Linear Regression：线性回归"
分类: "经典机器学习"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/经典机器学习"
状态: "已整理"
摘要: "围绕“Linear Regression - 线性回归”，整理建模：从一维直线到高维超平面、元线性回归（Single Variable）、多元线性回归（Multiple Variables）。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/经典机器学习/总结 - 经典机器学习]]"
相关笔记:
  - "[[10_基础与算法/经典机器学习/Logistic Regression - 逻辑回归]]"
  - "[[40_课程笔记/吴恩达机器学习/线性回归 - 课程笔记]]"
论文条目: []
论文标题: []
发表载体: []
发表年份: []
发表类型: []
发表链接: []
预印本链接: []
---

[[首页|首页]] · [[10_基础与算法/经典机器学习/总结 - 经典机器学习|经典机器学习总结]]

# Linear Regression - 线性回归

**线性回归（Linear Regression）** 是统计学与机器学习中最基础、最经典，但也最具启发性的**监督学习算法**。无论是预测房屋价格、评估股票走势，还是作为深度神经网络（Deep Neural Networks）中单个神经元的数学基石，线性回归都扮演着至关重要的人物。

线性回归的核心使命，就是**用一条“直线”（或高维空间中的“超平面”），去拟合自变量（特征）与因变量（标签）之间的线性映射关系**。

下面我们将从 **物理直观与数学模型 $\rightarrow$ 损失函数推导 $\rightarrow$ 参数求解（解析解与迭代解） $\rightarrow$ 概率解释（最大似然估计） $\rightarrow$ 经典正则化（Ridge与Lasso） $\rightarrow$ PyTorch 代码实现** 进行全景拆解。

### 一、 建模：从一维直线到高维超平面

#### 1. 一元线性回归（Single Variable）

假设我们只有一个特征（例如：房屋面积 $x$）来预测目标值（例如：房价 $y$）。我们希望找到一条直线：

$$f(x) = w x + b$$

- $w$（Weight, 权重/斜率）：决定了特征对预测结果的影响程度和方向。
    
- $b$（Bias, 偏置/截距）：代表当特征 $x=0$ 时的基础预测值。
    

#### 2. 多元线性回归（Multiple Variables）

现实问题中往往包含多个特征 $x = [x_1, x_2, \dots, x_d]^T$（例如：面积、房间数、地段评分等）。此时模型扩展为：

$$f(x) = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$

为了在数学计算上更简洁，我们通常采用向量化（Vectorization）表达：

- 定义权重向量 $\boldsymbol{w} = [w_1, w_2, \dots, w_d]^T$。
    
- 预测公式写为点积形式：
    

$$f(\boldsymbol{x}) = \boldsymbol{w}^T \boldsymbol{x} + b$$

#### 3. 吸收偏置（Bias Absorption）

为了方便矩阵运算，我们将偏置 $b$ 融合进权重向量中：

- 在特征向量末尾补充一个常数项 $1$：$\tilde{\boldsymbol{x}} = [x_1, x_2, \dots, x_d, 1]^T$。
    
- 将偏置合并入权重向量：$\tilde{\boldsymbol{w}} = [w_1, w_2, \dots, w_d, b]^T$。
    

此时模型可以极为优雅地表达为：

$$f(\boldsymbol{x}) = \tilde{\boldsymbol{w}}^T \tilde{\boldsymbol{x}}$$

### 二、 损失函数：如何衡量“拟合的好坏”？

拥有了模型形态后，我们需要定义一个指标来评估当前参数下的预测值 $\hat{y}_i = f(\boldsymbol{x}_i)$ 与真实值 $y_i$ 之间的误差。

线性回归最常用的损失函数是 **均方误差（Mean Squared Error, MSE）**，又称 **欧氏距离/残差平方和（Residual Sum of Squares, RSS）**：

$$J(\boldsymbol{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f(\boldsymbol{x}_i) - y_i \right)^2$$

- $m$ 为训练样本数量。
    
- **前面的系数 $\frac{1}{2}$** 是为了后续在对平方项求导数时能够与指数 $2$ 约简，不影响优化结果。
    

#### 几何意义：最小二乘法（Ordinary Least Squares, OLS）

均方误差在几何上对应的就是**预测点到真实点在纵轴（标签轴）方向的垂直距离平方和**。最小化 MSE 的过程，在几何上就是寻找到一条最佳拟合直线，使得所有样本点到该直线的纵向偏差平方和最小。

### 三、 求解策略：如何找到最优参数？

寻找使损失函数 $J(\boldsymbol{w})$ 最小的参数 $\boldsymbol{w}$，有两种主流路线：**闭形式解析解（正规方程）** 与 **数值迭代解（梯度下降）**。

把全量数据集写成矩阵形式：

- 数据矩阵 $\boldsymbol{X} \in \mathbb{R}^{m \times (d+1)}$（包含全 1 列）。
    
- 标签向量 $\boldsymbol{y} \in \mathbb{R}^{m \times 1}$。
    
- 矩阵形式损失函数：$J(\boldsymbol{w}) = \frac{1}{2m} \Vert{}\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y}\Vert{}_2^2 = \frac{1}{2m} (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})^T (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})$
    

#### 1. 闭形式解析解：正规方程（Normal Equation）

利用矩阵求导法则，对 $\boldsymbol{w}$ 求偏导并令其等于 $0$：

$$\nabla_{\boldsymbol{w}} J(\boldsymbol{w}) = \frac{1}{m} \boldsymbol{X}^T (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y}) = 0$$

$$\boldsymbol{X}^T \boldsymbol{X} \boldsymbol{w} = \boldsymbol{X}^T \boldsymbol{y}$$

当 $\boldsymbol{X}^T \boldsymbol{X}$ 可逆时，直接得到最优解：

$$\boldsymbol{w}^* = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$$

- **优缺点**：
    
    - **优点**：无需迭代、无需手动设置学习率，一步到位算出精确最优解。
        
    - **缺点**：求解矩阵逆 $(\boldsymbol{X}^T \boldsymbol{X})^{-1}$ 的时间复杂度为 $O(d^3)$（$d$ 为特征数）。当特征数量极多（如 $d > 10^5$）或存在严重的多重共线性导致不可逆时，求解非常吃力甚至失效。
        

#### 2. 数值迭代解：梯度下降法（Gradient Descent）

当数据量巨大或不可求逆时，采用梯度下降逐步更新参数：

$$\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \nabla_{\boldsymbol{w}} J(\boldsymbol{w})$$

展开得到迭代公式：

$$\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \frac{1}{m} \boldsymbol{X}^T (\boldsymbol{X}\boldsymbol{w} - \boldsymbol{y})$$

其中 $\alpha$ 为学习率（Learning Rate）。

### 四、 概率解释：为什么偏偏选择平方损失？

很多人会问：**为什么损失函数一定要用残差的平方和（MSE），而不是绝对值和或者四次方和？**

答案藏在高斯分布（正态分布）与最大似然估计（Maximum Likelihood Estimation, MLE）的结合中。

假设真实标签 $y_i$ 与模型预测值之间存在不可避免的随机噪声 $\epsilon_i$：

$$y_i = \boldsymbol{w}^T \boldsymbol{x}_i + \epsilon_i$$

根据中心极限定理，假设噪声 $\epsilon_i$ 独立同分布于均值为 $0$、方差为 $\sigma^2$ 的高斯分布：$\epsilon_i \sim \mathcal{N}(0, \sigma^2)$。

这意味着，在给定 $\boldsymbol{x}_i$ 和参数 $\boldsymbol{w}$ 条件下，$y_i$ 的条件概率分布为：

$$p(y_i \mid \boldsymbol{x}_i; \boldsymbol{w}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y_i - \boldsymbol{w}^T \boldsymbol{x}_i)^2}{2\sigma^2} \right)$$

构建整个数据集的**似然函数（Likelihood Function）**：

$$L(\boldsymbol{w}) = \prod_{i=1}^{m} p(y_i \mid \boldsymbol{x}_i; \boldsymbol{w})$$

对似然函数取对数，得到**对数似然函数（Log-Likelihood）**：

$$\ln L(\boldsymbol{w}) = \sum_{i=1}^{m} \ln \left( \frac{1}{\sqrt{2\pi}\sigma} \right) - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y_i - \boldsymbol{w}^T \boldsymbol{x}_i)^2$$

最大化对数似然 $\max_{\boldsymbol{w}} \ln L(\boldsymbol{w})$，等价于忽略常数项后最小化后半部分的二次项：

$$\min_{\boldsymbol{w}} \sum_{i=1}^{m} (y_i - \boldsymbol{w}^T \boldsymbol{x}_i)^2$$

**结论**：**最小二乘法（MSE损失）本质上就是在噪声服从正态分布假设下的最大似然估计！**

### 五、 防止过拟合：正则化与线性回归演进

当数据特征维度很多，或者特征之间高度相关（多重共线性）时，普通的线性回归极易发生过拟合（参数 $\boldsymbol{w}$ 的数值变得极大）。此时需要加入**正则化项（Penalty Term）**。

|**算法名称**|**损失函数定义**|**几何特征与作用**|
|---|---|---|
|**岭回归 (Ridge Regression)**|$\min_{\boldsymbol{w}} \Vert{} \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \Vert{}_2^2 + \lambda \Vert{}\boldsymbol{w}\Vert{}_2^2$|加入 $L_2$ 正则化。使参数数值整体缩小，平滑模型；使得 $(\boldsymbol{X}^T \boldsymbol{X} + \lambda \boldsymbol{I})$ 一定可逆。|
|**Lasso 回归 (Lasso Regression)**|$\min_{\boldsymbol{w}} \Vert{} \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \Vert{}_2^2 + \lambda \Vert{}\boldsymbol{w}\Vert{}_1$|加入 $L_1$ 正则化。在参数空间形成棱角分明的约束域，容易产生**稀疏解**，实现自动特征选择。|
|**弹性网络 (Elastic Net)**|$\min_{\boldsymbol{w}} \Vert{} \boldsymbol{X}\boldsymbol{w} - \boldsymbol{y} \Vert{}_2^2 + \lambda_1 \Vert{}\boldsymbol{w}\Vert{}_1 + \lambda_2 \Vert{}\boldsymbol{w}\Vert{}_2^2$|融合 $L_1$ 与 $L_2$ 正则化，兼具特征选择与平滑抗共线性的能力。|

### 六、 PyTorch 实现：从底层手写到标准模块

#### 1. 采用纯矩阵运算与梯度下降手写实现

Python

```
import torch

# 1. 构造合成数据集: y = 2 * x_1 - 3.4 * x_2 + 4.2 + noise
torch.manual_seed(42)
num_inputs = 2
num_examples = 1000

true_w = torch.tensor([2.0, -3.4]).unsqueeze(1)
true_b = 4.2

X = torch.randn(num_examples, num_inputs)
y = torch.mm(X, true_w) + true_b
y += torch.randn_like(y) * 0.1 # 添加高斯噪声

# 2. 初始化可学习参数
w = torch.randn((num_inputs, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 3. 超参数设置
lr = 0.03
epochs = 500

# 4. 训练循环 (梯度下降)
for epoch in range(epochs):
    # 前向传播: 计算预测值
    y_hat = torch.mm(X, w) + b
    
    # 计算 MSE 损失
    loss = torch.mean((y_hat - y) ** 2) / 2
    
    # 反向传播求梯度
    loss.backward()
    
    # 更新参数 (禁止追踪梯度)
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        # 清空梯度
        w.grad.zero_()
        b.grad.zero_()

print("估计的 w:", w.squeeze().detach().numpy())
print("真实的 w:", true_w.squeeze().numpy())
print("估计的 b:", b.item())
print("真实的 b:", true_b)
```

#### 2. 使用 PyTorch 高阶 API (`torch.nn.Linear`) 实现

Python

```
import torch
import torch.nn as nn

# 使用 nn.Module 快速构建线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(input_dim=2)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)

# 拟合数据...
for epoch in range(500):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

print("\n[nn.Module 验证]")
print("学习得到的权重:", model.linear.weight.detach().numpy())
print("学习得到的偏置:", model.linear.bias.item())
```

### 七、 总结与知识脉络

1. **基本模型**：用线性的特征组合 $y = \boldsymbol{w}^T \boldsymbol{x} + b$ 拟合连续目标值。
    
2. **优化目标**：最小化残差平方和（MSE），几何上是点到线的距离平方，概率上对应正态分布假设下的**最大似然估计（MLE）**。
    
3. **求解算法**：
    
    - 小数据/低维：直接求正规方程解析解 $\boldsymbol{w}^* = (\boldsymbol{X}^T \boldsymbol{X})^{-1} \boldsymbol{X}^T \boldsymbol{y}$；
        
    - 大数据/高维：采用梯度下降（SGD、Adam）进行迭代优化。
        
4. **扩展延申**：通过引入 $L_1/L_2$ 正则化演变为 Lasso 与 Ridge；通过引入激活函数（如 Sigmoid），可进一步衍生为解决分类问题的**逻辑回归（Logistic Regression）**。

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/经典机器学习/Logistic Regression - 逻辑回归|Logistic Regression - 逻辑回归]]：比较连续值回归与概率分类。
- [[40_课程笔记/吴恩达机器学习/线性回归 - 课程笔记|线性回归 - 课程笔记]]：对照课程记录与系统讲解。

<!-- knowledge-links:end -->
