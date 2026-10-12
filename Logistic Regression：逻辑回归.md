**逻辑回归（Logistic Regression）** 虽然名字里带着“回归”二字，但它本质上是一个经典的**监督学习分类算法**（主要用于**二分类问题**，也可以扩展至多分类）。

线性回归用来预测连续数值（如房价、温度），而逻辑回归则是预测**某个事件发生的概率**（如这封邮件是垃圾邮件的概率、用户点击广告的概率）。

下面将从 **从回归到分类的物理直观 $\rightarrow$ Sigmoid 函数映射 $\rightarrow$ 损失函数（交叉熵/对数损失）推导 $\rightarrow$ 梯度下降优化 $\rightarrow$ 与线性回归/SVM/神经网络的对比 $\rightarrow$ PyTorch 代码实现** 进行全景拆解。

### 一、 核心思想：从线性回归到概率映射

线性回归的模型形式为：

$$z = \boldsymbol{w}^T \boldsymbol{x} + b$$

其输出 $z$ 的取值范围是整个实数域 $(-\infty, +\infty)$。如果直接用 $z$ 来表示概率，会出现两个严重问题：

1. **边界溢出**：概率必须在 $[0, 1]$ 之间，而 $z$ 会超出这个范围。
    
2. **缺乏鲁棒性**：异常值（Outliers）会剧烈拉动线性拟合直线，导致分类阈值发生严重偏移。
    

#### 解决办法：加入激活函数映射

逻辑回归的核心机制是：**先用线性回归计算一个综合打分 $z$，再通过一个非线性的激活函数（Sigmoid 函数），将 $z$ 压缩映射到 $(0, 1)$ 的区间内，解释为事件发生的概率。**

$$\hat{y} = p(y=1 \mid \boldsymbol{x}) = \sigma(z) = \sigma(\boldsymbol{w}^T \boldsymbol{x} + b)$$

### 二、 数学基石：Sigmoid 函数与对数几率（Odds）

#### 1. Sigmoid（S型）函数定义

Sigmoid 函数（也叫 Logistic 函数）公式为：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- **定义域**：$(-\infty, +\infty)$
    
- **值域**：$(0, 1)$，完美匹配概率的定义。
    
- **决策边界（Decision Boundary）**：
    
    - 当 $z = \boldsymbol{w}^T \boldsymbol{x} + b > 0$ 时，$\sigma(z) > 0.5$，预测为类别 1；
        
    - 当 $z = \boldsymbol{w}^T \boldsymbol{x} + b < 0$ 时，$\sigma(z) < 0.5$，预测为类别 0；
        
    - $z = 0$ 对应的超平面即为**决策边界**。
        

#### 2. Sigmoid 函数的关键导数性质

Sigmoid 函数在数学上有一个非常优雅的导数性质，这使得后续的梯度求导极其简洁：

$$\sigma'(z) = \sigma(z) (1 - \sigma(z))$$

> **推导简证**：
> 
> $\sigma'(z) = \frac{d}{dz}(1 + e^{-z})^{-1} = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z)(1 - \sigma(z))$。

#### 3. 统计学解释：对数几率（Log-Odds / Logit）

在统计学中，一个事件发生的概率为 $p$，不发生的概率为 $1-p$。

- **几率（Odds）** 定义为事件发生与不发生的概率比值：$\text{Odds} = \frac{p}{1-p}$。
    
- 对几率取自然对数，即得到**对数几率（Logit）**：
    

$$\ln \left( \frac{p}{1-p} \right) = \ln \left( \frac{\sigma(z)}{1 - \sigma(z)} \right) = z = \boldsymbol{w}^T \boldsymbol{x} + b$$

**结论**：**逻辑回归本质上是用线性回归模型去拟合样本为正例的“对数几率”（Log-Odds）。**

### 三、 损失函数：交叉熵损失（Cross-Entropy Loss / Log Loss）

线性回归使用均方误差（MSE）作为损失函数，但如果把 Sigmoid 函数代入 MSE 损失中，导出的损失函数是一个**非凸函数（Non-convex Function）**，拥有大量的局部极小值点，无法使用梯度下降保证找到全局最优解。

为了得到凸优化问题，逻辑回归采用了最大似然估计（MLE）推导出的 **二元交叉熵损失函数（Binary Cross-Entropy Loss, BCE）**。

#### 1. 概率模型表达

设 $p = P(y=1 \mid \boldsymbol{x}) = \sigma(\boldsymbol{w}^T \boldsymbol{x} + b)$，则：

- 当 $y = 1$ 时，$P(y \mid \boldsymbol{x}) = p$
    
- 当 $y = 0$ 时，$P(y \mid \boldsymbol{x}) = 1 - p$
    

将这两个式子巧妙地合并写成一个统一的伯努利分布概率密度公式：

$$P(y \mid \boldsymbol{x}) = p^y (1 - p)^{1 - y} \quad (y \in \{0, 1\})$$

#### 2. 最大似然估计（MLE）推导

假设训练集中有 $m$ 个独立同分布的样本 $\{(\boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_2), \dots, (\boldsymbol{x}_m, y_m)\}$，构造整体似然函数：

$$L(\boldsymbol{w}, b) = \prod_{i=1}^{m} p_i^{y_i} (1 - p_i)^{1 - y_i}$$

取对数得到**对数似然函数**：

$$\ell(\boldsymbol{w}, b) = \ln L(\boldsymbol{w}, b) = \sum_{i=1}^{m} \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$

#### 3. 定义损失函数 $J(\boldsymbol{w}, b)$

机器学习的目标是**最小化损失**。因此将对数似然取负数并取平均，就得到了**二元交叉熵损失函数**：

$$J(\boldsymbol{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \right]$$

其中 $p_i = \sigma(\boldsymbol{w}^T \boldsymbol{x}_i + b)$。

- 当 $y_i = 1$ 时，损失为 $-\ln(p_i)$：若 $p_i \to 1$，损失趋于 $0$；若 $p_i \to 0$，损失趋于 $+\infty$（严重惩罚误判）。
    
- 当 $y_i = 0$ 时，损失为 $-\ln(1 - p_i)$：若 $p_i \to 0$，损失趋于 $0$；若 $p_i \to 1$，损失趋于 $+\infty$。
    

### 四、 参数求解：梯度下降与导数推导

为了利用梯度下降更新参数：$\boldsymbol{w} \leftarrow \boldsymbol{w} - \alpha \nabla_{\boldsymbol{w}} J$，我们需要求损失函数对 $\boldsymbol{w}$ 的偏导数。

利用链式法则（Chain Rule）：

$$\frac{\partial J}{\partial \boldsymbol{w}} = \frac{\partial J}{\partial p} \cdot \frac{\partial p}{\partial z} \cdot \frac{\partial z}{\partial \boldsymbol{w}}$$

各个部分分别求解：

1. $\frac{\partial J}{\partial p} = -\frac{1}{m} \left( \frac{y}{p} - \frac{1-y}{1-p} \right) = -\frac{1}{m} \frac{y - p}{p(1-p)}$
    
2. $\frac{\partial p}{\partial z} = \sigma'(z) = p(1-p)$
    
3. $\frac{\partial z}{\partial \boldsymbol{w}} = \boldsymbol{x}$
    

将三项相乘（注意到 $p(1-p)$ 刚好被完全约简）：

$$\frac{\partial J}{\partial \boldsymbol{w}} = \frac{1}{m} \sum_{i=1}^{m} (p_i - y_i) \boldsymbol{x}_i$$

同样的，对偏置 $b$ 的偏导数为：

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (p_i - y_i)$$

**发现惊人的一致性**：

逻辑回归对 $\boldsymbol{w}$ 的梯度形式为 $\frac{1}{m} \mathbf{X}^T (\mathbf{p} - \mathbf{y})$，**这与线性回归的梯度公式形式完全一模一样！** 唯一的区别在于这里的 $p_i = \sigma(\boldsymbol{w}^T \boldsymbol{x}_i + b)$ 经过了 Sigmoid 函数的非线性变换。

### 五、 扩展与对比

#### 1. 多分类扩展：Softmax 回归（Multinomial Logistic Regression）

当类别数 $K > 2$ 时，逻辑回归可以扩展为 Softmax 回归：

- 将单一输出改为 $K$ 个神经元输出 $z_1, z_2, \dots, z_K$。
    
- 使用 Softmax 函数计算每个类别的概率：
    
    $$P(y = k \mid \boldsymbol{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$
    
- 损失函数扩展为**多分类交叉熵损失（Categorical Cross-Entropy）**。
    

#### 2. 与其他模型的横向对比

|**比较维度**|**逻辑回归 (Logistic Regression)**|**线性回归 (Linear Regression)**|**支持向量机 (SVM)**|
|---|---|---|---|
|**任务类型**|分类 (输出概率)|回归 (输出连续值)|分类 / 回归|
|**激活/映射**|Sigmoid / Softmax|无（恒等映射）|Sign (符号函数)|
|**损失函数**|交叉熵损失 (BCE / CE)|均方误差 (MSE)|合页损失 (Hinge Loss)|
|**样本敏感度**|依赖全量样本建立概率分布|依赖全量样本|仅依赖**支持向量**（决策边界附近的点）|

### 六、 PyTorch 实战实现

#### 1. 使用原生矩阵运算与梯度下降手写实现

Python

```
import torch

# 1. 构建模拟的二分类数据集
torch.manual_seed(42)
n_samples, n_features = 100, 2

# 构造两个类别的点，做简单线性可分
X_class0 = torch.randn(n_samples // 2, n_features) - 1.5
X_class1 = torch.randn(n_samples // 2, n_features) + 1.5

X = torch.cat([X_class0, X_class1], dim=0)
y = torch.cat([torch.zeros(n_samples // 2, 1), torch.ones(n_samples // 2, 1)], dim=0)

# 2. 初始化参数
w = torch.randn(n_features, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 3. 训练循环
lr = 0.1
epochs = 200

for epoch in range(epochs):
    # 线性计算 z = Xw + b
    z = torch.mm(X, w) + b
    # Sigmoid 激活得到预测概率 p
    p = 1.0 / (1.0 + torch.exp(-z))
    
    # 计算二元交叉熵损失
    # 加上 1e-7 防止 log(0) 导致数值不稳定
    loss = -torch.mean(y * torch.log(p + 1e-7) + (1 - y) * torch.log(1 - p + 1e-7))
    
    # 反向传播求梯度
    loss.backward()
    
    # 参数更新
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        w.grad.zero_()
        b.grad.zero_()

print("手写实现的最终损失:", loss.item())
print("拟合权重 w:", w.detach().squeeze().numpy())
print("拟合偏置 b:", b.item())
```

#### 2. 使用 PyTorch 标准模块 `nn.BCELoss` 实现

Python

```
import torch
import torch.nn as nn

class LogisticRegressionModel(nn.Module):
    def __init__(self, in_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))

model = LogisticRegressionModel(in_features=2)
# 使用 BCE 损失函数与 SGD 优化器
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    optimizer.zero_grad()
    predictions = model(X)
    loss = criterion(predictions, y)
    loss.backward()
    optimizer.step()

print("\n[PyTorch 标准模块验证]")
print("网络学习得到的权重:", model.linear.weight.detach().numpy())
print("网络学习得到的偏置:", model.linear.bias.item())
```

### 七、 核心知识点梳理

1. **基本本质**：逻辑回归是广义线性模型（GLM）的一种，本质是利用线性模型拟合样本的对数几率（Logit）。
    
2. **激活函数**：Sigmoid 函数将连续的拟合值映射到 $(0, 1)$ 区间，提供直观的概率解释。
    
3. **损失函数**：由于 MSE 会导致非凸优化，逻辑回归基于最大似然估计推导出**二元交叉熵损失**（BCE Loss），具备良好的凸函数性质。
    
4. **梯度求解**：其梯度的数学形式与线性回归惊人地一致，可以采用 SGD、Adam 等优化器快速求解。