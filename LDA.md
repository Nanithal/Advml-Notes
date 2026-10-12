
线性判别分析（LDA）由统计学家 Fisher 于 1936 年提出（因此有时也叫 **Fisher's Linear Discriminant**）。它的核心思想用一句话高度概括就是：**“投影后，类内方差最小，类间方差最大”**。

为了让你彻底教会并搞懂它，我们将从 **“几何直观图像 $\rightarrow$ 数学定义与矩阵化 $\rightarrow$ 严格公式推导 $\rightarrow$ 多分类扩展 $\rightarrow$ Python 手写与验证”** 逐步全景拆解。

### 一、 核心几何直观：寻找最佳“投影方向”

假设我们有一组二维数据（特征 $x_1, x_2$），分为红、蓝两类：

Plaintext

```
       x_2 ▲          
           │       红 类  (Class 1)
           │      *   *
           │    *   *   *
           │
           │       o   o   o
           │     o   o   o  蓝 类 (Class 2)
           └────────────────────────► x_1
```

如果我们想把这些数据从 **二维降到一维**（即投影到一条直线上）：

1. **如果随意选择一条直线**（比如直接投影到 $x_1$ 轴）：红蓝两类的点可能会混杂堆叠在一起，无法区分。
    
2. **LDA 的策略**：找到一条特殊的直线，使得投影之后：
    
    - **不同类别之间的中心点距离尽量远**（类间分离度高）；
        
    - **同一类别内部的数据点尽量紧凑**（类内离散度低）。
        

这样，在投影后的低维空间里，两类数据就能被极其轻松地用一个分割点彻底分开。

### 二、 数学符号定义与核心目标

我们先以 **二分类任务** 为例进行数学建模：

#### 1. 符号定义

- 数据集包含两个类别 $C_1$ 和 $C_2$，样本量分别为 $N_1, N_2$。
    
- 原始样本特征向量为 $x_i \in \mathbb{R}^{d \times 1}$（高维空间）。
    
- 两个类别的**均值向量**分别为：
    
    $$\mu_1 = \frac{1}{N_1} \sum_{x \in C_1} x, \quad \mu_2 = \frac{1}{N_2} \sum_{x \in C_2} x$$
    
- 我们需要寻找一个**投影向量 $w \in \mathbb{R}^{d \times 1}$**，将样本 $x$ 映射到一维标量 $y$：
    
    $$y = w^T x$$
    

投影后：

- 两个类别的投影均值为：$\tilde{\mu}_1 = w^T \mu_1, \quad \tilde{\mu}_2 = w^T \mu_2$
    
- 两个类别的投影方差为：
    
    $$\tilde{s}_1^2 = \sum_{x \in C_1} (w^T x - w^T \mu_1)^2, \quad \tilde{s}_2^2 = \sum_{x \in C_2} (w^T x - w^T \mu_2)^2$$
    

#### 2. 矩阵化描述（类内散度与类间散度）

为了优雅地用线性代数推导，我们引入两个核心矩阵：

##### ① 类内散度矩阵（Within-class Scatter Matrix, $S_w$）

衡量每个类别内部数据的离散程度：

$$S_w = S_1 + S_2$$

其中 $S_1 = \sum_{x \in C_1} (x - \mu_1)(x - \mu_1)^T$， $S_2 = \sum_{x \in C_2} (x - \mu_2)(x - \mu_2)^T$。

将投影后的类内方差和用 $S_w$ 表示：

$$\tilde{s}_1^2 + \tilde{s}_2^2 = w^T S_1 w + w^T S_2 w = w^T S_w w$$

##### ② 类间散度矩阵（Between-class Scatter Matrix, $S_b$）

衡量两个类别的均值中心之间的距离：

$$S_b = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$$

将投影后的均值距离平方用 $S_b$ 表示：

$$(\tilde{\mu}_1 - \tilde{\mu}_2)^2 = (w^T \mu_1 - w^T \mu_2)^2 = w^T (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T w = w^T S_b w$$

### 三、 目标函数与公式严格推导（Fisher 准则）

我们将 LDA 的核心思想表达为优化目标函数 **Fisher 准则函数（Fisher's Criterion）** $J(w)$：

$$J(w) = \frac{(\tilde{\mu}_1 - \tilde{\mu}_2)^2}{\tilde{s}_1^2 + \tilde{s}_2^2} = \frac{w^T S_b w}{w^T S_w w}$$

**我们的目标是：求一个 $w$，使得 $J(w)$ 达到最大值！**

#### 极简推导过程：

1. **确定求解条件**：
    
    由于分子和分母都是关于 $w$ 的二次项，放大或缩小 $w$ 的模长（缩放倍数）对 $J(w)$ 的结果毫无影响（因为我们只需要方向）。
    
    因此，我们可以固定分母为常数 1，即约束条件为：
    
    $$w^T S_w w = 1$$
    
    优化问题转化为：
    
    $$\max_w \ w^T S_b w \quad \text{s.t.} \ w^T S_w w = 1$$
    
2. **拉格朗日乘子法（Lagrange Multiplier）**：
    
    构造拉格朗日函数：
    
    $$L(w, \lambda) = w^T S_b w - \lambda (w^T S_w w - 1)$$
    
3. **对 $w$ 求偏导并令其等于 0**：
    
    $$\frac{\partial L}{\partial w} = 2 S_b w - 2 \lambda S_w w = 0$$
    
    $$S_b w = \lambda S_w w$$
    
4. **转化为广义特征值问题**：
    
    若类内散度矩阵 $S_w$ 可逆，两边同乘以 $S_w^{-1}$：
    
    $$S_w^{-1} S_b w = \lambda w$$
    
    > **结论**：**投影向量 $w$ 就是矩阵 $S_w^{-1} S_b$ 的特征向量！**
    
5. **进一步简化（二分类特例）**：
    
    注意到 $S_b w = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T w$。
    
    其中 $(\mu_1 - \mu_2)^T w$ 是一个**标量（记为常数 $c$）**。
    
    所以：
    
    $$S_b w = c (\mu_1 - \mu_2)$$
    
    代入 $S_b w = \lambda S_w w$ 得：
    
    $$\lambda S_w w = c (\mu_1 - \mu_2) \implies w = \frac{c}{\lambda} S_w^{-1} (\mu_1 - \mu_2)$$
    
    因为我们只关心 $w$ 的方向，忽略常数缩放，二分类下最优投影方向极其简洁：
    
    $$w^* = S_w^{-1} (\mu_1 - \mu_2)$$
    

### 四、 多分类扩展（Multi-class LDA）

如果是 $C$ 个类别的多分类任务（$C > 2$），如何将 $d$ 维特征降维到 $k$ 维（$k < C$）？

1. **重新定义散度矩阵**：
    
    - **全局均值**：$\mu = \frac{1}{N} \sum_{i=1}^N x_i$
        
    - **类内散度矩阵**：$S_w = \sum_{j=1}^C \sum_{x \in C_j} (x - \mu_j)(x - \mu_j)^T$
        
    - **类间散度矩阵**：$S_b = \sum_{j=1}^C N_j (\mu_j - \mu)(\mu_j - \mu)^T$
        
2. **求解特征值**：
    
    求解矩阵 $S_w^{-1} S_b$ 的特征值与特征向量：
    
    $$S_w^{-1} S_b W = \lambda W$$
    
3. **选择前 $k$ 个最大特征值对应的特征向量**，组合成投影矩阵 $W \in \mathbb{R}^{d \times k}$。
    
    降维变换公式为：$Y = W^T X$。
    

> **致命约束（维度的上限）**：
> 
> 因为 $S_b$ 是由 $C$ 个秩为 1 的矩阵相加而成（且存在全局均值约束），**$S_b$ 的矩阵秩最大只有 $C - 1$**。
> 
> 因此，**LDA 最多只能将数据降维到 $C - 1$ 维**！
> 
> （例如三分类数据，最高只能降维到 2 维）。

### 五、 LDA 与 PCA 的核心对比

这是面试和实际工程中最常被拿来对比的两个算法：

|**维度**|**PCA（主成分分析）**|**LDA（线性判别分析）**|
|---|---|---|
|**学习模式**|**无监督**（Unsupervised）|**监督**（Supervised，需要类别标签）|
|**核心思想**|寻找使数据**方差最大**（信息留存最多）的方向|寻找使**类间方差最大、类内方差最小**的方向|
|**应用目标**|通用特征降维、去噪、可视化|特征提取、针对分类任务的预处理|
|**降维上限**|最高可降至原始特征数 $d$|**最高只能降至 $C - 1$ 维**（$C$ 为类别数）|

Plaintext

```
       PCA 视角 (只看数据整体分布，容易混淆类别)
              \  
               \  [o  *  o  *  o  *]
                \
─────────────────────────────────────────────►
       LDA 视角 (利用标签信息，完美拉开类间距离)
```

### 六、 Python 手写实现（基于 NumPy）

下面我们不调用 `scikit-learn`，直接根据上面推导的公式用纯 NumPy 手写一个二分类 LDA：

Python

```
import numpy as np
import matplotlib.pyplot as plt

class LDA:
    def __init__(self):
        self.w = None  # 最佳投影方向

    def fit(self, X, y):
        # 1. 拆分类别
        X1 = X[y == 0]
        X2 = X[y == 1]

        # 2. 计算各类的均值向量 (d x 1)
        mu1 = np.mean(X1, axis=0)
        mu2 = np.mean(X2, axis=0)

        # 3. 计算类内散度矩阵 Sw
        # (x - mu1)^T * (x - mu1)
        S1 = np.dot((X1 - mu1).T, (X1 - mu1))
        S2 = np.dot((X2 - mu2).T, (X2 - mu2))
        Sw = S1 + S2

        # 4. 根据推导公式求解 w = Sw^-1 * (mu1 - mu2)
        # 使用 np.linalg.pinv 求伪逆，防止 Sw 不可逆矩阵报错
        Sw_inv = np.linalg.pinv(Sw)
        self.w = np.dot(Sw_inv, (mu1 - mu2))

    def transform(self, X):
        # 将原始数据投影到一维空间
        return np.dot(X, self.w)

# --- 实验验证 ---
if __name__ == "__main__":
    np.random.seed(42)

    # 模拟构建两类二维高斯分布数据
    cov = [[1, 0.8], [0.8, 1]]
    X1 = np.random.multivariate_normal([2, 2], cov, 100) # 类 0
    X2 = np.random.multivariate_normal([6, 6], cov, 100) # 类 1
    
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(100), np.ones(100)))

    # 训练 LDA
    lda = LDA()
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("推导出的最佳投影向量 w:", lda.w)
    print("降维后数据形状:", X_projected.shape) # (200,)
```

### 七、 LDA 的优缺点与适用场景

#### 优点：

1. **分类导向强**：在降维过程中充分利用了类别标签信息，降维后的特征非常利于分类器（如逻辑回归、SVM）进行分类。
    
2. **计算高效**：解析解直接通过矩阵求导和特征值分解得到，不需要昂贵的梯度下降迭代训练。
    

#### 局限性：

1. **线性假设**：LDA 假设数据是线性可分的，如果数据的决策边界是非线性的（如环形分布），LDA 将失效（此时需借助于 Kernel LDA 核技巧）。
    
2. **假设数据服从高斯分布**：LDA 隐式假设各类数据服从相同协方差矩阵的正态分布，若实际数据偏离较远，效果会下降。
    
3. **样本量限制**：当特征维度 $d$ 远远大于样本量 $N$ 时（如基因序列数据），类内散度矩阵 $S_w$ 会奇异不可逆（奇异矩阵问题），需要配合正则化（Regularized LDA）。