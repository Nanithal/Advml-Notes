---
title: "PCA - 主成分分析"
aliases:
  - "PCA"
分类: "经典机器学习"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[10_基础与算法/经典机器学习/索引 - 经典机器学习|经典机器学习索引]]

# PCA - 主成分分析


**主成分分析（Principal Component Analysis, PCA）**是机器学习与数据挖掘领域最经典、应用最广泛的**无监督线性降维算法**。由卡尔·皮尔逊（Karl Pearson）于 1901 年提出。

在实际项目中，我们经常遇到**高维数据**（比如几千维的图像像素、上百种指标的表格数据）。高维数据不仅会导致“维度灾难”（计算极慢、模型容易过拟合），而且数据间往往存在大量冗余和噪声。

PCA 的核心使命就是：**在尽量少丢失原始数据信息的前提下，把高维数据映射到低维空间中。**

为了让你彻底学会，我们将从 **“几何直观图像 $\rightarrow$ PCA 的两大核心哲学观点 $\rightarrow$ 严格数学推导 $\rightarrow$ SVD 优化解法 $\rightarrow$ Python 手写与验证”** 逐步深度拆解。

### 一、 核心几何直观：什么是“最大方差”与“最小投影误差”？

假设我们在二维平面上有一组样本点（特征 $x_1, x_2$），这些点呈椭圆形分布：



```Plaintext
       x_2 ▲          
           │           *   / u_1 (第一主成分方向: 方差最大)
           │         *   * / 
           │       *   *  /
           │     *   *   /
           │   *   *    /
           │  /        /
           └───────────┴──────────► x_1
                       \
                        \ u_2 (第二主成分方向: 与 u_1 正交)
```

如果我们想把数据从 **二维（$x_1, x_2$）降到一维**（即投影到一条直线上），应该选哪条直线？

1. **观察方案 A**：如果把点垂直投影到 $x_1$ 轴，原本在斜上方拉开距离的点，投影后会挤在一块。
    
2. **观察方案 B（PCA 方案）**：选择沿椭圆长轴方向的斜线 $u_1$ 投影！
    
    - **从方差角度看**：数据在 $u_1$ 方向上的**分散程度（方差）最大**，说明保留的数据动态范围最大、差异性最明显，信息损失最少。
        
    - **从误差角度看**：所有原始样本点垂直投影到 $u_1$ 线的**距离平方和（投影误差）最小**。
        

**直观结论**：PCA 找到的新坐标轴（主成分 $u_1, u_2, \dots$），本质上就是数据分布**变异程度（方差）最大的方向**，且各个新坐标轴之间保持**互相垂直（正交）**。

### 二、 PCA 的数学目标与符号定义

假设我们有 $N$ 个样本，每个样本有 $d$ 个特征。

#### 1. 数据预处理：中心化（Zero-Centered）

这是 PCA 运行前**最关键**的一步：将每个特征维度减去其均值，使数据的中心移动到坐标原点。

设中心化后的数据矩阵为 $X \in \mathbb{R}^{N \times d}$：

$$\mu = \frac{1}{N} \sum_{i=1}^N x_i$$

$$x_i \leftarrow x_i - \mu$$

中心化后，样本的均值为零向量（$\mathbb{E}[x] = \mathbf{0}$）。

#### 2. 协方差矩阵（Covariance Matrix）

中心化后，数据集的**协方差矩阵 $C$** 可极其优雅地写为矩阵乘法形式：

$$C = \frac{1}{N} X^T X \quad \in \mathbb{R}^{d \times d}$$

- 对角线元素 $C_{ii}$ 是第 $i$ 个特征的**方差**（Variance）。
    
- 非对角线元素 $C_{ij}$ 是第 $i$ 个特征与第 $j$ 个特征的**协方差**（Covariance）。
    
- **PCA 的终极目的**：寻找一个新的正交基，使得降维后的数据**对角线方差尽可能大，非对角线协方差为 0（彻底消除特征间的冗余线性相关性）**。
    

### 三、 严格数学推导：最大方差理论

我们来推导如何寻找**第一主成分（第一个投影向量 $u_1 \in \mathbb{R}^{d \times 1}$）**。

#### 1. 建模投影后的方差

设 $u_1$ 为单位方向向量（满足约束 $u_1^T u_1 = 1$）。 样本 $x_i$ 投影到 $u_1$ 上的值为标量 $u_1^T x_i$。

由于数据已中心化，投影后的均值也为 0。因此，所有样本在 $u_1$ 方向上的**投影方差**为：

$$\text{Var}(u_1) = \frac{1}{N} \sum_{i=1}^N (u_1^T x_i)^2 = \frac{1}{N} \sum_{i=1}^N u_1^T x_i x_i^T u_1 = u_1^T \left( \frac{1}{N} X^T X \right) u_1 = u_1^T C u_1$$

#### 2. 构建目标函数（拉格朗日乘子法）

我们的目标是最大化投影方差 $\text{Var}(u_1) = u_1^T C u_1$，同时满足约束 $u_1^T u_1 = 1$。

构造拉格朗日函数 $L(u_1, \lambda)$：

$$L(u_1, \lambda) = u_1^T C u_1 - \lambda (u_1^T u_1 - 1)$$

#### 3. 求偏导并令其为 0

对向量 $u_1$ 求偏导：

$$\frac{\partial L}{\partial u_1} = 2 C u_1 - 2 \lambda u_1 = 0$$

得出核心方程：

$$C u_1 = \lambda u_1$$

> **数学结论**：
> 
> 1. **投影向量 $u_1$ 正好是协方差矩阵 $C$ 的特征向量（Eigenvector）！**
>     
> 2. 将 $C u_1 = \lambda u_1$ 代回方差公式：$\text{Var}(u_1) = u_1^T C u_1 = u_1^T (\lambda u_1) = \lambda (u_1^T u_1) = \lambda$。
>     
> 3. **投影后的方差大小，正好等于对应的特征值（Eigenvalue） $\lambda$！**
>     

为了使方差最大，我们只需要选择**最大特征值对应的特征向量**作为第一主成分；第二大特征值对应的特征向量作为第二主成分，依此类推。

### 四、 现代工程求解：SVD（奇异值分解）优化

在实际工程实现（如 PyTorch、`scikit-learn`）中，**几乎从不直接去计算 $X^T X$ 并显式求解特征值**。因为当特征数 $d$ 非常大时，计算 $d \times d$ 的协方差矩阵极其消耗内存和计算资源。

工业界采用 **SVD（Singular Value Decomposition）** 求解：

1. 对中心化后的矩阵 $X \in \mathbb{R}^{N \times d}$ 直接做奇异值分解：
    
    $$X = U \Sigma V^T$$
    
    其中 $V \in \mathbb{R}^{d \times d}$ 的列向量就是 $X^T X$ 的特征向量（主成分方向）。
    
2. 则协方差矩阵：
    
    $$C = \frac{1}{N} X^T X = \frac{1}{N} (U \Sigma V^T)^T (U \Sigma V^T) = \frac{1}{N} V \Sigma^T U^T U \Sigma V^T = V \left( \frac{\Sigma^2}{N} \right) V^T$$
    
3. 特征值与奇异值 $\sigma_i$ 的关系为：
    
    $$\lambda_i = \frac{\sigma_i^2}{N}$$
    

通过 SVD，不仅计算数值更稳定，而且能以极快的速度直接提取出前 $k$ 个最大主成分。

### 五、 核心应用决策：如何选择保留多少个主成分（$k$）？

我们通过计算 **累计方差贡献率（Explained Variance Ratio）** 来决定将维度降到多少：

单个主成分 $i$ 的方差解释率：

$$\text{Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

前 $k$ 个主成分的累计方差解释率：

$$\text{Cumulative Ratio}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

在工程中，通常设定一个阈值（例如 **$85\%$ 或 $95\%$**），取能满足该累计方差比例的最小 $k$ 值。

### 六、 Python 手写实现（基于 NumPy）

下面我们不调用 `scikit-learn`，直接根据上文推导的公式，用纯 NumPy 实现一个完整的 PCA 类：

Python

```
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components # 目标降维维度 k
        self.components = None           # 主成分矩阵 (k x d)
        self.mean = None                 # 样本均值 (1 x d)
        self.explained_variance_ratio_ = None # 方差贡献率

    def fit(self, X):
        # 1. 中心化数据 (Zero-Centering)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 计算协方差矩阵 C = (1/N) * X^T * X
        # np.cov 默认行是变量、列是样本，所以需要转置 .T
        cov_matrix = np.cov(X_centered.T)

        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 特征向量是以列形式排列的，转置为行向量方便处理
        eigenvectors = eigenvectors.T

        # 4. 按特征值从大到小排序
        sort_idxs = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sort_idxs]
        sorted_eigenvectors = eigenvectors[sort_idxs]

        # 5. 提取前 k 个主成分
        self.components = sorted_eigenvectors[:self.n_components]
        
        # 6. 计算方差解释率
        total_variance = np.sum(sorted_eigenvalues)
        self.explained_variance_ratio_ = sorted_eigenvalues[:self.n_components] / total_variance

    def transform(self, X):
        # 降维变换：Y = (X - mean) * W^T
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# --- 实验验证 ---
if __name__ == "__main__":
    np.random.seed(42)
    # 构造一个 100 样本、4 特征的随机数据集
    X = np.random.randn(100, 4)

    # 实例化 PCA，将 4 维降至 2 维
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print("原始数据形状:", X.shape)            # (100, 4)
    print("降维后数据形状:", X_transformed.shape) # (100, 2)
    print("各主成分方差贡献率:", pca.explained_variance_ratio_)
    print("累计保留的信息量:", np.sum(pca.explained_variance_ratio_))
```

### 七、 PCA 的优缺点与适用场景

#### 优点：

1. **完全无监督**：不需要任何类别标签，仅依赖数据自身的分布即可完成降维。
    
2. **彻底消除线性冗余**：降维后的各个主成分之间两两正交（协方差为 0），排除了特征重叠问题。
    
3. **计算高效**：具有明确的解析解（特征值分解/SVD），不存在局部极小值问题。
    

#### 局限性：

1. **主成分缺乏可解释性**：降维后的新特征是原始所有特征的线性组合，难以赋予具体的物理或业务含义。
    
2. **仅能捕获线性关系**：PCA 是线性降维方法，如果数据存在高度非线性的流形结构（如 S 型曲线、瑞士卷），PCA 会失效（此时需要使用 **Kernel PCA** 或 **t-SNE / UMAP** 等非线性降维算法）。
    
3. **对异常值敏感**：因为方差计算包含平方项，离群点/极值会对主成分的方向产生巨大干扰（训练前需做数据清洗或 Standard Scaling 标准化）。

---

## 笔记联系

- [[10_基础与算法/数学基础/统计基础 - 方差、协方差与协方差矩阵|统计基础 - 方差、协方差与协方差矩阵]]：回顾方差与协方差矩阵。
- [[10_基础与算法/经典机器学习/LDA - 线性判别分析|LDA - 线性判别分析]]：比较无监督保方差与有监督类别分离。
- [[10_基础与算法/经典机器学习/LLE - 局部线性嵌入|LLE - 局部线性嵌入]]：比较线性投影与局部几何保持。
- [[20_深度学习/网络结构/AE - 自编码器|AE - 自编码器]]：比较线性降维与非线性重构表示。
- [[10_基础与算法/经典机器学习/t-SNE - t分布随机近邻嵌入|t-SNE - t分布随机近邻嵌入]]：比较低维可视化与线性主成分投影。
- [[20_深度学习/训练基础/归一化 - 图片笔记|归一化 - 图片笔记]]：可对照 PCA 中的数据预处理讨论；原归一化页为图片笔记。
