---
title: "Graph Laplacian - 图拉普拉斯矩阵"
aliases:
  - "Grap Loplacian 图拉普拉斯矩阵"
分类: "数学基础"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[10_基础与算法/数学基础/索引 - 数学基础|数学基础索引]]

# Graph Laplacian - 图拉普拉斯矩阵

**图拉普拉斯矩阵（Graph Laplacian Matrix）** 是图信号处理（Graph Signal Processing, GSP）、图神经网络（GNN，特别是谱卷积 GCN）、图聚类（Spectral Clustering）以及流形学习（Manifold Learning）的**绝对理论核心与基石**。

如果把图（Graph）看作是连续物理空间在离散点上的采样，**图拉普拉斯矩阵就是连续空间中“微积分拉普拉斯算子（$\Delta = \nabla^2$）”在离散图结构上的完美延伸**。

为了让你全面且透彻地理解 Graph Laplacian，我们将从 **“直观物理图像与定义 $\rightarrow$ 离散拉普拉斯算子的推导 $\rightarrow$ 三种常见形式对比 $\rightarrow$ 核心数学性质 $\rightarrow$ 谱图理论与图频域 $\rightarrow$ PyTorch 核心实现”** 进行全景拆解。

### 一、 物理直观：什么是“拉普拉斯”？

在连续物理学中，**拉普拉斯算子 $\Delta f = \nabla^2 f$** 描述的是**某种物理量在空间的“扩散”或“差异”程度**。

- **热传导方程**：$\frac{\partial u}{\partial t} = \alpha \Delta u$（热量从高温度点向四周低温度点扩散）。
    
- **物理意义**：某一点的拉普拉斯值，等于**该点的当前值与周围邻居平均值之间的差值**。
    

把这个物理直观搬到**离散的图结构 $G=(V, E)$** 上：

- 假设图中每个节点 $i$ 上都有一个信号值/特征值 $f_i$（例如：节点的温度、社交用户的兴趣偏好）。
    
- 节点 $i$ 与周围邻居节点的**差异总和**，就是图上的拉普拉斯作用。
    

### 二、 从微积分推导图拉普拉斯矩阵

假设图 $G$ 有 $N$ 个节点，邻接矩阵为 $A \in \mathbb{R}^{N \times N}$，度矩阵为对角阵 $D \in \mathbb{R}^{N \times N}$（其中 $D_{ii} = \sum_j A_{ij}$）。

#### 1. 计算单个节点上的拉普拉斯算子

对于节点 $i$，我们要衡量它的信号值 $f_i$ 与所有邻居节点信号值 $f_j$ 的**加权差值和**：

$$(\Delta f)_i = \sum_{j \in \mathcal{N}(i)} A_{ij} (f_i - f_j)$$

我们把这个式子展开拆项：

$$(\Delta f)_i = f_i \sum_{j \in \mathcal{N}(i)} A_{ij} - \sum_{j \in \mathcal{N}(i)} A_{ij} f_j$$

- 注意到 $\sum_{j \in \mathcal{N}(i)} A_{ij} = D_{ii}$（即节点 $i$ 的度）。
    
- 所以上式可以写为：
    
    $$(\Delta f)_i = D_{ii} f_i - \sum_{j=1}^N A_{ij} f_j$$
    

#### 2. 写成矩阵向量形式

将所有 $N$ 个节点的公式拼成一个向量 $\Delta f$（其中 $f = [f_1, f_2, \dots, f_N]^T$）：

$$\Delta f = D f - A f = (D - A) f$$

这里提取出来的矩阵 **$L = D - A$**，就是最基础的 **非正则化图拉普拉斯矩阵（Unnormalized Graph Laplacian）**！

### 三、 三种常见的图拉普拉斯矩阵形式

在实际科研与工程中，为了消除节点“度数大小”对特征更新的量级影响（例如避免度数极大的“明星节点”掩盖其他节点），通常会对 $L$ 进行标准化归一化。

|**拉普拉斯矩阵类型**|**数学定义公式**|**矩阵元素表达 Lij​**|**适用场景与代表模型**|
|---|---|---|---|
|**非归一化 (Unnormalized)**|$L = D - A$|$L_{ij} = \begin{cases} D_{ii} & i = j \\ -A_{ij} & i \neq j \end{cases}$|图聚类（Spectral Clustering）、经典图论|
|**对称归一化 (Symmetric Normalized)**|$L_{sym} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$|$(L_{sym})_{ij} = \begin{cases} 1 & i = j, D_{ii} > 0 \\ -\frac{A_{ij}}{\sqrt{D_{ii} D_{jj}}} & i \neq j \text{ 且有边} \end{cases}$|**GCN (Kipf & Welling)**、图信号处理（GSP）|
|**随机游走归一化 (Random Walk Normalized)**|$L_{rw} = D^{-1} L = I - D^{-1} A$|$(L_{rw})_{ij} = \begin{cases} 1 & i = j \\ -\frac{A_{ij}}{D_{ii}} & i \neq j \text{ 且有边} \end{cases}$|图上的马尔可夫随机游走、PageRank|

### 四、 图拉普拉斯矩阵的核心数学性质

图拉普拉斯矩阵（特别是无向图的 $L$ 和 $L_{sym}$）拥有极佳的代数性质：

#### 1. 半正定性（Positive Semi-Definiteness）

对于任意信号向量 $f \in \mathbb{R}^N$，有二次型：

$$f^T L f = f^T (D - A) f = \frac{1}{2} \sum_{i,j=1}^N A_{ij} (f_i - f_j)^2 \geq 0$$

- **物理意义**：$f^T L f$ 被称为图上的**平滑度（Smoothness）**或**狄利克雷能量（Dirichlet Energy）**。如果相连节点的信号值 $f_i$ 与 $f_j$ 很接近，则能量很小；如果相邻节点的信号突变很大，则能量很高。
    

#### 2. 特征值全为非负数

由于 $L$ 是实对称半正定矩阵，其所有特征值满足：

$$0 = \lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_N$$

#### 3. 最小特征值 $\lambda_1 = 0$ 及其特征向量

- $L \mathbf{1} = (D - A) \mathbf{1} = \mathbf{0}$（其中 $\mathbf{1}$ 为全 1 向量）。
    
- 因此，最小特征值永远为 $\lambda_1 = 0$，对应的特征向量为常数向量 $\mathbf{1}$。
    

#### 4. 零特征值的重数与图连通块（Connected Components）

- 矩阵 $L$ 的特征值 $0$ 的**重数（即 $0$ 出现了几次），精确等于该图中的连通子图（连通块）的数量**！
    
- **代数连通度（Fiedler 值）**：第二小的特征值 $\lambda_2$ 称为 **Fiedler Value**。$\lambda_2$ 越大，说明图越难被切割成两个孤立部分；在谱聚类（Spectral Clustering）中，$\lambda_2$ 对应的特征向量（Fiedler Vector）被用来划分图。
    

### 五、 谱图理论（Spectral Graph Theory）与图频域分析

为什么在图卷积网络（GCN）中一定要提到图拉普拉斯矩阵？因为 **$L$ 是从图的时域（节点域）走向频域（谱域）的唯一桥梁！**

#### 1. 经典傅里叶变换 vs. 图傅里叶变换（Graph Fourier Transform, GFT）

在传统信号处理中，傅里叶变换的基函数是 $e^{-i\omega t}$，它们其实是连续拉普拉斯算子 $\Delta$ 的**特征函数**（即 $\Delta e^{-i\omega t} = -\omega^2 e^{-i\omega t}$）。

同理，在图上：

- 对实对称矩阵 $L$ 进行正交对角化分解：
    
    $$L = U \Lambda U^T = U \begin{bmatrix} \lambda_1 & & \\ & \ddots & \\ & & \lambda_N \end{bmatrix} U^T$$
    
    - **特征值 $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$**：扮演频率（Frequencies）的角色。$\lambda$ 越大，代表图信号变化越剧烈的高频信号。
        
    - **特征向量矩阵 $U = [u_1, u_2, \dots, u_N]$**：正交基矩阵，扮演傅里叶变换基底（Fourier Bases）的角色。
        

#### 2. 图傅里叶变换正反向公式

- **图傅里叶变换（时域 $\rightarrow$ 频域）**：
    
    $$\hat{f} = U^T f$$
    
- **图傅里叶逆变换（频域 $\rightarrow$ 时域）**：
    
    $$f = U \hat{f}$$
    

#### 3. 谱域图卷积（Spectral Graph Convolution）

有了傅里叶变换，根据**卷积定理（时域卷积等于频域相乘）**，两个图信号 $f$ 与滤波器 $g$ 的图卷积就可以定义为：

$$f \star_G g = U \left( (U^T g) \odot (U^T f) \right) = U g_\theta(\Lambda) U^T f$$

这也是 Kipf 提出的经典 **GCN（Graph Convolutional Network）** 的理论起源！

### 六、 PyTorch 核心实现

下面展示使用 PyTorch 从邻接矩阵计算三种 Graph Laplacian 的实现：

Python

```
import torch

def compute_graph_laplacian(A, laplacian_type='sym'):
    """
    根据邻接矩阵 A 计算图拉普拉斯矩阵
    :param A: 邻接矩阵 Tensor, 形状 (N, N)
    :param laplacian_type: 'unnormalized', 'sym', 'rw'
    :return: 拉普拉斯矩阵 L
    """
    N = A.size(0)
    
    # 1. 计算度矩阵 D 的对角元素 (Degree)
    deg = torch.sum(A, dim=1)
    
    if laplacian_type == 'unnormalized':
        # L = D - A
        D = torch.diag(deg)
        L = D - A
        
    elif laplacian_type == 'sym':
        # L_sym = I - D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0 # 处理孤立节点 deg=0 的情况
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        I = torch.eye(N, device=A.device)
        L = I - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
        
    elif laplacian_type == 'rw':
        # L_rw = I - D^(-1) * A
        deg_inv = torch.pow(deg, -1.0)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        D_inv = torch.diag(deg_inv)
        
        I = torch.eye(N, device=A.device)
        L = I - torch.mm(D_inv, A)
        
    else:
        raise ValueError(f"Unknown laplacian type: {laplacian_type}")
        
    return L

# --- 验证代码 ---
if __name__ == "__main__":
    # 模拟一个 4 节点的无向图: 0-1, 1-2, 2-3 (一条链)
    A = torch.tensor([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.float32)

    L_un = compute_graph_laplacian(A, 'unnormalized')
    L_sym = compute_graph_laplacian(A, 'sym')

    print("非归一化图拉普拉斯矩阵 L:\n", L_un)
    print("\n对称归一化图拉普拉斯矩阵 L_sym:\n", L_sym)

    # 计算特征值与特征向量
    eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
    print("\nL_sym 特征值 (图频率):\n", eigenvalues)
```

### 七、 总结

1. **物理直观**：拉普拉斯矩阵刻画了图中信号在节点间的**差异与流动**。
    
2. **代数本质**：$L = D - A$，将拓扑结构编码为实对称半正定矩阵。
    
3. **频域桥梁**：$L$ 的特征值是图的“频率”，特征向量是图的“傅里叶基底”，支撑起了谱域图神经网络（GCN、ChebNet）的完整逻辑。

---

## 笔记联系

- [[10_基础与算法/数学基础/Laplacian - 拉普拉斯算子|Laplacian - 拉普拉斯算子]]：回顾拉普拉斯算子的连续直觉。
- [[10_基础与算法/经典机器学习/LLE - 局部线性嵌入|LLE - 局部线性嵌入]]：比较图结构、邻域关系与低维表示。
- [[20_深度学习/网络结构/DSC - 深度子空间聚类|DSC - 深度子空间聚类]]：联系图结构与深度聚类中的样本关系。
