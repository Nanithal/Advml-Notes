---
title: "DSC - 深度子空间聚类"
aliases:
  - "DSC：Deep Subspace Clustering"
分类: "网络结构"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/网络结构"
状态: "已整理"
摘要: "围绕“DSC - 深度子空间聚类”，整理DSC 的整体网络架构全景、编码器（Encoder, $E_\\theta$）、自表达层（Self-Expressive Layer）。"
更新日期: "2026-10-30"
上级目录: "[[20_深度学习/网络结构/总结 - 网络结构]]"
相关笔记:
  - "[[10_基础与算法/数学基础/Graph Laplacian - 图拉普拉斯矩阵]]"
  - "[[20_深度学习/网络结构/AE - 自编码器]]"
  - "[[20_深度学习/网络结构/DEC - 深度嵌入聚类]]"
论文条目:
  - "[[论文来源目录#dsc]]"
论文标题:
  - "Deep Subspace Clustering Networks"
发表载体:
  - "NIPS"
发表年份:
  - "2017"
发表类型:
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper/6608-deep-subspace-clustering-networks"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[20_深度学习/网络结构/总结 - 网络结构|网络结构总结]]

# DSC - 深度子空间聚类

从**传统的线性自表达**跨越到**深度子空间聚类（Deep Subspace Clustering, DSC）**，是该领域从“理论玩具”走向“工业实用”的关键一步。

前面我们提到，传统子空间聚类（如 SSC、LRR）的核心假设是：数据分布在**多个线性子空间的并集**上。然而，现实中的真实图像、高光谱数据或语音特征，往往分布在高度弯曲的非线性流形（Nonlinear Manifolds）上。如果强行用线性公式 $X = XC$ 去拟合，会导致严重的重构误差和错聚。

**深度子空间聚类（DSC）**的核心解决方案是：**“用深度神经网络（如卷积自编码器）将非线性数据映射到平坦的隐空间，在隐空间中嵌入一个线性自表达层，并实现端到端的联合优化。”**

为了让你彻底通透，我们将从 **“网络架构设计 $\rightarrow$ 核心数学损失函数 $\rightarrow$ 训练机制与关键细节 $\rightarrow$ PyTorch 核心代码实现 $\rightarrow$ 挑战与演进方向”** 逐步深度拆解。

### 一、 DSC 的整体网络架构全景

DSC 最经典的架构由三种核心组件拼接而成：

Plaintext

```
  原始数据 X                    隐空间特征 Z                    重建隐空间 Z'                 重构数据 X'
(高度非线性流形)               (低维线性子空间)               (满足 Z' = ZC)               (恢复原图)
 ┌───────────┐                ┌───────────┐                ┌───────────┐                ┌───────────┐
 │           │ ─── Encoder ──►│           │ ──► Self- ──────►│           │ ─── Decoder ──►│           │
 │  [Image]  │    E_θ(X)      │  Latent   │     Expressive   │ Reconstructed│   D_ϕ(Z')    │Reconstructed│
 │           │                │  Vector Z │     Layer (C)    │  Latent Z'│                │   Image   │
 └───────────┘                └───────────┘                └───────────┘                └───────────┘
       ▲                                                                                       │
       └───────────────────────────── 重构损失 L_recon ─────────────────────────────────────────┘
```

#### 1. 编码器（Encoder, $E_\theta$）

- **作用**：扮演“流形解卷（Unfolding）”的角色。
    
- **过程**：将非线性分布的原始输入 $X \in \mathbb{R}^{D \times N}$ 映射为线性可分的低维隐空间特征 $Z = E_\theta(X) \in \mathbb{R}^{d \times N}$。
    

#### 2. 自表达层（Self-Expressive Layer）

- **作用**：DSC 最精妙的核心模块，本质上是一个**没有偏置项（Bias-free）的全连接层**，其权重矩阵正好就是子空间自表示系数矩阵 $C \in \mathbb{R}^{N \times N}$。
    
- **过程**：输入矩阵 $Z$，输出重构后的隐向量矩阵 $Z' = Z C$。
    
- **约束**：强制要求矩阵 $C$ 的对角线为 0（即 $\text{diag}(C) = \mathbf{0}$），防止神经元自己复制自己。
    

#### 3. 解码器（Decoder, $D_\phi$）

- **作用**：将自表达层重构出的隐向量 $Z' = ZC$ 重新还原回原始数据空间，得到重建样本 $X' = D_\phi(Z C)$。
    

### 二、 核心数学建模与联合损失函数

DSC 之所以强大，是因为它打破了传统“先提取特征、后做自表达”的两阶段分离痛点，将特征提取与自表达优化放入同一个损失函数中**联合训练（Joint Training）**。

其整体损失函数由三部分组成：

$$\mathcal{L}_{DSC}(\theta, \phi, C) = \mathcal{L}_{\text{recon}} + \gamma_1 \mathcal{L}_{\text{express}} + \gamma_2 \mathcal{L}_{\text{reg}}$$

#### 1. 图像重构损失（Reconstruction Loss, $\mathcal{L}_{\text{recon}}$）

保证编码器和解码器不会丢弃图像的关键语义信息：

$$\mathcal{L}_{\text{recon}} = \frac{1}{2} \Vert{}X - D_\phi(Z C)\Vert{}_F^2 \quad \text{或} \quad \frac{1}{2} \Vert{}X - D_\phi(E_\theta(X) C)\Vert{}_F^2$$

> **关键物理意义**：注意看！这里解码器的输入是经过自表达层乘以 $C$ 之后的 $ZC$，而不是原始的 $Z$！这意味着，自表示系数 $C$ 的质量直接决定了 Decoder 能否还原出干净的图片，强制促使 $Z$ 拟合出良好的线性子空间。

#### 2. 隐空间自表达损失（Self-Expressive Loss, $\mathcal{L}_{\text{express}}$）

约束隐特征向量矩阵 $Z$ 必须严格满足线性组合关系：

$$\mathcal{L}_{\text{express}} = \frac{1}{2} \Vert{}Z - Z C\Vert{}_F^2 = \frac{1}{2} \Vert{}E_\theta(X) - E_\theta(X)C\Vert{}_F^2$$

#### 3. 系数矩阵正则化项（Regularization Loss, $\mathcal{L}_{\text{reg}}$）

为了防止矩阵 $C$ 出现平凡解，并促使其产生块对角（Block-Diagonal）形态，通常对 $C$ 加上 $L_1$ 范数（稀疏）或 $L_2$ 范数（平滑）：

$$\mathcal{L}_{\text{reg}} = \Vert{}C\Vert{}_p \quad (p=1 \text{ 即 } \text{DSC-L1}, \quad p=2 \text{ 即 } \text{DSC-L2})$$

- **带有约束**：$\text{diag}(C) = \mathbf{0}$。
    

### 三、 DSC 的训练机制与工程细节

如果你直接随机初始化一个 DSC 模型并开始梯度下降，**训练极其容易崩溃（发散或收敛到平凡解）**。在工程落地中，必须遵循以下两阶段训练策略：

#### 1. 第一阶段：自编码器预训练（Pre-training）

- 先剔除自表达层 $C$，只训练标准的自编码器（Encoder + Decoder），使 $\mathcal{L} = \Vert{}X - D_\phi(E_\theta(X))\Vert{}_F^2$ 收敛。
    
- **目的**：让神经网络先具备基本的图像降维与重构能力，提供一个良好的初始隐空间 $Z_0$。
    

#### 2. 第二阶段：端到端联合微调（Joint Fine-tuning）

- 插入自表达层 $C$（$C$ 矩阵通常初始化为全 0 或小随机数）。
    
- 同时将参数 $(\theta, \phi, C)$ 放入优化器中，按照综合损失函数 $\mathcal{L}_{DSC}$ 进行梯度更新。
    

#### 3. 第三阶段：后处理与谱聚类（Spectral Clustering）

- 训练完成后，提取收敛后的系数矩阵 $C$。
    
- 构造对称相似度矩阵：$W = \frac{\vert{}C\vert{} + \vert{}C\vert{}^T}{2}$。
    
- 将 $W$ 输入到谱聚类算法中，最终输出每个样本的类别标签。
    

### 四、 PyTorch 核心实现（以 DSC-L2 为例）

下面展示一个可以直接运行的深度子空间聚类（DSC）核心模块的 PyTorch 代码实现：

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 自表达层 (Self-Expressive Layer)
class SelfExpressive(nn.Module):
    def __init__(self, num_samples):
        super(SelfExpressive, self).__init__()
        # C 矩阵维度为 (N, N)，其中 N 为 Batch 中的样本数（或全量数据集样本数）
        self.Coefficient = nn.Parameter(1e-8 * torch.ones(num_samples, num_samples, dtype=torch.float32))

    def forward(self, z):
        # z 形状: (d, N)，代表 N 个 d 维隐向量
        # 强制将对角线元素置零
        C = self.Coefficient - torch.diag(torch.diag(self.Coefficient))
        
        # 隐空间自表达 Z' = Z * C
        z_recon = torch.matmul(z, C)
        return z_recon, C

# 2. 完整 DSC 网络结构
class DeepSubspaceClustering(nn.Module):
    def __init__(self, num_samples, latent_dim=64):
        super(DeepSubspaceClustering, self).__init__()
        self.num_samples = num_samples

        # 编码器: 图像 -> 隐向量
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim)                      # (B, d)
        )

        # 自表达层
        self.self_expressive = SelfExpressive(num_samples)

        # 解码器: 隐向量 -> 重建图像
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x 形状: (N, C, H, W)
        z = self.encoder(x)          # (N, d)
        z_t = z.T                    # (d, N)

        # 隐空间自表示
        z_recon_t, C = self.self_expressive(z_t)
        z_recon = z_recon_t.T        # (N, d)

        # 重建原始图片
        x_recon = self.decoder(z_recon)

        return x_recon, z, z_recon, C

# --- 训练与损失计算逻辑示范 ---
def loss_function(x, x_recon, z, z_recon, C, gamma1=1.0, gamma2=1.0):
    # 1. 图像重构 Loss
    loss_recon = F.mse_loss(x_recon, x, reduction='sum')
    
    # 2. 隐空间自表达 Loss
    loss_express = F.mse_loss(z_recon, z, reduction='sum')
    
    # 3. C 矩阵正则化 Loss (L2 范数)
    loss_reg = torch.sum(C ** 2)

    total_loss = loss_recon + gamma1 * loss_express + gamma2 * loss_reg
    return total_loss

if __name__ == "__main__":
    N = 100 # 假设一次性传入 100 张图片做全量子空间聚类
    x = torch.randn(N, 1, 28, 28)

    model = DeepSubspaceClustering(num_samples=N, latent_dim=32)
    x_recon, z, z_recon, C = model(x)

    loss = loss_function(x, x_recon, z, z_recon, C)
    print("DSC 模型前向传播成功！")
    print("自表示矩阵 C 尺寸:", C.shape) # (100, 100)
    print("总 Loss:", loss.item())
```

### 五、 传统 SSC vs. 深度子空间聚类（DSC）对比

|**维度**|**传统 SSC / LRR**|**深度子空间聚类（DSC）**|
|---|---|---|
|**空间假设**|假设数据分布在**线性**子空间|假设数据分布在**非线性**流形/低维子空间|
|**特征提取能力**|纯线性，依赖原始特征质量|强（利用 CNN / Transformer 自动提取高级语义特征）|
|**建模方式**|两阶段（提取特征与构建图分离）|**端到端联合优化（Joint Optimization）**|
|**算法鲁棒性**|对光照、旋转、复杂背景极度敏感|通过卷积核提取不变性特征，对噪声极度鲁棒|
|**优化求解器**|ADMM（交替方向乘子法，凸优化）|梯度下降 (Adam/SGD + 谱聚类)|

### 六、 DSC 的局限性与前沿演进方向

虽然 DSC 在人脸聚类（YaleB）、物体识别（COIL20）等数据集上拿下了 SOTA 的性能，但它依然存在一些固有的局限性：

1. **可扩展性（Scalability）问题（最大痛点）**：
    
    自表达矩阵 $C$ 的尺寸为 $N \times N$（$N$ 为总样本数）。当数据集包含几十万或上百万张图片时，矩阵 $C$ 将会爆炸（显存无法容纳），导致 DSC **无法直接应用于大 Mini-batch 训练**。
    
    - **演进**：后续提出了 **k-Subspace DSC** 或 **Sampling-based DSC**，通过采样一小部分代表性样本（Anchors）来构建 $N \times m$ 的子空间图，降低空间复杂度。
        
2. **对图拓扑与多观角的拓展**：
    
    - **Multiview DSC**：扩展到多视角学习（如结合图像的 RGB、深度图、边缘特征共同构建多视角子空间）。
        
    - **Self-Supervised DSC**：结合对比学习（Contrastive Learning），在没有标签的情况下进一步约束隐空间 $Z$ 的表征质量。

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/数学基础/Graph Laplacian - 图拉普拉斯矩阵|Graph Laplacian - 图拉普拉斯矩阵]]：联系图结构与深度聚类中的样本关系。
- [[20_深度学习/网络结构/AE - 自编码器|AE - 自编码器]]：回顾编码器与解码器的基本结构。
- [[20_深度学习/网络结构/DEC - 深度嵌入聚类|DEC - 深度嵌入聚类]]：比较两种深度聚类建模方式。

## 论文出处

- [Deep Subspace Clustering Networks](https://papers.nips.cc/paper/6608-deep-subspace-clustering-networks)：NIPS，2017，会议论文。

<!-- knowledge-links:end -->
