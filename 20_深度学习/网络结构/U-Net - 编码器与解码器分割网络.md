---
title: "U-Net - 编码器与解码器分割网络"
aliases:
  - "U-Net"
分类: "网络结构"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[20_深度学习/网络结构/索引 - 网络结构|网络结构索引]]

# U-Net - 编码器与解码器分割网络


**U-Net** 是深度学习在医学图像分割（Medical Image Segmentation）乃至通用语义分割领域的里程碑式架构。它由 Olaf Ronneberger 等人于 2015 年在 MICCAI 会议上提出。

U-Net 最经典的地方在于其对称的 **“U 型结构”** 和强大的 **“跳跃连接（Skip Connections）”**。相比于 FCN，U-Net 能在极少训练样本（甚至只有几十张图）的情况下，实现像素级的高精度边缘分割。

为了让你彻底搞懂，我们将从 **“设计背景 $\rightarrow$ 架构全景解析 $\rightarrow$ 核心机制剖析 $\rightarrow$ 损失函数与训练技巧 $\rightarrow$ PyTorch 代码实现”** 逐步拆解。

### 一、 U-Net 的设计背景：为什么医学图像需要它？

在 U-Net 出现之前，将深度学习用于医学图像处理面临两大痛点：

1. **标注数据极度稀缺**：医学图像（如 MRI、CT、病理切片）需要专业医生手动逐像素标注，数据量通常只有几十张，难以训练深层网络。
    
2. **边缘精度要求极高**：器官、肿瘤或细胞的边界识别必须非常精准，而传统的 FCN 等网络下采样后再直接上采样，恢复出的边缘往往十分模糊。
    

**U-Net 的解决方案**：通过构建一个高度对称的“特征提取（编码）- 特征恢复（解码）”网络，并在同层级之间加入**通道拼接（Concat）形式的跳跃连接**，实现了低级位置细节与高级语义信息的完美融合。

### 二、 U-Net 架构全景剖析

U-Net 的网络结构因形状酷似字母 **“U”** 而得名，主要由三部分组成：

Plaintext

```
输入图像 (1x572x572)                                                        输出分割图 (2x388x388)
   │                                                                               ▲
[Conv 3x3]                                                                    [Conv 1x1]
   │                                                                               │
(64, 568, 568) ─── Skip Connection (Crop & Concatenate) ───> (128, 388, 388) ──> [Conv 3x3]
   │                                                                               ▲
[MaxPool 2x2]                                                                 [Up-Conv 2x2]
   ▼                                                                               │
(128, 280, 280) ── Skip Connection (Crop & Concatenate) ───> (256, 198, 198) ──> [Conv 3x3]
   │                                                                               ▲
[MaxPool 2x2]                                                                 [Up-Conv 2x2]
   ▼                                                                               │
(256, 136, 136) ── Skip Connection (Crop & Concatenate) ───> (512, 100, 100) ──> [Conv 3x3]
   │                                                                               ▲
[MaxPool 2x2]                                                                 [Up-Conv 2x2]
   ▼                                                                               │
(512, 64, 64) ──── Skip Connection (Crop & Concatenate) ───> (1024, 52, 52) ───> [Conv 3x3]
   │                                                                               ▲
[MaxPool 2x2]                                                                 [Up-Conv 2x2]
   ▼                                                                               │
(1024, 30, 30) ───────────────── [Conv 3x3, ReLU] ─────────────────────────────────┘
                                   (Bottleneck 瓶颈层)
```

#### 1. 左侧：收缩路径（Contracting Path / Encoder）

- **作用**：提取图像的上文特征（Context），捕获“图像里有什么”。
    
- **结构**：由 4 个连续的下采样模块组成。
    
    - 每个模块包含：**两次 $3 \times 3$ 卷积 + ReLU 激活** $\rightarrow$ **一次 $2 \times 2$ Max Pooling（步长为 2）**。
        
    - 每经过一次 Pooling，特征图的尺寸（高、宽）**减半**，而通道数（Channel）**翻倍**（64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512 $\rightarrow$ 1024）。
        

#### 2. 底部：瓶颈层（Bottleneck）

连接编码器与解码器的最深层，包含两次 $3 \times 3$ 卷积，此时特征图尺寸最小、抽象语义程度最高。

#### 3. 右侧：扩展路径（Expanding Path / Decoder）

- **作用**：将抽象特征还原回原始分辨率，实现精细定位（Localization），回答“目标在哪里”。
    
- **结构**：与左侧对称，由 4 个连续的上采样模块组成。
    
    - 每个模块包含：**一次 $2 \times 2$ 转置卷积（上采样，尺寸翻倍，通道减半）** $\rightarrow$ **与左侧对应层特征图拼接** $\rightarrow$ **两次 $3 \times 3$ 卷积 + ReLU**。
        

#### 4. 核心关键：跳跃连接（Skip Connections）

- **与 FCN 的区别**：
    
    - FCN 的跳跃连接采用的是**按元素相加（Element-wise Add）**；
        
    - U-Net 的跳跃连接采用的是**通道维度拼接（Channel Concatenation）**！
        
- **物理意义**：将 Encoder 留存的**高分辨率低级特征（边缘、纹理、位置）**直接输送给 Decoder，与 Decoder 的**高级语义特征**强行拼接在一起。这样在做像素分类时，既知道了物体的类别，又掌握了精准的边缘轮廓。
    

### 三、 核心细节剖析

#### 1. 为什么原版 U-Net 输入与输出尺寸不一致？（Valid Padding）

观察原论文图纸会发现，输入是 $572 \times 572$，输出却是 $388 \times 388$。

- **原因**：原版 U-Net 采用了 **`padding=0`（Valid Convolution）**，每次 $3 \times 3$ 卷积都会让特征图的高宽减少 2 个像素。因此在跳跃连接拼接时，左侧的特征图比右侧大，需要先对左侧特征图进行裁剪（Crop）再拼接。
    
- **现代改进（Same Padding）**：在现代工业界和代码实现中，大家普遍将卷积设为 `padding=1`（Same Convolution），使得卷积前后高宽不变，**输入与输出尺寸保持完全一致**（如 $512 \times 512 \rightarrow 512 \times 512$），极大地简化了代码实现。
    

#### 2. 重叠切片策略（Overlap-tile Strategy）

针对极其庞大的医学图像（如 $2000 \times 2000$ 的超大切片），GPU 显存无法一次性放下。U-Net 采用切片预测，并在图像边缘使用镜像无缝拼接（Mirroring Extension）填充补齐，保证了切片边缘像素的预测准确度。

#### 3. 强数据增强（Data Augmentation）

由于医学图像极其匮乏，U-Net 论文中大量使用了**随机弹性形变（Random Elastic Deformations）**、平移、旋转及灰度变化。通过模拟生物组织（细胞、器官）的真实形变，极大地提升了模型的泛化能力。

### 四、 损失函数与加权交叉熵

医学图像分割常面临严重的分隔难题（如两个相邻的细胞边缘极易粘连在一起）。

原论文提出了**带权重的像素级交叉熵损失（Weighted Cross-Entropy Loss）**：

$$L = -\sum_{x \in \Omega} w(x) \log(p_k(x))$$

其中权重 $w(x)$ 的计算公式引入了对**细胞间隙边缘**的重罚：

$$w(x) = w_c(x) + w_0 \cdot \exp\left( -\frac{(d_1(x) + d_2(x))^2}{2\sigma^2} \right)$$

- $w_c(x)$：用于平衡前景与背景类别不均衡的权重。
    
- $d_1(x)$：当前像素到**最近**物体边界的距离。
    
- $d_2(x)$：当前像素到**第二近**物体边界的距离。
    

> **效果**：两个物体靠得越近，$(d_1 + d_2)$ 越小，该区域像素的权重 $w(x)$ 就会呈指数级暴涨！这强迫网络极其严厉地学习**紧密相连的边界区域**，完美解决了边缘粘连问题。

### 五、 PyTorch 现代通用代码实现（Same Padding 版）

以下是目前最通用的 **现代 U-Net** 实现（保持输入输出尺寸一致，直接可用于项目）：

Python

```
import torch
import torch.nn as nn

# 1. 基础双卷积块 (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# 2. U-Net 完整网络架构
class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(UNet, self).__init__()
        
        # Encoder (下采样路径)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024)) # Bottleneck
        
        # Decoder (上采样路径)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512) # 512(来自Up) + 512(来自Skip) = 1024
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)  # 256 + 256 = 512
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)  # 128 + 128 = 256
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)   # 64 + 64 = 128
        
        # 输出卷积 (1x1 卷积映射到类别数)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder 前向传播并保存跳跃连接特征
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder 前向传播，并在通道维度拼接 (torch.cat)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1) # 拼接 Skip Connection
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

# 测试网络
if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=1) # 3通道RGB输入，单类别分割
    x = torch.randn(2, 3, 256, 256)            # Batch Size=2, 256x256 图像
    output = model(x)
    print("输入尺寸:", x.shape)
    print("输出分割图尺寸:", output.shape)      # 应为 (2, 1, 256, 256)
```

### 六、 U-Net 的后续演进家族

由于 U-Net 结构的巨大成功，后续诞生了一系列改进架构：

1. **3D U-Net**：将 2D 卷积扩展为 3D 卷积，直接处理 CT / MRI 等三维立体的医学体数据。
    
2. **Res-UNet**：在编码器中引入残差连接（Residual Blocks），缓解深层网络训练困难问题。
    
3. **U-Net++**：重新设计了 Skip Connections，引入了嵌套与密集跳跃连接（Nested Skip Pathways），缩短 Encoder 与 Decoder 之间的语义差距。
    
4. **Swin-UNet**：将 Transformer（Swin Transformer）作为骨干替代传统 CNN，更好地捕获全局长距离依赖。

---

## 笔记联系

- [[20_深度学习/网络结构/FCN - 全卷积网络|FCN - 全卷积网络]]：比较两种语义分割网络的结构设计。
- [[20_深度学习/网络结构/采样 - 上采样与下采样|采样 - 上采样与下采样]]：结合下采样与上采样理解 U 形结构。
- [[20_深度学习/网络结构/论文 - U-Net：Convolutional Networks for Biomedical Image Segmentation|论文 - U-Net：Convolutional Networks for Biomedical Image Segmentation]]：从网络讲解回到原论文阅读。
- [[30_对抗机器学习/ViT与基础模型攻击/SAM Attack - 分割基础模型攻击发展历程|SAM Attack - 分割基础模型攻击发展历程]]：结合分割网络背景比较攻击目标与输出结构。
