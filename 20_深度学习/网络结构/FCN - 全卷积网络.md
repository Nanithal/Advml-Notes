---
title: "FCN - 全卷积网络"
aliases:
  - "FCN全连接卷积网络"
分类: "网络结构"
笔记类型: "概念笔记"
整理日期: 2026-10-27
---

[[首页|首页]] · [[20_深度学习/网络结构/索引 - 网络结构|网络结构索引]]

# FCN - 全卷积网络

全卷积网络（Fully Convolutional Network, **FCN**）是深度学习在图像语义分割（Semantic Segmentation）领域的开创性工作（CVPR 2015 最佳论文候选）。

在 FCN 诞生之前，传统的卷积神经网络（如 AlexNet、VGG）主要用于**图像分类**任务，其输出是整张图像的类别标签。而 FCN 的目标是实现**像素级（Pixel-level）预测**——即对输入图像的每一个像素都打上类别标签（例如识别出哪些像素是“人”、哪些是“车”、哪些是“背景”）。

为了彻底学会 FCN，我们将从 **“核心动机 $\rightarrow$ 三大核心技术创新 $\rightarrow$ 网络架构与跳跃连接（Skip Architecture） $\rightarrow$ 损失函数与训练细节 $\rightarrow$ PyTorch 代码实现”** 逐步深度剖析。

### 一、 核心动机：为什么传统 CNN 无法直接做语义分割？

传统的分类网络（如 VGG16）结构为：

`输入图像 (H × W × 3) → 卷积层与池化层提取特征 → 全连接层 (FC Layer) → 分类概率`

分类网络存在两个致命限制：

1. **全连接层限制了固定的输入尺寸**：全连接层的权重矩阵形状是固定的，因此输入图像的尺寸必须完全一致（如 $224 \times 224$）。
    
2. **丢失了空间位置信息**：全连接层会把二维/三维的特征图拉平（Flatten）成一维向量，这彻底破坏了像素与像素之间的二维空间拓扑结构。
    

**FCN 的核心思想**：**把传统 CNN 末尾的全连接层全部替换为卷积层！** 使得网络能够接收**任意尺寸**的图像，并输出与输入图像同等空间大小的**像素级分割图**。

### 二、 FCN 的三大核心技术创新

#### 1. 全卷积化（Convolutionalization）

如何把全连接层（FC Layer）改写为卷积层？

假设全连接层的输入特征图形状为 $h \times w \times c$（例如 $7 \times 7 \times 512$），节点数为 $K$（如 $4096$）：

- **原全连接层**：将 $7 \times 7 \times 512$ 拉平为长 $25088$ 的向量，乘以形状为 $(25088, 4096)$ 的权重矩阵。
    
- **卷积化等价替换**：改用 $K$ 个卷积核大小为 $h \times w$（即 $7 \times 7$）、通道数为 $c$ 的卷积层。
    
    - 输出特征图形状变为 $1 \times 1 \times K$（即 $1 \times 1 \times 4096$）。
        
    - **物理意义**：全连接层的计算在数学上完全等价于核大小等于输入特征图大小的卷积运算。
        

**卷积化的巨大优势**：

如果输入的图像尺寸大于 $224 \times 224$（例如 $h' \times w'$ 变大），全卷积网络不会报错，而是会输出形状为 $h_{out} \times w_{out} \times K$ 的二维热力图（Heatmap），天然支持任意尺寸图像输入！

#### 2. 上采样：转置卷积（Transposed Convolution / Deconvolution）

经过多次卷积和下采样（Pooling，通常下采样 32 倍），特征图的尺寸大幅缩小（例如 $512 \times 512$ 的图像缩小为 $16 \times 16$）。

为了获得与原图尺寸（$512 \times 512$）一模一样的像素级预测，网络必须将低分辨率的特征图“放大”还原回去。FCN 引入了**转置卷积（Transposed Convolution，有时被称为反卷积）**。

- **原理**：转置卷积是卷积的逆向操作（但不是数学上的逆矩阵）。通过在特征图元素之间填充零（Padding），并乘以可学习的卷积核参数，将小尺寸特征图放大恢复为大尺寸特征图。
    
- **参数可学习**：不同于双线性插值（Bilinear Interpolation）这种固定的数学规则，转置卷积的权重是通过反向传播自动学习更新的，因此恢复出的图像细节更精准。
    

#### 3. 跳跃连接（Skip Connections / Skip Architecture）

仅仅依靠将缩放了 32 倍的特征图直接做转置卷积恢复原图，分割结果会极其粗糙（边缘呈马赛克状）。

- **矛盾点**：
    
    - **深层特征图**：经过多次下采样，感受野大，包含丰富的高层语义信息（“是什么”），但丢失了细节位置；
        
    - **浅层特征图**：分辨率高，包含丰富的低层边缘、纹理和几何细节信息（“在哪里”），但缺少语义分类能力。
        
- **解决方案**：**融合深层与浅层特征！**
    
    FCN 提出了 **FCN-32s, FCN-16s, FCN-8s** 三种融合策略：
    

Plaintext

```
输入原图 
   │
 [ Conv1 ~ Conv3 ] ───> Pooling3 (下采样8倍, 8x 浅层细节) ─────┐
   │                                                         │
 [ Conv4 ] ───────────> Pooling4 (下采样16倍, 16x 中层特征) ─┐│
   │                                                        ││
 [ Conv5 ] ───────────> Pooling5 (下采样32倍, 32x 深层语义)  ││
                           │                                ││
                      [ Conv6-7 ] (全卷积化)                 ││
                           │                                ││
                           ▼                                ││
                       FCN-32s: 直接 32x 上采样 ────────────┼┼───> 预测图 (最粗糙)
                           │                                ││
                      [ 2x 上采样 ]                         ││
                           │                                ││
                           └───> [ + ] 融合 Pooling4 ───────┼───> FCN-16s: 16x 上采样 (较精细)
                                     │                      │
                                [ 2x 上采样 ]               │
                                     │                      │
                                     └───> [ + ] 融合 Pooling3 ───> FCN-8s: 8x 上采样 (最精细)
```

- **FCN-32s**：直接将 Conv7 的结果转置卷积放大 32 倍还原至原图大小。
    
- **FCN-16s**：将 Conv7 结果放大 2 倍，与 Pooling4 的特征图按元素相加（Element-wise Add），再统一放大 16 倍还原至原图大小。
    
- **FCN-8s**：将 FCN-16s 融合后的特征放大 2 倍，再与 Pooling3 的特征图按元素相加，最后统一放大 8 倍还原至原图大小。
    

> **结论**：**FCN-8s** 结合了最高级的语义与最细致的边缘信息，分割效果最为精准细腻。

### 三、 损失函数与训练细节

#### 1. 像素级交叉熵损失（Pixel-wise Cross-Entropy Loss）

对于输出的 $N$ 通道预测图（$N$ 为类别数，如背景 + 20 类物体 = 21 通道），网络对图像上的**每一个像素**做 Softmax 多分类。

设图像高度为 $H$，宽度为 $W$，类别数为 $C$：

$$L = -\frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{c=1}^{C} y_{i, j, c} \log(\hat{y}_{i, j, c})$$

#### 2. 迁移学习与阶段训练

FCN 通常采用在 ImageNet 上预训练好的骨干网络（如 VGG16 或 ResNet）初始化权重。训练时通常先训练 FCN-32s，收敛后再利用其权重初始化 FCN-16s，最后训练 FCN-8s（逐级精细化）。

### 四、 PyTorch 代码极简实现（以 FCN-8s 为例）

下面演示基于经典 VGG16 骨干网络的 **FCN-8s** 核心架构逻辑：

Python

```
import torch
import torch.nn as nn
from torchvision import models

class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        
        # 1. 加载预训练的 VGG16 骨干
        vgg = models.vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        # 截取 VGG16 的不同阶段
        self.stage1_to_3 = nn.Sequential(*features[:17])  # 直至 Pooling3 (1/8 尺寸)
        self.stage4      = nn.Sequential(*features[17:24]) # 直至 Pooling4 (1/16 尺寸)
        self.stage5      = nn.Sequential(*features[24:])   # 直至 Pooling5 (1/32 尺寸)
        
        # 2. 将 VGG 的全连接层 FC6, FC7 改写为 1x1 卷积层 (全卷积化)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        
        # 3. 映射到类别数的 1x1 卷积
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        
        # 4. 转置卷积（上采样）
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        # 前向传播提取各阶段特征
        pool3 = self.stage1_to_3(x)   # 1/8 尺寸，通道 256
        pool4 = self.stage4(pool3)    # 1/16 尺寸，通道 512
        pool5 = self.stage5(pool4)    # 1/32 尺寸，通道 512
        
        # 深层特征卷积化
        h = self.relu6(self.fc6(pool5))
        h = self.relu7(self.fc7(h))
        score_fr = self.score_fr(h)   # 1/32 尺寸，通道 num_classes
        
        # --- 跳跃连接与融合阶段 ---
        # A. 放大 32x 预测层 2 倍 -> 1/16 尺寸
        upscore2 = self.upscore2(score_fr)
        # B. 融合 Pooling4 映射
        score_pool4 = self.score_pool4(pool4)
        fuse_pool4 = upscore2 + score_pool4
        
        # C. 放大融合特征 2 倍 -> 1/8 尺寸
        upscore_pool4 = self.upscore_pool4(fuse_pool4)
        # D. 融合 Pooling3 映射
        score_pool3 = self.score_pool3(pool3)
        fuse_pool3 = upscore_pool4 + score_pool3
        
        # E. 最终放大 8 倍还原回原图大小 (1/8 -> 1/1)
        out = self.upscore8(fuse_pool3)
        return out

# 测试代码
if __name__ == "__main__":
    model = FCN8s(num_classes=21)
    x = torch.randn(2, 3, 512, 512) # Batch Size=2, 3通道, 512x512图像
    output = model(x)
    print("输入图像尺寸:", x.shape)
    print("输出分割图尺寸:", output.shape) # 应为 (2, 21, 512, 512)
```

### 五、 FCN 的优缺点总结

#### 优点：

1. **开创性**：首次实现了端到端（End-to-End）、像素到像素（Pixel-to-Pixel）的深度学习语义分割。
    
2. **高效性**：通过全卷积化，避免了重叠区域的重复计算，训练和推理速度远超传统的 Patch-based 分割方法。
    
3. **尺寸灵活性**：可接收任意大小的输入图像，无需拉伸变形。
    

#### 局限性与后续改进：

1. **分割边缘依然不够精细**：转置卷积上采样的恢复能力有限，细节依然会有所丢失（促成了后续 **U-Net** 和 **DeepLab** 的诞生）。
    
2. **缺乏多尺度上下文捕获**：对全局上下文信息的利用较弱（促成了 **PSPNet** 空洞空间金字塔池化 ASPP 的引入）。

---

## 笔记联系

- [[20_深度学习/网络结构/CNN - 卷积神经网络|CNN - 卷积神经网络]]：回顾卷积神经网络基础。
- [[20_深度学习/网络结构/论文 - Fully Convolutional Networks for Semantic Segmentation|论文 - Fully Convolutional Networks for Semantic Segmentation]]：从结构讲解回到原论文阅读。
- [[20_深度学习/网络结构/U-Net - 编码器与解码器分割网络|U-Net - 编码器与解码器分割网络]]：比较两种语义分割网络的结构设计。
- [[20_深度学习/网络结构/采样 - 上采样与下采样|采样 - 上采样与下采样]]：结合上采样理解像素级输出的恢复。
