# 论文总结：Adversarial Token Attacks on Vision Transformers

**作者**：Ameya Joshi, Gauri Jagatap, Chinmay Hegde（纽约大学）
**发表**：2022

---

## 1. 研究背景与核心问题

- **Vision Transformer (ViT)** 依赖**图像块（patch）作为 token** 的自注意力机制，与 CNN 的局部卷积有本质区别。
- 现有对抗鲁棒性研究多关注 $\ell_2$ 和 $\ell_\infty$ 扰动（修改全图所有像素），但这类威胁模型**不切实际**。
- 对于 Transformer 而言，更自然的威胁是**攻击部分 token（图像块）**。
- **核心问题**：Transformer 对**部分 token 的扰动**是否鲁棒？与 CNN 相比有何差异？

---

## 2. 攻击方法：基于块稀疏的 Token 攻击

### 2.1 威胁模型
- 输入图像 $\mathbf{x} \in \mathbb{R}^d$ 被划分为 $B$ 个**不重叠块** $[\mathbf{x}_1, \dots, \mathbf{x}_b, \dots, \mathbf{x}_B]$，每块大小为 $p$。
- 攻击者最多可修改 $K \le B$ 个块（**$K$-token 攻击**）。
- 白盒设置：攻击者知道模型架构、参数和梯度。
- 两种扰动预算：
  1. **无约束块稀疏**：每块内可任意修改。
  2. **混合范数块稀疏**：每块内扰动受 $\ell_\infty$ 球约束（半径 $\epsilon$）。

### 2.2 攻击算法（Alg. 1）

**步骤**：

1. **计算块显著性（Patch Saliency）**：
   $$
   S(\mathbf{x}_b) = \sqrt{\sum_{x_i \in \mathbf{x}_b} \left| \frac{\partial L(f(\mathbf{x}, y))}{\partial x_i} \right|^2}
   $$
   即块内所有像素梯度平方和的平方根。

2. **选择 Top-$K$ 块**：根据 $S(\mathbf{x}_b)$ 降序排列，选出最显著的 $K$ 个块。

3. **迭代梯度更新**：
   - 仅对选中的 $K$ 个块进行梯度上升：
     $$
     \mathbf{x}_{b_k} = \mathbf{x}_{b_k} + \nabla_{\mathbf{x}_{b_k}} L
     $$
   - 若使用混合范数约束，则投影到 $\ell_\infty$ 球内：
     $$
     \mathbf{x}_{b_k} = \text{Project}_{\epsilon_\infty}(\mathbf{x}_{b_k})
     $$
   - 重复直到分类标签改变或达到最大迭代次数。

### 2.3 变体
- **稀疏攻击（Sparse Attack）**：块大小 $p=1$，即允许修改 $s = K$ 个像素（标准 $\ell_0$ 攻击）。
- **块 Token 攻击（Patch Token Attack）**：块大小 $p = 16 \times 16$，匹配 ViT 的 token 尺寸。
- **混合范数攻击（Mixed-Norm Attack）**：块稀疏 + 每像素 $\ell_\infty$ 约束（如 $1/255$），确保扰动不可见。

---

## 3. 实验设置

- **数据集**：ImageNet 验证集随机选取 300 张图像。
- **模型（4类架构）**：
  - **Transformer**：ViT-B/16（224 和 384 分辨率）、DeiT、DeiT-Distilled
  - **CNN**：ResNet-50、ResNet-101、WideResNet
  - **MLP**：MLP-Mixer
- **优化**：SGD，步长 0.1（ResNet-101 为 0.2），最多 100 步。

---

## 4. 主要实验结果

### 4.1 混合范数攻击（表1）
固定每像素扰动 $1/255$，改变 token 预算 $K$：

| 模型 | Clean | $K=1$ | $K=2$ | $K=5$ | $K=25$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ViT-224** | 88.70 | 68.77 | 50.83 | 15.28 | — |
| **ViT-384** | 90.03 | 53.48 | 28.57 | 4.98 | — |
| **DeiT** | 85.71 | 72.42 | 46.84 | 6.31 | — |
| **DeiT-Distilled** | 87.70 | 68.77 | 54.15 | 16.61 | — |
| **ResNet-101** | 85.71 | 69.10 | 55.14 | **32.89** | — |
| **ResNet-50** | 85.38 | 67.44 | 55.81 | **31.22** | — |
| **WideResNet** | 87.04 | 54.81 | 32.89 | 11.62 | — |
| **MLP-Mixer** | 83.78 | 63.78 | 37.87 | 5.98 | — |

**结论**：小 token 预算（$K=1,2$）下各模型表现接近；**$K$ 增大后，ResNet 显著比 Transformer 更鲁棒**。

### 4.2 稀疏 vs Token 攻击（表2）
固定稀疏度 $s = 256$ 像素（约 $0.5\%$ 像素）：

| 模型 | Clean | Sparse (256 px) | Patch ($1 \times 16 \times 16$) |
| :--- | :--- | :--- | :--- |
| **ViT-224** | 88.70 | 5.98 | 13.62 |
| **ViT-384** | 90.03 | 3.32 | 1.33 |
| **DeiT** | 85.71 | 4.65 | 17.27 |
| **DeiT-Distilled** | 87.70 | 14.95 | 17.94 |
| **MLP-Mixer** | 83.72 | 5.98 | 26.91 |
| **ResNet-50** | 85.38 | 13.95 | 19.90 |
| **ResNet-101** | 85.71 | **23.59** | **49.50** |
| **WideResNet** | 87.04 | 1.33 | 26.57 |

**结论**：
- **稀疏攻击比 patch 攻击更强**（同样预算下），但稀疏像素分散，**实际不可实现**。
- 同预算下，**ResNet-101 远优于所有 Transformer**（patch 攻击下 $49.50\%$ vs $1.33\% \sim 17.94\%$）。

### 4.3 Token 预算 vs 鲁棒性（表3，表4）
- **表3**：增大 token 预算（$K=1,2,5,10,20,40$），ViT/DeiT 的鲁棒准确率迅速降至 **$0\%$**；ResNet 保持较高（$K=5$ 时 ResNet-101 为 $28.64\%$）。
- **表4**：改变攻击块大小（1,4,8,16），**匹配架构 token 大小（$16 \times 16$）时 Transformer 最脆弱**；小于 token 大小时，Transformer 与 ResNet 表现接近。

### 4.4 可视化（图3，图4）
- 攻击仅修改少数 patch，扰动**不可见**（放大 10 倍才能看到）。
- Transformer 被攻击的 patch 往往集中在**语义关键区域**（如蛇头、花朵中心）。

---

## 5. 关键发现与讨论

1. **Transformer 对 token 攻击更敏感**：匹配 token 尺寸（$16 \times 16$）的 patch 攻击下，ResNet 比 Transformer 鲁棒准确率高 **$\sim 30\%$**。
2. **小 patch 攻击下 Transformer 较鲁棒**：攻击块小于 token 尺寸时，Transformer 与 ResNet 表现相当（可能因注意力机制能补偿局部扰动）。
3. **ResNet 和 MLP-Mixer 一致优于 Transformer**（在 token 攻击下）。
4. **先前研究的局限性**：之前认为 ViT 比 ResNet 更鲁棒的工作（如 Bhojanapalli et al.）仅考虑了 $\ell_2$ 和 $\ell_\infty$ 全图扰动，**忽略了 token 级威胁模型**。
5. **仅需 $1\%$ 像素**即可攻破 Vision Transformer（图1），远少于 $\ell_2$ 和 $\ell_\infty$ 攻击修改全图像素。

---

## 6. 贡献总结

| 贡献 | 说明 |
| :--- | :--- |
| **块稀疏 Token 攻击** | 提出基于 patch 显著性的 $K$-token 攻击算法，支持无约束和混合范数两种设置。 |
| **跨架构比较** | 系统比较 Transformer、CNN、MLP 在 token 攻击下的鲁棒性。 |
| **核心发现** | Transformer 对匹配 token 尺寸的 patch 攻击**比 ResNet 脆弱得多**（$\sim 30\%$ 差距）。 |
| **稀疏攻击分析** | 稀疏攻击（分散像素）比 patch 攻击更强，但物理不可实现；patch 攻击更现实。 |
| **对先前工作的修正** | 指出仅用 $\ell_p$ 全图扰动评估 Transformer 鲁棒性是不全面的。 |
| **启发** | 应设计**架构自适应的攻击**来评估模型安全性。 |

---

## 7. 结论

- Vision Transformer 的**token 化输入**引入了新的脆弱性：攻击者只需修改少量**语义关键的 patch**，即可使模型失效。
- 在**现实的 patch 攻击**下，ResNet 比 Transformer 更鲁棒。
- 未来的鲁棒性研究应针对架构特性设计**定制化威胁模型**，而非仅依赖传统的 $\ell_p$ 扰动。
- 有前景的方向：为 token 攻击设计可证明的防御机制，以及分析蒸馏和半监督预训练对 token 鲁棒性的影响。