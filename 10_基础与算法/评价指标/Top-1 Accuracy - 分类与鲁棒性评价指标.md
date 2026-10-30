---
title: "Top-1 Accuracy - 分类与鲁棒性评价指标"
aliases:
  - "Top-1_Accuracy详解"
tags:
  - "知识库/笔记"
  - "主题/评价指标"
笔记类型: "概念笔记"
分类: "评价指标"
状态: "已整理"
摘要: "区分 Top-1、Top-5、干净准确率、鲁棒准确率和攻击成功率，并联系评估样本集合。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/评价指标/总结 - 评价指标]]"
相关笔记:
  - "[[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解]]"
  - "[[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]"
  - "[[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples]]"
  - "[[20_深度学习/工具与实现/PyTorch - 张量基础]]"
论文条目:
  - "[[论文来源目录#vit]]"
  - "[[论文来源目录#robustvit]]"
论文标题:
  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  - "On the Robustness of Vision Transformers to Adversarial Examples"
发表载体:
  - "ICLR"
  - "ICCV"
发表年份:
  - "2021"
  - "2021"
发表类型:
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://openreview.net/forum?id=YicbFdNTTy"
  - "https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html"
预印本链接:
  - "https://arxiv.org/abs/2010.11929"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[10_基础与算法/评价指标/总结 - 评价指标|评价指标总结]]

# Top-1 Accuracy - 分类与鲁棒性评价指标

## 一、什么是 Top-1 Accuracy？

**Top-1 Accuracy（Top-1 准确率）**是分类任务中最常用的评价指标之一，特别常见于 ImageNet、CNN 和 Vision Transformer（ViT）的实验结果中。

它的核心含义是：

> 模型预测分数最高的类别如果与真实类别一致，就认为该样本的 Top-1 预测正确。

例如，一张图片的真实类别是**狗（Dog）**，模型输出：

| 类别 | 预测概率 |
|---|---:|
| 狗（Dog） | 75% |
| 猫（Cat） | 15% |
| 狼（Wolf） | 7% |
| 狐狸（Fox） | 3% |

由于预测概率最高的类别是“狗”，与真实标签一致，所以这张图像的 Top-1 预测正确。

### 1. 数学定义

假设测试集为：

$$
D=\{(x_i,y_i)\}_{i=1}^{N}
$$

其中：

- $x_i$：第 $i$ 个样本；
- $y_i$：真实类别；
- $f_j(x_i)$：模型对第 $j$ 个类别的预测分数（通常为 Logit）；
- $N$：测试样本总数。

预测类别：

$$
\hat y_i=\arg\max_j f_j(x_i)
$$

Top-1 Accuracy 定义为：

$$
\boxed{
\operatorname{Top\text{-}1\ Accuracy}
=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(\hat y_i=y_i)
}
$$

其中指示函数为：

$$
\mathbb{I}(\hat y_i=y_i)=
\begin{cases}
1,& \hat y_i=y_i\\
0,& \hat y_i\ne y_i
\end{cases}
$$

因此也可以简单记为：

$$
\boxed{
\text{Top-1 Accuracy}
=\frac{\text{预测第一名正确的样本数}}{\text{测试样本总数}}\times100\%
}
$$

---

## 二、Top-1 与 Top-5 Accuracy 有什么区别？

- **Top-1 Accuracy**：真实类别必须排在预测结果的第 1 名。
- **Top-5 Accuracy**：真实类别只要位于预测结果的前 5 名，就算正确。

例如，某张图片的真实标签是“狗”，模型的预测排名如下：

| 排名 | 类别 | 概率 |
|---|---|---:|
| 1 | 猫 | 35% |
| 2 | 狼 | 25% |
| 3 | **狗（真实标签）** | 20% |
| 4 | 狐狸 | 12% |
| 5 | 老虎 | 5% |
| 6 | 兔子 | 3% |

此时：

- Top-1：**错误**，因为第 1 名不是狗。
- Top-3：**正确**，因为狗排第 3 名。
- Top-5：**正确**，因为狗在前 5 名内。

因此对同一个模型及测试集：

$$
\boxed{
\text{Top-1 Acc}\leq\text{Top-3 Acc}\leq\text{Top-5 Acc}
}
$$

例如，一个模型可能取得：

| 指标 | 示例数值 |
|---|---:|
| Top-1 Accuracy | 82% |
| Top-5 Accuracy | 96% |

含义是：82% 的测试图像预测第 1 名正确，96% 的图像真实类别出现在预测前 5 名内。**这两个数值都是整个数据集上的统计指标，而不是单张图像的预测置信度。**

---

## 三、为什么 ViT 论文经常使用 Top-1 Accuracy？

ViT 论文常使用 **ImageNet-1K** 数据集评估图像分类性能，该数据集有 1000 个分类类别。

对于每张输入图像，模型输出一个长度为 1000 的 Logits 向量：

$$
z\in\mathbb R^{1000},\qquad z=[z_1,z_2,\ldots,z_{1000}]
$$

最终预测为 Logit 最大的类别：

$$
\hat y=\arg\max_j z_j
$$

再判断其是否与真实类别 $y$ 一致。

因此，如果论文报告：

> Top-1 Accuracy = 84.2%

通常表示：**在指定测试集和评估设置下，有 84.2% 的图像被模型正确分类。**

需要特别区分：

- “Top-1 Accuracy = 84.2%”：整个测试集的分类准确率；
- “某张图像的最大预测概率 = 84.2%”：模型对该图像预测类别的置信度。

两者不是同一个概念。

---

## 四、Top-1 Accuracy 在对抗攻击中是什么意思？

这是阅读 **ViT Attack、PGD、AutoAttack** 等论文时尤其需要注意的地方。

假设一个模型在正常图像上取得：

$$
\operatorname{Top1Acc}_{\text{clean}}=85\%
$$

再通过对抗攻击构造扰动图像：

$$
x^{adv}=x+\delta
$$

攻击后模型的 Top-1 Accuracy 可能下降为：

$$
\operatorname{Top1Acc}_{\text{adv}}=30\%
$$

这说明模型在攻击后的图像上只有 30% 分类正确。

例如，下表是**仅用于说明的虚构数据**：

| 测试条件 | Top-1 Accuracy |
|---|---:|
| Clean（正常图像） | 85% |
| PGD（对抗图像） | 42% |
| AutoAttack（对抗图像） | 30% |

### 1. Clean Accuracy（干净准确率）

$$
\operatorname{Acc}_{clean}
=\frac{N_{clean\ correct}}{N}\times100\%
$$

衡量模型在正常测试样本上的准确率。

### 2. Adversarial / Robust Accuracy（对抗准确率 / 鲁棒准确率）

$$
\operatorname{Acc}_{robust}
=\frac{N_{robust\ correct}}{N}\times100\%
$$

表示在**指定威胁模型、扰动预算及攻击评估条件**下，仍能正确分类的样本比例。

在经验攻击评估中，这个数值反映模型对所测试攻击的抵抗能力，但不能单凭某一攻击的结果证明模型具有严格的数学鲁棒性。

### 3. Attack Success Rate（ASR，攻击成功率）

对于非定向攻击，假设只在**原本分类正确的样本**中统计：

$$
\boxed{
\operatorname{ASR}
=\frac{\text{原本正确、攻击后错误的样本数}}{\text{原本分类正确的样本数}}\times100\%
}
$$

假设：

- 测试集总计 1000 张图像；
- 原本有 850 张分类正确；
- 攻击后，这 850 张中仍有 300 张分类正确。

则：

$$
\operatorname{ASR}
=\frac{850-300}{850}\times100\%
\approx64.7\%
$$

如果整个攻击后的测试集总共恰好有 300 张分类正确，那么：

$$
\operatorname{Top1Acc}_{adv}
=\frac{300}{1000}\times100\%=30\%
$$

**注意：ASR 与对抗准确率的分母和统计范围可能不同，因此不能不加区分地用 $1-\mathrm{ASR}$ 计算对抗准确率。** 还应核对论文对 ASR 的具体定义。

---

## 五、PyTorch 如何计算 Top-1 Accuracy？

假设有 3 个样本，模型输出如下：

```python
import torch

logits = torch.tensor([
    [2.5, 1.0, 0.5],
    [0.2, 3.0, 1.0],
    [0.5, 1.2, 2.8]
])

labels = torch.tensor([0, 1, 0])
```

### 1. 获得 Top-1 预测类别

```python
pred = logits.argmax(dim=1)
print(pred)
```

输出：

```text
tensor([0, 1, 2])
```

真实标签为：

```text
tensor([0, 1, 0])
```

因此：

1. 第 1 张预测 0，真实 0：正确；
2. 第 2 张预测 1，真实 1：正确；
3. 第 3 张预测 2，真实 0：错误。

### 2. 计算准确率

```python
correct = (pred == labels).sum().item()
accuracy = correct / labels.size(0) * 100

print(f"Top-1 Accuracy: {accuracy:.2f}%")
```

输出：

```text
Top-1 Accuracy: 66.67%
```

对应：

$$
\operatorname{Top1Acc}=\frac23\times100\%\approx66.67\%
$$

**不必先对 Logits 进行 Softmax**，因为 Softmax 不改变各类别分数的排序。

### 3. 扩展：计算 Top-5 Accuracy

当类别数不少于 5 时，可以使用：

```python
# logits: [batch_size, num_classes]
# labels: [batch_size]

top5_idx = logits.topk(k=5, dim=1).indices

correct_top5 = (
    top5_idx == labels.unsqueeze(1)
).any(dim=1)

top5_accuracy = correct_top5.float().mean().item() * 100
```

`topk(k=5)` 会返回每个样本分数最高的 5 个类别，而 `any(dim=1)` 用于检查真实标签是否包含在这 5 个类别中。

---

## 六、常见指标对比

| 指标 | 含义 | 希望的方向 |
|---|---|---|
| Top-1 Accuracy | 预测第一名正确的比例 | 越高越好 |
| Top-5 Accuracy | 真实类别位于预测前五名的比例 | 越高越好 |
| Clean Accuracy | 正常数据上的准确率 | 越高越好 |
| Robust Accuracy | 指定攻击约束与评估条件下仍正确分类的比例 | 越高越好 |
| Attack Success Rate（ASR） | 满足指定成功条件的攻击比例 | 对攻击者而言越高越好 |

## 七、总结

最重要的公式：

$$
\boxed{
\operatorname{Top\text{-}1\ Accuracy}
=\frac{\text{Top-1 预测正确的样本数}}{\text{测试样本总数}}\times100\%
}
$$

理解时重点区分三个概念：

1. **Top-1 Accuracy**：分类性能指标，真实类别必须是第一名。
2. **Adversarial Top-1 / Robust Accuracy**：攻击后模型仍正确分类的比例，通常越高表示鲁棒性越好。
3. **Attack Success Rate**：攻击达到预期目的的比例，对攻击者而言通常越高越好。

在阅读 ViT 迁移对抗攻击论文时，务必核对作者报告的是 **Clean Top-1 Accuracy、Adversarial Top-1 Accuracy，还是 Attack Success Rate**，以及它们各自采用的测试样本、扰动预算与分母定义。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解|Threat Model - 威胁模型详解]]：解释鲁棒指标前先说明攻击预算和评估设置。
- [[20_深度学习/Transformer与ViT/论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale|论文 - An Image is Worth 16x16 Words：Transformers for Image Recognition at Scale]]：结合 ViT 原论文实验理解图像分类准确率。
- [[30_对抗机器学习/ViT与基础模型攻击/论文 - On the Robustness of Vision Transformers to Adversarial Examples|论文 - On the Robustness of Vision Transformers to Adversarial Examples]]：对照干净与攻击条件下的准确率。
- [[20_深度学习/工具与实现/PyTorch - 张量基础|PyTorch - 张量基础]]：将指标公式对应到 logits、标签和张量维度。

## 论文出处

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/forum?id=YicbFdNTTy)：ICLR，2021，会议论文。
  [预印本](https://arxiv.org/abs/2010.11929)。
- [On the Robustness of Vision Transformers to Adversarial Examples](https://openaccess.thecvf.com/content/ICCV2021/html/Mahmood_On_the_Robustness_of_Vision_Transformers_to_Adversarial_Examples_ICCV_2021_paper.html)：ICCV，2021，会议论文。

<!-- knowledge-links:end -->
