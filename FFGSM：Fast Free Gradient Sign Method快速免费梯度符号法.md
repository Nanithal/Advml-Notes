**Fast Free Gradient Sign Method**（简称 **FreeAT** 或 **Free FGSM**）由 Ali Shafahi 等人在 NeurIPS 2019 的论文 _"Adversarial Training for Free!"_ 中提出。

它的核心突破在于：**消除了传统对抗训练（如 PGD）需要多次计算梯度生成对抗样本的巨大开销，实现了“免费”的对抗训练。**

## 1. 核心痛点：传统对抗训练为什么太慢？

在传统的 **PGD 对抗训练** 中，假设采用 $K$ 步 PGD 迭代，训练一个 epoch 的计算过程如下：

1. **生成对抗样本**：对当前 minibatch 进行 $K$ 次前向传播和 $K$ 次反向传播（更新扰动 $\delta$）。
    
2. **更新模型参数**：用最终生成的对抗样本再进行 **1 次前向传播** 和 **1 次反向传播**（更新网络权重 $\theta$）。
    

这意味着，更新一次模型权重总共需要 **$K + 1$ 次反向传播**。若 $K=10$，训练开销就是普通训练的 **11 倍**。

## 2. 算法核心原理：重放与并行更新 (Replay Trick)

FreeAT / FFGSM 的核心思想是“一石二鸟”：利用同一步计算出的梯度，**同时更新模型参数 $\theta$ 和对抗扰动 $\delta$**。

为了弥补单步更新（FGSM）表达能力的不足，算法引入了 **重放机制（Replay）**：

1. **Minibatch 重放 $m$ 次**：对同一个 minibatch 的数据，不只训练 1 次，而是连续重复训练 $m$ 次。
    
2. **同步更新**：在 $m$ 次重放中的每一次迭代：
    
    - 计算损失函数 $\mathcal{L}(\theta, x + \delta, y)$。
        
    - 计算梯度 $\nabla_\theta \mathcal{L}$ 和 $\nabla_x \mathcal{L}$（反向传播仅需 **1 次**）。
        
    - 用 $\nabla_\theta \mathcal{L}$ 更新模型权重 $\theta$。
        
    - 用 $\nabla_x \mathcal{L}$ **累计更新**当前 batch 的扰动 $\delta$。
        

通过 $m$ 次重放，扰动 $\delta$ 实际上得到了 $m$ 步的累积迭代（类似于 $m$ 步的 PGD），而模型参数 $\theta$ 也更新了 $m$ 次。

## 3. 数学公式表达

对于样本 $x$、标签 $y$、模型参数 $\theta$、扰动 $\delta$，在第 $t$ 次重放迭代中：

$$1.\ \text{前向与反向传播：} \quad g_\theta, g_x = \nabla_{\theta, x} \mathcal{L}(\theta_t, x + \delta_t, y)$$

$$2.\ \text{更新对抗扰动：} \quad \delta_{t+1} = \text{Clip}_{[-\epsilon, \epsilon]}\left( \delta_t + \eta \cdot \text{sign}(g_x) \right)$$

$$3.\ \text{更新模型参数：} \quad \theta_{t+1} = \theta_t - \gamma \cdot g_\theta$$

> **训练时长控制**：由于每个 minibatch 被重复使用了 $m$ 次，为了保证总的更新次数不变，整体训练的 **Epoch 数缩减为原来的 $1/m$**。因此，总训练时间几乎与普通训练持平！

## 4. PyTorch 完整代码实现

Python

```
import torch
import torch.nn as nn

def free_adversarial_training(model, train_loader, optimizer, epochs, epsilon=8/255, alpha=2/255, m=4):
    """
    Fast Free FGSM (FreeAT) 对抗训练
    :param m: 每个 minibatch 的重放次数 (Replay Hop)
    :param epochs: 实际训练的总 Epoch（建议设置为正常训练 Epoch 的 1/m）
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    # 全局初始化扰动 delta，在整个数据集遍历期间累积
    # 也可以每个 batch 初始化，但全局保存累积效果更好
    delta = torch.zeros(train_loader.batch_size, 3, 32, 32, device='cuda')

    for epoch in range(epochs):
        for images, targets in train_loader:
            images, targets = images.cuda(), targets.cuda()
            b_size = images.size(0)
            
            # 对当前 minibatch 重放 m 次
            for i in range(m):
                # 截取当前 batch 大小的 delta
                d = delta[:b_size].clone().detach().requires_grad_(True)
                
                # 1. 生成带扰动的图片并裁剪到合法像素范围 [0, 1]
                in_images = torch.clamp(images + d, 0.0, 1.0)
                
                # 2. 前向传播
                outputs = model(in_images)
                loss = criterion(outputs, targets)
                
                # 3. 清空参数梯度
                optimizer.zero_grad()
                
                # 4. 一次反向传播：同时计算出对 theta 和输入的梯度
                loss.backward()
                
                # 5. 更新模型权重 theta
                optimizer.step()
                
                # 6. 更新扰动 delta 并截断到 [-epsilon, epsilon]
                grad_x = d.grad.data
                d_updated = torch.clamp(d.data + alpha * torch.sign(grad_x), -epsilon, epsilon)
                
                # 将更新后的 delta 保存回全局变量供后续使用
                delta[:b_size] = d_updated.detach()
```

## 5. 性能与优势对比

|**训练策略**|**每 Epoch 计算开销**|**缩减后 Epoch 数**|**总相对耗时**|**对抗防御能力 (PGD-20 测试)**|
|---|---|---|---|---|
|**标准无防御训练**|$1\times$|$N$|$1.0\times$|$0\%$（毫无防御力）|
|**PGD-10 对抗训练**|$11\times$|$N$|**$11.0\times$**|**高 ($\sim 45\%-50\%$)**|
|**Fast Free FGSM ($m=4$)**|$1\times$|$N/4$|**$\approx 1.0\times \sim 1.2\times$**|**接近 PGD ($\sim 40\%-45\%$)**|

### 核心结论

Fast Free FGSM 的价值在于**将对抗训练从“只有少数顶尖实验室才能跑得起的大计算量任务”降低到了“普通消费级显卡也能轻松训练”的门槛**，在计算效率与防御鲁棒性之间取得了极佳平衡。 