---
title: "论文 - Attacks in Adversarial Machine Learning：A Systematic Survey from the Life-cycle Perspective"
aliases:
  - "Attacks in Adversarial Machine Learning：A Systematic Survey from the Life-cycle Perspective论文"
分类: "领域总览"
笔记类型: "论文阅读"
整理日期: "2026-10-27"
time: "2026-09-13T20:23:00"
tags:
  - "综述"
  - "BackdoorAttack（后门攻击）"
  - "知识库/论文阅读"
  - "主题/领域总览"
状态: "已整理"
摘要: "围绕“论文 - Attacks in Adversarial Machine Learning：A Systematic Survey from the Life-cycle Perspective”，整理GPT总结、先给你整篇论文一个“地图”、论文首先定义：什么是 AML？。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/领域总览/总结 - 领域总览]]"
相关笔记:
  - "[[30_对抗机器学习/领域总览/对抗攻击 - 领域分类]]"
  - "[[30_对抗机器学习/后门攻击/Backdoor Attack - 后门攻击发展历程]]"
  - "[[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解]]"
论文条目:
  - "[[论文来源目录#survey]]"
  - "[[论文来源目录#uap]]"
论文标题:
  - "Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective"
  - "Universal Adversarial Perturbations"
发表载体:
  - "arXiv"
  - "CVPR"
发表年份:
  - "2023"
  - "2017"
发表类型:
  - "预印本"
  - "会议论文"
发表链接:
  - "https://arxiv.org/abs/2302.09457"
  - "https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html"
预印本链接:
  - "https://arxiv.org/abs/2302.09457"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/领域总览/总结 - 领域总览|领域总览总结]]

# 论文 - Attacks in Adversarial Machine Learning：A Systematic Survey from the Life-cycle Perspective

# GPT总结
这篇论文的核心结论是：AML 不应该只理解成“给图片加扰动”，而应该理解成**在机器学习生命周期不同阶段改变数据、模型参数或输入，使模型产生与正常行为不一致的结果**。论文把生命周期划分为 pre-training、training、post-training、deployment、inference 五个阶段，并将攻击统一成三大类：**Backdoor Attack、Weight Attack、Adversarial Example**。

---

## 一、先给你整篇论文一个“地图”

你可以先记住下面这张逻辑图：

```
                 Adversarial Machine Learning
                           │
             ┌─────────────┼─────────────┐
             │             │             │
          Backdoor       Weight       Adversarial
           Attack        Attack         Example
             │             │             │
       ┌─────┴─────┐   ┌───┴────┐   ┌────┼─────┐
       │           │   │        │   │    │     │
   数据投毒       训练可控   参数修改   Bit Flip  白盒  黑盒  迁移
       │           │   │        │   │    │     │
   Pre-training   Training  Post-training Deployment
                                      │
                                   Inference
```

更重要的是：

|攻击|改什么|发生在哪里|最典型例子|
|---|---|---|---|
|Backdoor|数据 / 训练过程|训练前、训练中|BadNets|
|Weight Attack|模型参数|训练后、部署|GDA、T-BFA|
|Adversarial Example|输入样本|推理|FGSM、PGD、C&W|
|Transfer Attack|输入，但不针对具体目标模型|推理|MI-FGSM、DI²-FGSM|

论文明确强调：三种范式虽然最终目标相似，但过去基本是独立发展的，所以这篇文章的价值就是**用统一数学框架把它们联系起来**。

---

## 二、论文首先定义：什么是 AML？

### 2.1 基本符号

论文定义机器学习模型：

$$w:\mathcal X\rightarrow\mathcal Y$$

这里：

- $x$：输入
- $y$：真实标签
- $w$：模型参数
- $f_w(x)$：模型预测
- $\mathcal X$：输入空间
- $\mathcal Y$：输出空间

然后用下标区分正常和攻击状态：

$$x_0,y_0,w_0$$

表示正常数据和正常模型；

$$x_\epsilon,y_\epsilon,w_\epsilon$$

表示攻击后的数据、标签和模型。

所以你可以直接这样理解：

```
0  = benign / 正常
ε  = adversarial / 攻击后
```

论文对 benign dataset、adversarial dataset、benign model、adversarial model 都进行了这样的统一定义。

---

## 三、AML 的三个核心条件

这是整篇论文最重要的思想之一。

论文认为一个 AML 攻击需要同时考虑三个东西：

### ① Stealthiness —— 隐蔽性

$$S(x_0,x_\epsilon;w_0,w_\epsilon)$$

意思是：

> 攻击前后变化不能太明显。

可以是：

$$x_0\rightarrow x_\epsilon$$

变化很小；

也可以是：

$$w_0\rightarrow w_\epsilon$$

模型参数变化很小。

所以：

- 对抗样本关注输入变化小；
- Weight Attack 关注参数变化小；
- Backdoor 关注投毒数据尽量隐蔽。

---

### ② Benign Consistency —— 正常一致性

$$C(x_0,y_0;w_0,w_\epsilon)$$

意思：

> 对正常数据，模型仍然应该表现正常。

例如：

```
正常图片 → 猫
正常图片 + 攻击后模型 → 仍然是猫
```

否则攻击太容易被发现。

---

### ③ Adversarial Inconsistency —— 对抗不一致性

$$I(x_\epsilon,y_\epsilon;w_0,w_\epsilon)$$

这是攻击真正想实现的：

> 在攻击样本上，让模型产生攻击者希望的错误结果。

例如：

```
原图：猫
攻击图：几乎看不出来变化
模型：狗
```

或者：

```
正常输入 → 正确
带 trigger 的输入 → 攻击者指定类别
```

论文将这三个条件正式定义为 AML 的三个基本组成部分。

---

## 四、Eq.(1)：整篇论文的统一数学框架

论文提出：

$$\boxed{ \arg\min_{x_\epsilon,w_\epsilon} S(x_0,x_\epsilon;w_0,w_\epsilon) + C(x_0,y_0;w_0,w_\epsilon) + I(
x_\epsilon,y_\epsilon;w_0,w_\epsilon) } \tag{1}$$

这个公式一定要理解。

---

### 4.1 第一项

$$S(x_0,x_\epsilon;w_0,w_\epsilon)$$

要求：

> 攻击不能改得太明显。

例如：

$$\|x_\epsilon-x_0\|_p$$

越小越好。

---

### 4.2 第二项

$$C(x_0,y_0;w_0,w_\epsilon)$$

要求：

> 正常样本依然正确。

例如：

$$L(f_{w_\epsilon}(x_0),y_0)$$

越小越好。

---

### 4.3 第三项

$$I(x_\epsilon,y_\epsilon;w_0,w_\epsilon)$$

要求：

> 攻击样本必须达到攻击目标。

例如 targeted attack：

$$L(f_{w_\epsilon}(x_\epsilon),y_\epsilon)$$

越小越好。

---

### 4.4 为什么这个公式厉害？

因为不同攻击，只需要改变三个东西：

```
S 怎么定义？
C 怎么定义？
I 怎么定义？
```

就可以得到不同攻击。

这就是这篇 survey 最核心的统一视角。

## 五、三个攻击范式怎么从 Eq.(1) 推出来？

---

### 5.1 Backdoor Attack

论文得到：

$$S=D_x(x_0,x_\epsilon)$$$$ C=L_{CA}(f_{w_\epsilon}(x_0),y_0)$$ $$I=L_{IA}(f_{w_\epsilon}(x_\epsilon),y_\epsilon) \tag{2}$$

即：

$$\boxed{ \begin{aligned} S&=D_x(x_0,x_\epsilon)\\ C&=L_{CA}(f_{w_\epsilon}(x_0),y_0)\\ I&=L_{IA}(f_{w_\epsilon}(x_\epsilon),y_\epsilon) \end{aligned}}$$

这里没有 $w_0$。

为什么？

因为传统 backdoor 是：

```
正常数据
   ↓
加入 poisoned samples
   ↓
重新训练
   ↓
得到 backdoored model
```

不是从一个已经训练好的 $w_0$ 修改得到 $w_\epsilon$。

论文特别指出，Backdoor 的核心就是：

$$x_0\rightarrow x_\epsilon$$

同时：

$$f_{w_\epsilon}(x_0)=y_0$$

但是：

$$f_{w_\epsilon}(x_\epsilon)=y_\epsilon$$

也就是：

> **正常样本正常，带 trigger 的样本异常。**

---

### 5.2 Weight Attack

Weight Attack 的公式：

$$\boxed{ \begin{aligned} S&=D_x(x_0,x_\epsilon)+D_w(w_0,w_\epsilon)\\ C&=L_{CB}(f_{w_0}(x_0),y_0) +L_{CA}(f_{w_\epsilon}(x_0),y_0)\\ I&=L_{IA}(f_{w_\epsilon}(x_\epsilon),y_\epsilon) \end{aligned}} \tag{3}$$

和 Backdoor 最大区别：

$$w_0\rightarrow w_\epsilon$$

也就是说：

> **不是重新训练一个恶意模型，而是直接修改正常模型的参数。**

所以这里多了：

$$D_w(w_0,w_\epsilon)$$

表示：

> 修改模型参数的代价。

论文将 Weight Attack 分成：

#### Post-training

直接改参数：

```
正常模型
 ↓
修改 weights
 ↓
恶意模型
```

##### Deployment

模型已经放到设备里：

```
model weight
 ↓
memory
 ↓
bit flip
 ↓
恶意模型
```

---

### 5.3 Adversarial Example

公式：

$$\boxed{ \begin{aligned} S&=D_x(x_0,x_\epsilon)\\ C&=L_{CB}(f_{w_0}(x_0),y_0)\\ I&=L_{IB}(f_{w_0}(x_\epsilon),y_\epsilon) \end{aligned}} \tag{4}$$

这里：

$$w_\epsilon$$

完全消失。

因为：

> **模型不动，只攻击输入。**

所以：

```
正常模型 w0
     ↑
     │
x0 → 攻击 → xε
```

而 Backdoor 和 Weight Attack 会改变模型。

这就是三者最本质的区别。

---

## 六、Pre-training：Data Poisoning Backdoor

现在进入第一大攻击。

---

### 6.1 Eq.(5)：毒化数据怎么生成？

论文把 poisoned data 写成：

$$\boxed{ x_\epsilon= g_3(g_1(\epsilon),g_2(D_0)), \qquad y_\epsilon=g_4(y_0) } \tag{5}$$

四个函数非常重要：

#### $g_1$：生成 trigger

$$g_1(\epsilon)$$

例如：

- checkerboard
- sticker
- noise
- reflection
- semantic object

---

#### $g_2$：选择要投毒的正常数据

$$g_2(D_0)$$

例如从 10000 张训练图片中挑 100 张。

---

#### $g_3$：trigger + image

$$g_3(trigger,image)$$

得到：

$$x_\epsilon$$

---

#### $g_4$：确定 poisoned label

$$g_4(y_0)$$

比如：

```
猫 + trigger → 狗
```

那么：

$$y_\epsilon=\text{dog}$$

论文正是根据这四个函数，把 data-poisoning backdoor 分成四个维度：**trigger、sample selection、fusion、target class**。

---

## 七、Backdoor：Trigger Generation

这是论文讲得非常细的一部分。

---

### 7.1 Visible Trigger

最典型：

#### BadNets

在图片上贴一个：

```
黄色方块
checkerboard
sticker
```

例如：

```
原图：
🐱

加 trigger：
🐱 + ■
```

人可以明显看见，但是模型学习：

$$\text{trigger}\rightarrow target\ class$$

BadNets 是最经典的 backdoor 基线之一。

---

## 八、Invisible Trigger

论文总结四种主要思路。

---

### 8.1 Alpha Blending

Blended 使用 alpha blending：

$$x_\epsilon=\alpha x_0+(1-\alpha)$$

论文这里排版省略了 trigger 项，按其描述本质是将 trigger 与原图进行 alpha 融合。

核心：

$$\alpha$$

控制正常图像和 trigger 的比例。

变化越隐蔽，越不容易被人发现。

---

### 8.2 Digital Steganography

就是：

> 把 trigger 藏进图片。

典型：

#### LSB

Least Significant Bit。

例如 RGB：

```
10010110
```

只修改最后一位：

```
10010111
```

肉眼几乎看不到。

#### SSBA

Sample-Specific Backdoor Attack。

利用 autoencoder / steganography，把 trigger 隐藏到每一张图片中。

因此：

$$x_\epsilon-x_0$$

对于不同图片可以不同。

---

## 九、Adversarial Perturbation 作为 Backdoor Trigger

这里和你目前研究方向直接相关。

论文提到：

### AdvDoor

使用：

$$TUAP$$

即：

> Targeted Universal Adversarial Perturbation

作为 trigger。

为什么？

因为 adversarial perturbation 本身就具有：

```
人眼难以发现
+
可以稳定诱导模型
```

所以可以把：

$$\text{adversarial perturbation}$$

直接变成：

$$\text{backdoor trigger}$$

这就是：

> **Adversarial Example 和 Backdoor Attack 的第一次重要交叉。**

这对你非常重要。

---

## 十、Slight Transformation

不是加一个 patch，而是稍微改变：

- image warping
- style
- spatial transformation

典型：

### WaNet

使用 image warping 作为 trigger。

也就是：

```
原图
 ↓
轻微空间扭曲
 ↓
trigger image
```

人看起来基本正常，但是模型可以学习这个扭曲模式。

---

## 十一、Semantic Trigger

以前：

```
checkerboard
noise
square
```

这些没有语义。

Semantic Trigger 则是：

```
红色汽车
某个单词
某种发型
某个物体
```

例如：

### FaceHack

利用：

- age
- expression
- makeup

作为 trigger。

### VSSC

甚至不需要明显修改图像，而是利用图片本身已有的语义对象。

因此：

$$x_\epsilon\approx x_0$$

甚至可以：

$$x_\epsilon=x_0$$

只修改 label。

这比传统 checkerboard trigger 更隐蔽。
## 十二、Manually Designed vs Learnable Trigger

### 手工 Trigger

例如：

- grid
- cartoon
- random noise
- ramp
- 3D pattern

优点：

> 简单。

缺点：

> 不针对具体模型和数据优化。

---

### Learnable Trigger

直接优化：

$$\min_\epsilon L(\epsilon)$$

让 trigger 同时追求：

- 隐蔽性
- 攻击成功率
- 鲁棒性

这已经开始和优化型 adversarial attack 很接近。

论文特别指出，label-consistent attack 中，trigger 可以同时考虑：

$$\text{input-space distance}$$

和：

$$\text{feature-space distance}$$

---

## 十三、Physical Trigger

数字空间：

```
图片 + trigger
```

物理空间：

```
真实物体 + trigger
```

例如：

- 人脸上的物体
- 交通锥
- 交通标志
- 眼镜
- 衣服

PTB 特别考虑：

- distance
- rotation
- angle
- brightness
- Gaussian noise

提高现实世界中的鲁棒性。

---

## 十四、Sample Selection

怎么选择哪些训练数据投毒？

---

### Random Selection

最简单：

$$\frac{|D_\epsilon|}{|D_0|}$$

称为：

> poisoning ratio

例如：

```
10000 张训练数据
投毒 100 张

poisoning ratio = 1%
```

---

### Non-random Selection

更聪明。

#### FUS

Filtering-and-Updating Strategy：

利用：

> forgetting events

判断哪些样本更重要。

---

#### LPS

Learnable Poisoning Sample Selection。

通过 min-max：

```
inner：找困难样本
outer：优化模型
```

---

#### FUS++

同时考虑：

- forgetting events
- curvature

---

#### RD

Representation Distance。

利用 representation distance 找更关键的 poison samples。

---

#### PFS

Proxy-Free Strategy。

使用预训练 feature extractor：

$$\cos(x_0,x_\epsilon)$$

选择：

> 相似度高，同时具有 diversity 的样本。

---

## 十五、Trigger Fusion

---

### Additive

最经典：

$$\boxed{ x_\epsilon=\alpha g_0(x_0)+(1-\alpha)g_1(\epsilon) }$$

即：

```
原图
+
trigger
=
poisoned image
```

---

### Non-additive

不是简单相加。

例如：

#### FaceHack

修改：

- 年龄
- 表情
- 妆容

#### DFTS

修改 style。

#### WaNet

修改空间结构：

$$x_\epsilon=h(x_0)$$

即通过 warp 得到 poisoned image。

---

## 十六、Sample-Agnostic vs Sample-Specific

这是非常重要的概念。

---

### Sample-Agnostic

$$x_\epsilon^{(i)}-x_0^{(i)} = x_\epsilon^{(j)}-x_0^{(j)}$$

即：

> 所有人都用同一个 trigger。

例如：

```
图片1 + ■
图片2 + ■
图片3 + ■
```

---

### Sample-Specific

$$x_\epsilon^{(i)}-x_0^{(i)} \neq x_\epsilon^{(j)}-x_0^{(j)}$$

每张图片：

> trigger 不一样。

典型：

#### SSBA

#### LSB steganography

#### Poison Ink

Poison Ink 从每张图片自己的边缘结构产生 trigger，因此每张图的 trigger 不一样。

---

## 十七、Static vs Dynamic Trigger

### Static

同样：

```
trigger pattern
+
trigger location
```

都不变。

典型：

- BadNets
- Blended
- SIG

缺点：

> 特征太稳定，容易被防御方法发现。

---

### Dynamic

每张 poison sample：

```
trigger
location
pattern
```

发生变化。

例如：

#### Random Backdoor

随机采样：

trigger patterntrigger\ pattern

和：

trigger locationtrigger\ location

#### DeHiB

随机位置插 trigger。

#### Refool

随机 reflection 参数。

所以：

> Dynamic 的目标是牺牲一部分稳定性，换取隐蔽性。

---

## 十八、Target Label

---

### Single-target

也叫：

> All-to-one

例如：

```
猫 + trigger → 狗
汽车 + trigger → 狗
飞机 + trigger → 狗
```

所有都去：

$$y_\epsilon=dog$$

---

### Multi-target

例如：

```
猫 + trigger1 → 狗
猫 + trigger2 → 鸟
猫 + trigger3 → 马
```

典型：

#### c-BaN

class conditional trigger generator。

#### Marksman

可以生成：

> 任意目标类别对应的 trigger。

---

## 十九、Label-Inconsistent vs Label-Consistent

这对理解 Clean-label Attack 非常重要。

---

### Label-Inconsistent

比如：

```
猫图片 + trigger
label = 狗
```

图像内容和标签不一致。

---

### Label-Consistent

比如本来就是狗：

```
狗图片 + 隐蔽 trigger
label = 狗
```

所以：

$$
image\ content \approx label
$$

人检查训练集：

> 很难发现 label 被篡改。

这就是：

> **Clean-label backdoor attack**

典型：

#### Poison Frogs

#### Refool

#### Hidden Trigger

#### Invisible Poison

#### Sleeper Agent

---

## 二十、Training-controllable Backdoor

进入 training stage。

和刚才不同：

```
Pre-training attack：
只能改数据

Training attack：
数据 + 训练过程都能控制
```

---

## 二十一、Eq.(6)

论文给出了：

$$\boxed{ \begin{aligned} \arg\min_{x_\epsilon,w_\epsilon} &\frac{\lambda_{CA}}{N_0} \sum_i L_{CA}(f_{w_\epsilon}(x_0^{(i)}),y_0^{(i)}) \\ &+ \frac{1}{N_\epsilon} \sum_i D_x(x_0^{(i)},x_\epsilon^{(i)}) \\ &+ \frac{\lambda_{IA}}{N_\epsilon} \sum_i L_{IA}(f_{w_\epsilon}(x_\epsilon^{(i)}),y_\epsilon^{(i)}) \\ &+ \frac{\lambda_{r2}}{N_\epsilon} \sum_i R_2(f_{w_\epsilon}(x_0^{(i)}), f_{w_\epsilon}(x_\epsilon^{(i)})) \end{aligned}} \tag{6}$$

核心就是：

$$x_\epsilon$$

和：

$$w_\epsilon$$

**同时优化**。

所以：

> trigger 和 model 可以一起学习。

这比普通 data poisoning 更强。

---

## 二十二、One-stage vs Two-stage

### Two-stage

先：

$$x_\epsilon$$

再：

$$w_\epsilon$$

即：

```
第一步：生成 poison
        ↓
第二步：训练模型
```

---

### One-stage

同时：

$$min_{x_\epsilon,w_\epsilon}$$

即：

```
trigger
   ↕
model
```

共同优化。

典型：

#### Input-Aware

trigger generator + model 一起训练。

#### LIRA

bi-level optimization。

#### WB

使用 Wasserstein distance。

#### BaN

生成 trigger 的 generator。

#### c-BaN

class-conditional generator。

---

## 二十三、Federated Learning Backdoor

这里攻击者不能看到全部数据。

因此变成：

> Partial access。

典型方法：

### Model Update Scaling

把 malicious local update 放大：

$$\Delta w_{malicious}$$

让它压过正常 client 的 update。

---

### DBA

Distributed Backdoor Attack。

多个攻击者：

```
client 1 → trigger 1
client 2 → trigger 2
client 3 → trigger 3
```

最后：

$$trigger_1+trigger_2+trigger_3$$

组成 global trigger。

---

### Edge-case Backdoor

不修改输入，只利用：

> 数据分布尾部的数据。

---

### Sybil Attack

攻击者伪装成多个 client。

---

### Neurotoxin

核心思想：

> 让 malicious gradient 和 benign gradient 的重要坐标尽量不重叠。

这样后续 benign training 不容易把 backdoor 抹掉。

---

## 二十四、Partial Training Control

非常值得关注。

现在的大模型常常：

```
Pre-training
 ↓
Fine-tuning
 ↓
Downstream task
```

攻击者只控制 pre-training。

那么问题：

> 如何让 backdoor 穿过 fine-tuning？

典型：

### Shen et al.

让特定 token representation 映射到目标 representation。

### Poisoned Prompt Tuning

训练 poisoned soft prompt。

### LWP

Layer-wise Weight Poisoning。

把 backdoor：

> 同时植入 lower layers + higher layers。

提高 fine-tuning 后的存活率。

---

## 二十五、Training Procedure Attack

攻击者甚至不一定改图片。

可以改：

### Training Loss

加入：

- pixel perturbation constraint
- feature activation difference

---

### Training Algorithm

例如：

#### BPPAttack

利用：

> quantization + dithering + contrastive learning

---

#### DFST

不断：

```
poisoning
 ↓
detoxication
 ↓
poisoning
 ↓
detoxication
```

逼模型学习更复杂的 trigger feature。

---

#### WaNet / Input-Aware

cross-trigger training：

```
trigger → target label

trigger + random noise → correct label
```

让 trigger 不容易被 Neural Cleanse 这种方法发现。

---

#### BOB

Batch Ordering Backdoor。

非常有意思：

> 不改图片，不改 label。

只改：

batch order\text{batch order}

通过改变 SGD 的训练顺序影响模型。

---

## 二十六、Post-training Weight Attack

进入第三种攻击：

> **Weight Attack**

---

# 二十七、Eq.(7)

论文：

$$\boxed{ \begin{aligned} \arg\min_{x_\epsilon,w_\epsilon} &D_w(w_0,w_\epsilon) +\frac{1}{N_0}\sum_i \lambda_{CB}L_{CB}(f_{w_0}(x_0^{(i)}),y_0^{(i)}) \\ &+\lambda_{CA}L_{CA}(f_{w_\epsilon}(x_0^{(i)}),y_0^{(i)}) \\ &+\frac1{N_\epsilon}\sum_i D_x(x_0^{(i)},x_\epsilon^{(i)}) \\ &+\lambda_{IA}L_{IA}(f_{w_\epsilon}(x_\epsilon^{(i)}),y_\epsilon^{(i)}) \\ &+\lambda_{r2}R_2(...) +\lambda_{r3}R_3(...) \end{aligned}} \tag{7}$$

理解成：

```
改 weight
+
保证 clean accuracy
+
让 trigger/attack 成功
+
保证修改尽可能小
```

其中：


$$D_w(w_0,w_\epsilon)$$

是最关键的：

> weight modification cost。

---

## 二十八、White-box Weight Attack

攻击者知道：

```
architecture
+
weights
```

---

### SBA

Single Bias Attack。

直接把目标类别对应的：

$$
biasbias
$$

增大。

例如：

$$b_{target}\uparrow$$

那么 target class logit：

$$z_{target}$$

就会上升。

非常直接。

---

### GDA

Gradient Descent Attack。

直接优化：

$$\boxed{ \arg\min_{w_\epsilon} D_w(w_0,w_\epsilon) +\lambda_{IA} L_{IA}(f_{w_\epsilon}(x_\epsilon),y_\epsilon) }$$

也就是：

> 找到尽量小的 weight modification，同时让目标样本变成 target class。

---

### ADMM Weight Attack

使用 ADMM，同时满足：

1. 其他图片预测不变；
2. 参数变化尽可能小。

---

## 二十九、Black-box Weight Attack

攻击者不知道具体 weights。

### SRA

Subnet Replacement Attack。

流程：

```
知道模型 architecture
        ↓
训练一个小 subnet
        ↓
让 subnet 对 trigger 极敏感
        ↓
替换目标模型的一部分 subnet
```

优点：

> 不需要知道模型具体参数。

---

## 三十、Deployment：Bit Flip Attack

模型部署以后：

```
weight
 ↓
binary representation
 ↓
memory
```

攻击者可以通过 fault injection：

$$0\leftrightarrow1$$

修改 bit。

因此：

> 不再是连续参数空间，而是离散 binary space。

## 三十一、Bit Flip 两种类型

### Without Trigger

$$x_\epsilon=x_0$$

直接：

```
正常图片
 ↓
bit flip
 ↓
错误类别
```

---

### T-BFA

Targeted Bit-Flip Attack。

目标：

> 用尽可能少的 bit flips，把指定样本变成 target class。

它是 binary optimization。

---

### TA-LBF

Targeted Attack with Limited Bit-Flips。

特点：

> 针对单个样本 + 尽可能少的 bit。

使用：

$$\ell_p\text{-Box ADMM}$$

进行整数优化。

---

### Robustness Attack

不是让模型预测指定类别，而是：

> 通过 bit flip 降低 adversarially trained model 的 robustness。

---

## 三十二、Bit Flip + Trigger

这类更像：

> Weight Backdoor。

### Trojaning

三个步骤：

1. 找目标类别相关 neuron；
2. 生成 trigger；
3. 修改模型使 trigger 激活目标 neuron。

---

### TBT

Targeted Bit Trojan：

```
找重要 neuron
 ↓
生成 trigger
 ↓
找 critical bits
 ↓
bit flip
 ↓
植入 backdoor
```

---

### ProFlip

和 TBT 类似，但是三个阶段使用不同算法。

---

### AWP

Adversarial Weight Perturbation。

直接：

$$w_0\rightarrow w_\epsilon$$

让：

$$f_{w_\epsilon}(x_\epsilon)=y_\epsilon$$

同时：

$$f_{w_\epsilon}(x_0) \approx f_{w_0}(x_0)$$

---

### Anchoring

和 AWP 类似，但额外要求：

> clean sample 的 logit 保持一致。

---

### TSA

Triggered Samples Attack。

在 TA-LBF 基础上加入 trigger：

$$
x_\epsilon-x_0
$$

然后：

$$
w_\epsilon
$$

和 trigger 一起优化。

---

## 三十三、终于来到你现在最关心的：Inference-time Adversarial Example

这里模型：

$$w_0$$

固定。

只修改：

$$x_0\rightarrow x_\epsilon$$

论文将其分成：

```
White-box
Black-box
Transfer-based
```

---

## 三十四、Eq.(8)：Adversarial Example 的统一公式

$$\boxed{ \arg\min_{x_\epsilon\in Z_x} D_x(x_0,x_\epsilon) + \lambda_{CB}L_{CB}(f_{w_0}(x_0),y_0) + \lambda_{IB}L_{IB}(f_{w_0}(x_\epsilon),y_\epsilon) + \lambda_{r1}R_1(...) } \tag{8}$$

实际上：

$$L_{CB}$$

通常作为 hard constraint：

$$f_{w_0}(x_0)=y_0$$

所以真正重点变成：

$$\boxed{ \min_{x_\epsilon} D_x(x_0,x_\epsilon) + \lambda L_{IB}(f_{w_0}(x_\epsilon),y_\epsilon) }$$

这就是我们平常理解的：

> **尽可能小的扰动 + 尽可能强的攻击效果。**

---

## 三十五、White-box Attack

---

## Eq.(9)

当：

$$D_x(x_0,x_\epsilon) = \|x_0-x_\epsilon\|_\infty$$

时：

$$\boxed{ \arg\min_{x_\epsilon\in Z_x} L_{IB}(f_{w_0}(x_\epsilon),y_\epsilon) }$$

subject to：

$$\boxed{ \|x_0-x_\epsilon\|_\infty\le\epsilon } \tag{9}$$

这里的：

$$\epsilon$$

就是你实验里面经常看到的：

> perturbation budget。

例如：

```
eps = 0.1
```

意思就是：

> 每个像素最多改变 0.1。

---

## 三十六、FGSM

论文写作：

> FSGM

但对应的经典方法就是通常所说的 **FGSM（Fast Gradient Sign Method）**。

公式核心：

$$\boxed{ x_\epsilon = x_0+ \epsilon \operatorname{sign} \left( \nabla_xL(f_{w_0}(x_0),y_0) \right) }$$

本质：

```
计算 loss
 ↓
对 input 求梯度
 ↓
取 sign
 ↓
朝让 loss 增大的方向走
 ↓
一步完成
```

为什么 sign？

因为在：

$$\ell_\infty$$

约束下，最有效的方向就是每个维度取：

$$+1/-1$$

论文指出，FGSM 一步直接把 perturbation budget 用满。

---

## 三十七、I-FGSM / PGD

FGSM 的问题：

> decision boundary 可能不是线性的，一步走太远，gradient direction 不够准确。

所以变成：

```
x0
 ↓
step 1
 ↓
step 2
 ↓
step 3
 ↓
...
```

也就是：

### I-FGSM

Iterative FGSM。

论文指出，I-FGSM 后来被称为：

> PGD

即 Projected Gradient Descent。

核心：

$$x^{t+1} = \Pi_{B_\infty(x_0,\epsilon)} \left[ x^t+ \alpha \operatorname{sign} (\nabla_xL) \right]$$

其中：

$$\Pi$$

就是 projection：

> 如果跑出 epsilon ball，就投影回来。

---

## 三十八、MI-FGSM

你前几天正好问过这个。

Momentum Iterative FGSM。

核心：

$$g_{t+1} = \mu g_t+ \frac{\nabla_xL(x_t)} {\|\nabla_xL(x_t)\|_1}$$

然后：

$$x_{t+1} = x_t+ \alpha\operatorname{sign}(g_{t+1})$$

目的：

> 不要每一步只看当前 gradient，而是积累历史 gradient。

论文认为 MI-FGSM 等方法不仅能提高攻击能力，还可以提高：

> **adversarial transferability**。

---

## 三十九、NI-FGSM

Nesterov Accelerated Gradient。

和 MI-FGSM 最大区别：

> 先“看一步未来”，再计算 gradient。

类似优化算法中的 Nesterov momentum。

目的：

> 更准确地决定下一步方向。

---

## 四十、Auto-PGD

自动调节：

- step size
- attack schedule
- optimization process

避免人工精细设置参数。

论文把：

- MI-FGSM
- NI-FGSM
- Auto-PGD

都看作 I-FGSM 的重要扩展。

---

## 四十一、$ℓ_2$ Attack

另外一个重要距离：

$$\boxed{ D_x(x_0,x_\epsilon) = \|x_0-x_\epsilon\|_2^2 }$$

---

### DeepFool

DeepFool 不直接追求一个固定：

$$\epsilon$$

而是：

> 找到离 decision boundary 最近的位置。

它不断：

```
当前位置
 ↓
估计 decision boundary
 ↓
走最短距离
 ↓
重新估计
 ↓
...
```

因此主要目标：

$$min\|x_\epsilon-x_0\|_2$$

subject to：

$$f(x_\epsilon)\neq y_0$$

---

## 四十二、C&W

Carlini & Wagner。

核心思想：

> 将 adversarial attack 写成一个可微优化问题。

例如：

$$\min_x \|x-x_0\|_2^2 + c\cdot f(x)4$$

其中：

$$f(x)$$

负责推动模型进入 target class。

论文指出 C&W-$\ell_2$ 可以使用：

- cross entropy
- hinge loss

然后用梯度优化。

---

## 四十三、Learning-based Attack

前面都是：

> 每张图单独优化。

Learning-based 则训练一个 generator。

---

## Eq.(10)

$$\boxed{ x_\epsilon=g_\theta(x_0) }$$

或者：

$$\boxed{ x_\epsilon=x_0+g_\theta(x_0) } \tag{10}$$

即：

```
image
 ↓
generator
 ↓
perturbation
 ↓
adversarial image
```

---

## 四十四、Eq.(11)

训练 generator：

$$\boxed{ \arg\min_\theta \frac1n \sum_i D_x(x_0^{(i)},g_\theta(x_0^{(i)})) + \lambda_{IB} L_{IB} (f_{w_0}(g_\theta(x_0^{(i)})),y_\epsilon^{(i)}) } \tag{11}$$

意思：

> 学一个模型，让它直接生成 adversarial example。

典型：

- advGAN
- PhysGAN
- CGAN-Adv
- AP-GAN
- AC-GAN
- MAG-GAN
- LG-GAN
- AdvFaces

---

## 四十五、Physical Adversarial Attack

数字攻击：

```
digital image
 ↓
digital model
```

物理攻击：

```
digital perturbation
 ↓
print / object / light
 ↓
real world
 ↓
camera
 ↓
digital image
 ↓
model
```

论文定义两个重要变换：

$$D2P$$

Digital → Physical

以及：

$$P2D$$

Physical → Digital。

因此物理攻击需要解决：

> perturbation 在现实世界中还能不能有效？

---

## 四十六、NPS

Non-Printability Score。

打印机不能打印所有 RGB。

因此：

$$RGB_{digital} \neq RGB_{physical}$$

NPS 的目的就是：

> 让 adversarial pattern 更容易被真实打印。

---
## 四十七、EOT

这是物理攻击最重要的概念之一。

Expectation over Transformation：

$$\boxed{ \mathbb E_{t\sim T} [ L(t(x_\epsilon),y_\epsilon) ] }$$

意思：

> 不要只让攻击在一个固定图像上成功，而是在各种现实变换下都成功。

例如：

- rotation
- scale
- translation
- lighting
- camera angle
- blur
- color

因此：

$$\min_\epsilon \mathbb E_{t\sim T} L(f(t(x+\epsilon)),y)$$

---

## 四十八、物理攻击代表方法

论文提到：

### RP2

加入 physical transformations。

### ShapeShifter

针对 Faster R-CNN。

### UPC

Universal Physical Camouflage。

### AdvHat

把 adversarial sticker 放在帽子上。

### AdvT-shirt

把 adversarial pattern 放在衣服上。

### PhysGAN

使用视频片段攻击 autonomous driving。

### SLAP

利用 projector 把 perturbation 投射到物体上。

### CAA

Curriculum Adversarial Attack。

模拟：

- printer error
- camera error
- sticker deformation
- face pose
- lighting

---

## 四十九、Eq.(12)：Universal Adversarial Perturbation

论文定义：

### Sample-specific

$$x_\epsilon^{(i)}-x_0^{(i)} \neq x_\epsilon^{(j)}-x_0^{(j)}$$

### Sample-agnostic

$$x_\epsilon^{(i)}-x_0^{(i)} = x_\epsilon^{(j)}-x_0^{(j)}$$

即：

$$\boxed{ \epsilon^{(i)}=\epsilon^{(j)} } \tag{12}$$

这种就是：

> Universal Adversarial Perturbation，UAP。

---

## 五十、Eq.(13)：UAP

$$\boxed{ \arg\min_\epsilon \|\epsilon\|_2^2 + \frac{\lambda_{CB}}{m} \sum_{i=1}^{n} L_{CB} (f_{w_0}(x_0^{(i)}+\epsilon),y_0^{(i)}) } \tag{13}$$

这里最重要的是：

> 一个 $\epsilon$，攻击很多图片。

而不是：

$$\epsilon_1,\epsilon_2,\epsilon_3...$$

每张图片一个。

---

### CD-UAP

Class Discriminative UAP。

只攻击：

```
某几个 class
```

而：

```
其他 class
```

尽量不受影响。

## 五十一、Data-free UAP

不需要正常训练数据。

典型：

### Fast Feature Fool

让 perturbation 激活 convolution layers。

### GD-UAP

最大化各层 activation norms。

### PD-UA

最大化：

- epistemic uncertainty
- aleatoric uncertainty

---

## 五十二、Eq.(14)：Additive vs Non-additive

$$\boxed{ x_\epsilon=x_0+\epsilon }$$

叫：

> additive perturbation

另一种：

$$\boxed{ x_\epsilon=h(x_0) }$$

叫：

> non-additive perturbation

即：

> 不一定是“加噪声”，也可以改变整个图像的 style、geometry 等。

---

## 五十三、Non-additive：Geometric Attack

例如：

### Manifool

把图片沿着：

> transformation manifold

移动，同时逼近 decision boundary。

### stAdv

优化：

> local spatial distortion。

可以理解为：

```
猫
 ↓
轻微扭曲
 ↓
模型认为是狗
```

---

## 五十四、Style Attack

### ReColorAdv

直接改变：

> color。

可以在：

- RGB
- CIELUV

空间中优化。

### AdvCam

把：

$$
style\ loss
$$

和：

$$
adversarial\ loss
$$

结合。

目标：

> 图片看起来像正常的自然风格，但模型被攻击。

---

## 五十五、Eq.(15)：Sparse Attack

Dense：

$$\boxed{ \|x_\epsilon-x_0\|_0=|x_0| }$$

意味着：

> 几乎所有像素都可能变化。

Sparse：

$$\boxed{ \|x_\epsilon-x_0\|_0<|x_0| } \tag{15}$$

意味着：

> 只改变少数像素。

---

## 五十六、Sparse Attack 方法

### LaVAN

手工选择一个：

> visible local patch。

---

### JSMA

Jacobian-based Saliency Map Attack。

计算：

> 哪些 pixel 对目标类别最重要。

然后优先修改 saliency 最大的位置。

---

### CornerSearch

根据模型输出变化排序 pixel，然后搜索。

---

### LocSearchAdv

在当前被攻击 pixel 附近的小区域继续搜索。

---

### One-Pixel Attack

只攻击：

$$
1
$$
个 pixel。

把：

$$(x,y,R,G,B)$$

拼成优化变量，然后用 Differential Evolution 搜索。

---

### Pointwise

先得到 adversarial image。

然后：

> 一个一个 pixel 删除。

只要删掉后仍然 attack 成功，就保留删除。

目标：

$$\min\|x_\epsilon-x_0\|_0$$

---

### GreedyFool

两阶段：

```
增加 perturbation pixels
        ↓
删除不必要 pixels
```

---

### SparseFool

使用：

$$\ell_1$$

鼓励 sparse。

---

### SAPF

非常有意思。

把：

$$\epsilon_i = m_i\cdot a_i$$

其中：

- $m_i\in\{0,1\}$：这个 pixel 攻不攻击
- $a_i$：攻击幅度

因此：

> 同时优化“位置”和“幅度”。

最终变成：

> Mixed Integer Programming。

再使用：

$$\ell_p\text{-Box ADMM}$$

求解。

---

## 五十七、Eq.(16)：Untargeted vs Targeted

这是你以后做实验必须搞懂的。

$$\boxed{ L_{IB} = -L(f(x_\epsilon),y_0) }$$

是：

> Untargeted Attack

目标：

$$f(x_\epsilon)\neq y_0$$

---

Targeted：

$$\boxed{ L_{IB} = L(f(x_\epsilon),y_\epsilon) }$$

目标：

$$f(x_\epsilon)=y_\epsilon$$

所以：

```
Untargeted：
猫 → 任何不是猫的类别

Targeted：
猫 → 指定狗
```

论文特别指出：

> targeted attack 比 untargeted attack 难，尤其是 black-box 和 transfer attack。

---

## 五十八、Structured Output Attack

普通 classification：

$$x\rightarrow y$$

是 factorized output。

但是：

```
image → caption
audio → sentence
```

属于：

> structured output。

---

### Complete Target

例如：

```
A dog is running in the park.
```

整个句子都指定。

---

### Partial Target

只要求：

```
必须出现 “dog”
```

或者：

```
第3个词必须是 dog
```

典型：

#### Target Keyword Attack

只控制 keyword。

#### Exact Structured Attack

控制：

> keyword + position。

---

## 五十九、Black-box Attack

这里进入你未来研究也非常重要的部分。

分：

```
Score-based
Decision-based
```

---

## 六十、Eq.(17)：Score-based

定义：

$$B_{x_0,\epsilon} = \{x' | \|x'-x_0\|_p\le\epsilon \}$$

然后：

$$\boxed{ \min_{x_\epsilon\in Z_x} \delta(x_\epsilon\in B_{x_0,\epsilon}) + \max(0,\Delta) } \tag{17}$$

这里：

$$\delta(a) = \begin{cases} 0,&a=true\\ \infty,&a=false \end{cases}$$

意思：

> 必须满足 epsilon constraint。

然后再优化：

$$\max(0,\Delta)$$

也就是 hinge loss。

---

## 六十一、Eq.(18)

Untargeted：

$$
\boxed{ \Delta = f(x_\epsilon,y_0) - \max_{j\neq y_0}f(x_\epsilon,j) }
$$

只要：

$$
Delta\le0
$$

就成功。

---

Targeted：

$$
\boxed{ \Delta = \max_{j\neq y_\epsilon}f(x_\epsilon,j) - f(x_\epsilon,y_\epsilon) }
$$

只要：

$$
\Delta\le0
$$

目标类别就超过其他类别。

因此：

$$
\max(0,\Delta)=0
$$

就意味着：

> attack success。

---

## 六十二、Score-based Random Search

### SimBA

随机选一个方向：

$$
x_{t+1}=x_t\pm\alpha q
$$

其中 $q$ 是 basis vector。

---

### ECO

在：

$$
\ell_\infty
$$

ball 的顶点中搜索。

---

### Square Attack

不是全图随机。

而是：

> 每次随机选择一个 square patch。

---

### PRFA

Square Attack 扩展到 object detection：

> 同时攻击多个 local patches。

---

### PPBA

利用：

$$
DCT/IDCT
$$

把搜索空间降低到：

> low-dimensional + low-frequency subspace。

---

## 六十三、Score-based Gradient Estimation

黑盒没有真实 gradient。

怎么办？

> 用 query 估计 gradient。

---

## NES

Natural Evolution Strategy。

随机采：

$$
u_i\sim\mathcal N(0,I)
$$

通过多个 query：

$$
f(x+\sigma u_i)
$$

估计：

$$
\nabla_x f(x)
$$

---

### Bandit

进一步利用：

- spatial prior
- temporal prior

即：

```
相邻 pixel gradient 相似
+
相邻 iteration gradient 相似
```

---

### NAttack

学习 adversarial distribution。

---

### AdvFlow

用 normalizing flow 替代 Gaussian distribution：

$$
z\rightarrow x_\epsilon
$$

让生成的 adversarial samples 更符合正常数据分布。

---

### ZO-signSGD

不估计完整 gradient：

> 只估计 gradient sign。

---

### SignHunter

采用 divide-and-conquer：

```
整个图像
 ↓
前半/后半
 ↓
四分之一
 ↓
更小区域
```

逐渐确定 gradient sign。

---

## 六十四、Combination-based Black-box

这是非常重要的一个方向。

不是：

```
只有 query
```

而是：

```
surrogate model
+
query feedback
```

---

### Subspace Attack

利用多个 surrogate model 的 gradient：

$$
g_1,g_2,\cdots,g_k
$$

形成 subspace。

在这个 subspace 里搜索。

---

### P-RGF

Prior-guided Random Gradient-Free。

：

$$
gradient\ prior + random\ directions
$$

---

### TREMBA

训练 autoencoder：

```
low-dimensional latent
 ↓
decoder
 ↓
adversarial perturbation
```

因此黑盒搜索空间变小。

---

### CG-Attack

学习：

$$
P(x_\epsilon|x_0)
$$

即：

> 条件 adversarial distribution。

---

### MSA

Meta Square Attack。

学习：

> square size、location、color 等超参数的分布。

---

### EigenBA

如果：

```
surrogate
target
```

共享 backbone。

就利用：

> Jacobian 的 singular vectors

寻找方向。

---

## 六十五、Adaptive Surrogate

这比普通黑盒更进一步：

```
target query
 ↓
更新 surrogate
 ↓
再次攻击
 ↓
再次 query
 ↓
继续更新
```

典型：

### Hybrid Batch Attack

用 target query 的结果更新 surrogate。

### LeBA

结合：

- SimBA
- TIMI

并通过高阶 gradient approximation 更新 surrogate。

### CSEA

ensemble 多个 surrogate：

$$
f_{surrogate} = \sum_i\alpha_i f_i
$$

并根据 query 调整：

$$
\alpha_i
$$

### Simulator Attack

meta learning 学一个：

> generalized simulator。

---

## 六十六、Decision-based Attack

比 score-based 更严格。

你只能得到：

```
cat
```

而不是：

```
cat 0.82
dog 0.13
bird 0.05
```

因此：

> 只能知道 decision。

---

## 六十七、Eq.(19)

$$
\boxed{ \min_{x_\epsilon\in Z_x} D_x(x_0,x_\epsilon) + \delta \left( C(f_{w_0}(x_\epsilon),y)=1 \right) } \tag{19}
$$

这里：

$$
C
$$

就是 attack success condition。

---

## 六十八、Eq.(20)

Untargeted：

$$
\boxed{ C(f_{w_0}(x_\epsilon),y) = I(f(x_\epsilon;w)\neq y_0) }
$$

Targeted：

$$
\boxed{ C(f_{w_0}(x_\epsilon),y) = I(f(x_\epsilon;w)=y_\epsilon) } \tag{20}
$$

其中：

$$
I(a)= \begin{cases} 1,&a=true\\ 0,&a=false \end{cases}
$$

即：

> 只告诉你 attack 成功还是失败。

---

## 六十九、Decision-based 方法

### Boundary Attack

在 decision boundary 附近随机搜索。

---

### Evolutionary Attack

使用 evolutionary strategy 调整：

- direction
- step size。

---

### CISA

Customized Iteration and Sampling。

使用 transfer perturbation 作为 initial point。

---

### GeoDA

假设 decision boundary：

> curvature relatively low。

然后估计：

> boundary normal direction。

---

### SFA

Sign Flip Attack。

在 $\ell_\infty$ ball 表面随机：

> sign flip。

---

### RayS

沿着：

> ray

搜索，并通过 binary search 确定 step size。

---

## 七十、Gradient-estimation Decision Attack

### Opt-attack

将 black-box attack 转成：

> continuous optimization。

---

### Sign-OPT

不估计完整 gradient：

> 只估计 gradient sign。

---

### HopSkipJumpAttack

利用：

> Monte Carlo

估计 decision boundary 附近 gradient。

---

### QEBA

在：

> low-dimensional subspace

估计 gradient，而不是原始高维空间。

---

### qFool

利用 decision boundary 附近低 curvature 的性质：

> 用邻近点估计 gradient。

## 七十一、现在进入最与你研究方向相关的：Transfer-based Attack

Transfer Attack 的核心：

> **我不知道目标模型，但我希望在 surrogate model 上生成的 adversarial example 可以攻击目标模型。**

也就是：

$$
f_{surrogate}(x_\epsilon)\neq y
$$

希望：

$$
f_{target}(x_\epsilon)\neq y
$$

---

## 七十二、Example-level Transferability

这是一个更细的新概念。

不是：

> 不同模型之间 transfer。

而是：

> 不同 benign examples 的 adversarial perturbations 之间也可能存在共同规律。

论文给出五种假设。

---

## 七十三、Eq.(21)：Common Perturbation

$$
\boxed{ f(x_0^{(i)}+\epsilon)=y_\epsilon^{(i)}, \quad f(x_0^{(j)}+\epsilon)=y_\epsilon^{(j)} } \tag{21}
$$

即：

> 不同图片可以共享一个 adversarial perturbation。

这就是：

$$
UAP
$$

的理论基础。

---

## 七十四、Eq.(22)：Common Parametric Generator

$$
\boxed{ f(x_0^{(i)}+g_\theta(x_0^{(i)})) = y_\epsilon^{(i)} }
$$

同时：

$$
f(x_0^{(j)}+g_\theta(x_0^{(j)})) = y_\epsilon^{(j)} \tag{22}
$$

意思：

> 不同图片的 adversarial perturbation 可以由同一个 generator 产生。

典型：

### GAP

Generative Adversarial Perturbations。

训练 generator。

---

## 七十五、Eq.(23)：Common Marginal Distribution

$$
\boxed{ x_\epsilon^{(i)}, x_\epsilon^{(j)} \sim P(x_\epsilon) } \tag{23}
$$

也就是说：

> 不要求 perturbation 完全一样，只要求 adversarial samples 来自相同 distribution。

典型：

### AdvFlow

用 normalizing flow 建模：

$$
P(x_\epsilon)
$$

---

## 七十六、Eq.(24)：Conditional Distribution

$$
\boxed{ x_\epsilon^{(i)}, x_\epsilon^{(j)} \sim P(x_\epsilon|x_0) } \tag{24}
$$

也就是：

> 给定 benign image $x_0$，adversarial example 遵循某个条件分布。

典型：

### CG-Attack

学习：

$$
P(x_\epsilon|x_0)
$$

使用 conditional Glow。

---

### MCG

进一步认为不同样本的 conditional distribution：

> 类似但不是完全一样。

然后使用 meta learning 快速适配新样本。

---

## 七十七、Eq.(25)：Common Gradient

$$
\boxed{ \frac{ \partial J_{G_1}(x_0^{(i)},y_\epsilon^{(i)}) }{ \partial x^{(i)} } \approx \frac{ \partial J_{G_2}(x_0^{(j)},y_\epsilon^{(j)}) }{ \partial x^{(j)} } } \tag{25}
$$

这就很接近你研究的：

> **迁移性为什么存在？**

意思：

> 不同样本的攻击梯度可能具有某种共同结构。

典型：

### Meta Attack

训练 meta model：

> 直接生成 gradient。

然后适应新的：

- sample
- model。

---

## 七十八、Model-level Transferability

这才是通常我们说的：

> Transfer-based adversarial attack。

即：

$$
f_s \rightarrow x_{adv} \rightarrow f_t
$$

其中：

- $f_s$：surrogate
- $f_t$：target

---

## 七十九、Data Perspective

核心：

> 不要让 adversarial example 过拟合 surrogate。

---

### DI²-FGSM

Diverse Input Iterative FGSM。

每次更新前：

```
原图
 ↓
random resize
 ↓
random padding
 ↓
gradient
```

因此：

> attack 不容易只针对 surrogate 的某一个输入分布。

---

其他：

- random scaling
- random mixup
- random translation
- Gaussian noise

---

### ODI

Object-based Diverse Input。

将 adversarial sample 放到：

> 不同 3D objects + rendering environments

中产生 diverse input。

---

### S²I-FGSM

在 frequency domain 使用：

DCTDCT

和：

IDCTIDCT

构造 diversity。

---

## 八十、Optimization Perspective

这个与你的研究最相关。

论文指出一个非常重要的现象：

### FGSM

一步：

> underfitting surrogate。

### I-FGSM

很多步：

> overfitting surrogate。

所以：

```
FGSM
  ↓
太简单

I-FGSM
  ↓
太容易过拟合

MI-FGSM / NI-FGSM
  ↓
寻找中间平衡
```

---

## 八十一、VMI-FGSM

Variance Tuning。

把当前点附近 gradient variance 加进 momentum。

目的：

> 避免当前 gradient 太局部。

---

## 八十二、EMI-FGSM

Expectation over Multiple Inputs。

对当前点附近多个样本：

$$
x+\delta_1, x+\delta_2, ...
$$

计算 gradient，再平均。

即：

$$
g= \frac1N \sum_i \nabla_xL(x+\delta_i)
$$

目的：

> 更稳定、更 transferable。

---

## 八十三、SVRE-MI-FGSM

Stochastic Variance Reduced Ensemble。

把：

> variance reduction

和：

> ensemble

结合。

目标：

> 降低 gradient variance，提高 transferability。

---

## 八十四、PI-FGSM

在 NI-FGSM 基础上改变 momentum：

> 只积累 local gradient。

---

## 八十五、SGM

Skip Gradient Method。

针对 ResNet：

> skip connection 对 transferability 很重要。

所以调整 backward gradient：

> 更多关注 skip connection。

---

## 八十六、LinBP

Linear Backpropagation。

正常 ReLU：

$$
ReLU(x)=\max(0,x)
$$

LinBP 在 backward 时：

> 不按普通 ReLU 那样传播。

而是近似线性传播。

目的：

> 获得更 transferable gradient。

---

## 八十七、Loss Perspective

不再简单：

$$
CE
$$

而是修改 attack loss。

---

### FDA

Feature Distribution Attack。

训练一个 auxiliary classifier：

$$
feature\rightarrow target
$$

然后最大化 target probability。

---

### ILAP

Intermediate Level Attack Projection。

最大化：

$$
feature(x_{adv})-feature(x)
$$

同时保持：

> 和已有 adversarial example 的 feature 接近。

---

### FIA

Feature Importance-aware Attack。

先计算：

> intermediate feature importance。

然后攻击重要 feature。

---

### NAA

Neuron Attribution Attack。

进一步从：

> neuron attribution

确定哪些 neuron 更重要。

---

### IR

Interaction-Reduced Attack。

论文观察：

> adversarial perturbation 内部 interaction 越强，transferability 越差。

所以加入 interaction loss。

---

### RAP

Reverse Adversarial Perturbation。

采用：

$$
\min-\max
$$

结构，在 adversarial sample 上再加 reverse perturbation。

目的是：

> 找 flat local minima。

因为：

> flat region 对模型变化更稳定，所以 transferability 更好。

---

## 八十八、Surrogate Model Perspective

### Robust Surrogate

一个稍微 robust 的 surrogate：

> 可能产生更 transferable 的 adversarial examples。

---

### Ghost Networks

对一个 surrogate：

```
dropout
+
random residual weights
```

产生多个 surrogate。

然后：

> 每一步随机选择一个 surrogate。

---

### IAA

Intrinsic Adversarial Attack。

假设：

> ground-truth data distribution 的 low-density region 更容易 transfer。

因此让 gradient 朝 low-density region 走。

---

### DRA

Distribution-Relevant Attack。

调整 surrogate：

> 使其 gradient 更接近 ground-truth data distribution。

---

## 八十九、Diffusion Model Attack

论文还把攻击扩展到生成模型。

---

### BadDiffusion

在 diffusion process 中植入：

$$
trigger \rightarrow target\ image
$$

正常输入：

> 正常生成。

触发 trigger：

> 生成攻击者指定结果。

---

### TrojDiff

修改 diffusion transition：

$$
target\ distribution \rightarrow Gaussian
$$

让 trigger 激活特定目标 distribution。

---

### TPA / TAA

针对 text-to-image diffusion 的：

> pretrained text encoder。

在 prompt 中加入：

> 单个 character trigger。

然后让模型生成：

> predefined attribute / hidden description。

---

### VillanDiffusion

统一的 diffusion backdoor framework。

---

## 九十、LLM Attack

论文提到的主要方法：

### Poisoned Prompt Tuning

在 poisoned dataset 上训练 soft prompt。

---

### PromptAttack

通过 gradient search：

> 自动寻找 discrete tokens。

---

### PromptInject

两种：

#### Goal Hijacking

把原 prompt 的目标：

$$
Goal_A
$$

改成：

$$
Goal_B
$$

#### Prompt Leaking

诱导模型：

> 输出原始 system prompt。

---

### BadPrompt

攻击 continuous prompt。

---

### BadGPT

攻击：

> RL fine-tuning 的 reward model。

---

### HOUYI

将：

> SQL injection / XSS

的思想迁移到 prompt injection。

---

### PoisonPrompt

bi-level optimization：

```
trigger optimization
+
prompt tuning
```

一起进行。

---

### AutoDAN

自动生成：

> interpretable jailbreak prompts。

---

### TrojLLM

black-box：

```
API query
 ↓
trigger discovery
 ↓
progressive prompt poisoning
```

不需要：

- architecture
- weights
- gradient。

---

## 九十一、论文最后真正想告诉你的是什么？

我把论文的 Discussion 浓缩成几个非常重要的结论。

---

### 1. Backdoor 的发展趋势

论文认为未来：

#### Trigger

从：

$$
visible \rightarrow invisible
$$

从：

$$
non\text{-}semantic \rightarrow semantic
$$

从：

$$
manual \rightarrow learnable
$$

---

#### Fusion

从：

$$
additive \rightarrow non\text{-}additive
$$

从：

$$
static \rightarrow dynamic
$$

从：

$$
sample\text{-}agnostic \rightarrow sample\text{-}specific
$$

原因：

> 更灵活，更难防御。

---

## 九十二、Weight Attack 的现状

论文很谨慎。

它指出：

> 大部分 Weight Attack 仍然停留在理论研究。

真正现实系统中的成功攻击还比较少。

最大原因：

> 需要精确控制 memory bit。

例如：

- Rowhammer
- Laser Beam

需要硬件知识和特殊设备。

所以这是一个：

> **AI + computer architecture + hardware security**

交叉研究方向。

---

## 九十三、White-box Attack 已经比较成熟

论文的判断非常明确：

> White-box adversarial attack 已经有大量成熟方法。

现在很少出现真正全新的：

> 基础白盒攻击范式。

更多时候：

```
FGSM / PGD
      ↓
迁移攻击
      ↓
Backdoor
      ↓
Adversarial Training
      ↓
Privacy
```

也就是说：

> **白盒 attack 现在更多是工具，而不是最终研究目的。**

---

## 九十四、Black-box Attack 的研究空间

论文认为：

### Score-based

已经非常强。

尤其：

> surrogate + query feedback

的 combination-based 方法。

甚至一些简单场景：

$$
\text{median queries}=1
$$

同时：

$$
ASR=100\%
$$

但是问题：

> 大部分是在没有防御的情况下测试。

因此未来应该：

> **Defense-aware Black-box Attack**

---

## 九十五、Decision-based 的问题

因为：

```
Score-based：
cat 0.8

Decision-based：
cat
```

Decision-based 信息少。

所以：

$$
Performance_{decision} < Performance_{score}
$$

论文认为未来值得研究：

> **Combination-based Decision Attack**

即：

$$
surrogate\ prior + query\ feedback
$$

---

## 九十六、Transfer Attack 是论文认为非常重要的方向

这一段你一定要认真看。

论文指出：

> Transfer-based attack 不要求知道目标模型。

所以现实威胁更高。

但是：

$$
P(\text{transfer})
$$

高度依赖：

```
surrogate dataset
+
target dataset
+
surrogate architecture
+
target architecture
```

如果相似：

> transfer 很好。

如果差异很大：

> transfer 很差。

---

## 九十七、论文认为真正没解决的问题是什么？

最重要的是：

> **我们还不知道 adversarial transferability 的内在理论原因。**

目前很多方法：

```
实验观察
 ↓
提出 heuristic
 ↓
transfer 提升
```

但是没有一个非常完整的理论解释：

$$
\boxed{ Why does adversarial transferability exist? }
$$

论文认为未来应该从：

- data distribution
- model architecture
- loss landscape
- decision surface
- generalization theory

等角度研究。

---

## 九十八、把这篇论文和你现在的研究方向联系起来

你现在研究的是：

> **基于多粒度特征对齐的迁移对抗攻击方法**

那么你应该重点关注论文中的这一条线：

```
Inference-time
      ↓
Adversarial Examples
      ↓
White-box
      ↓
Transfer-based
      ↓
Model-level Transferability
      ↓
Feature / Gradient / Distribution
```

其中最值得你研究的是：

### ① Gradient

$$
\nabla_xL
$$

代表：

- MI-FGSM
- VMI-FGSM
- EMI-FGSM
- NI-FGSM
- SGM
- LinBP
- RAP

---

### ② Feature

代表：

- FDA
- ILAP
- FIA
- NAA

这些已经非常接近：

> **Feature Alignment**

---

### ③ Distribution

代表：

- AdvFlow
- CG-Attack
- MCG
- DRA
- IAA

这里开始从：

$$
\text{single feature}
$$

走向：

$$
\text{feature distribution}
$$

---

### ④ Data distribution

论文认为：

> adversarial transferability 很可能和数据分布有关。

这与你的：

> 多粒度特征对齐

实际上可以形成一个非常自然的研究逻辑：

$$
\boxed{ Input \rightarrow Low-level\ feature \rightarrow Middle-level\ feature \rightarrow High-level\ semantic\ feature \rightarrow Distribution }
$$

然后研究：

$$
\text{multi-granularity alignment}
$$

是否能够：

$$
\boxed{ \text{reduce surrogate-specific overfitting} }
$$

从而：

$$
\boxed{ \text{improve adversarial transferability} }
$$

---

## 九十九、最后给你一张“整个领域知识树”

你现在可以把这篇论文压缩成：

```
Adversarial Machine Learning
│
├── 1. Backdoor Attack
│   │
│   ├── Data Poisoning
│   │   ├── Trigger
│   │   │   ├── Visible / Invisible
│   │   │   ├── Semantic / Non-semantic
│   │   │   ├── Manual / Learnable
│   │   │   └── Digital / Physical
│   │   │
│   │   ├── Sample Selection
│   │   │   ├── Random
│   │   │   └── Non-random
│   │   │
│   │   ├── Fusion
│   │   │   ├── Additive
│   │   │   ├── Non-additive
│   │   │   ├── Static
│   │   │   ├── Dynamic
│   │   │   ├── Sample-agnostic
│   │   │   └── Sample-specific
│   │   │
│   │   └── Target
│   │       ├── Single-target
│   │       ├── Multi-target
│   │       ├── Label-inconsistent
│   │       └── Label-consistent
│   │
│   └── Training-controllable
│       ├── One-stage / Two-stage
│       ├── Full / Partial data
│       ├── Full / Partial training control
│       └── Loss / Algorithm / Order
│
├── 2. Weight Attack
│   │
│   ├── Post-training
│   │   ├── White-box
│   │   │   ├── SBA
│   │   │   ├── GDA
│   │   │   └── ADMM
│   │   └── Black-box
│   │       └── SRA
│   │
│   └── Deployment
│       ├── Without trigger
│       │   ├── T-BFA
│       │   ├── TA-LBF
│       │   └── Robustness Attack
│       └── With trigger
│           ├── Trojaning
│           ├── TBT
│           ├── ProFlip
│           ├── AWP
│           ├── Anchoring
│           ├── SRA
│           └── TSA
│
└── 3. Adversarial Examples
    │
    ├── White-box
    │   ├── ℓ∞
    │   │   ├── FGSM
    │   │   ├── I-FGSM / PGD
    │   │   ├── MI-FGSM
    │   │   ├── NI-FGSM
    │   │   └── Auto-PGD
    │   │
    │   ├── ℓ2
    │   │   ├── DeepFool
    │   │   └── C&W
    │   │
    │   ├── Learning-based
    │   │   └── GAN-based
    │   │
    │   ├── Physical
    │   │   ├── RP2
    │   │   ├── ShapeShifter
    │   │   ├── UPC
    │   │   ├── AdvHat
    │   │   └── PhysGAN
    │   │
    │   ├── Universal
    │   │   ├── UAP
    │   │   ├── CD-UAP
    │   │   ├── GD-UAP
    │   │   └── PD-UA
    │   │
    │   ├── Non-additive
    │   │   ├── Manifool
    │   │   ├── stAdv
    │   │   ├── ReColorAdv
    │   │   └── AdvCam
    │   │
    │   └── Sparse
    │       ├── JSMA
    │       ├── One-Pixel
    │       ├── Pointwise
    │       ├── GreedyFool
    │       ├── SparseFool
    │       └── SAPF
    │
    ├── Black-box
    │   ├── Score-based
    │   │   ├── SimBA
    │   │   ├── Square
    │   │   ├── NES
    │   │   ├── Bandit
    │   │   ├── NAttack
    │   │   └── SignHunter
    │   │
    │   └── Decision-based
    │       ├── Boundary
    │       ├── GeoDA
    │       ├── RayS
    │       ├── Sign-OPT
    │       ├── HSJA
    │       └── QEBA
    │
    └── Transfer-based
        ├── Data
        │   ├── DI²-FGSM
        │   ├── Admix
        │   └── S²I-FGSM
        │
        ├── Optimization
        │   ├── MI-FGSM
        │   ├── VMI-FGSM
        │   ├── EMI-FGSM
        │   ├── NI-FGSM
        │   ├── SGM
        │   └── LinBP
        │
        ├── Feature
        │   ├── FDA
        │   ├── ILAP
        │   ├── FIA
        │   └── NAA
        │
        ├── Distribution
        │   ├── AdvFlow
        │   ├── CG-Attack
        │   ├── MCG
        │   └── DRA
        │
        └── Meta-learning
            ├── Meta Attack
            ├── MGAA
            └── MCG
```

这棵树基本就是这篇论文最重要的知识结构。论文自己也通过附录的 Tables VI–X，把各个方法进一步按照这些维度进行交叉归类。

---

## 一句话总结整篇论文

如果你以后导师问你：

> **“这篇 survey 到底讲了什么？”**

你可以这样回答：

> **这篇论文从机器学习生命周期角度重新定义了 Adversarial Machine Learning，将攻击统一为 Backdoor Attack、Weight Attack 和 Inference-time Adversarial Example 三种范式，并通过 Stealthiness、Benign Consistency 和 Adversarial Inconsistency 建立统一数学框架。对于 adversarial examples，又进一步按照 white-box、black-box 和 transfer-based 进行系统分类。论文最终认为，白盒攻击已经相对成熟，而未来更值得研究的是黑盒攻击、物理攻击以及尤其是 adversarial transferability；其中迁移性的理论机制仍缺乏统一解释，需要从数据分布、模型结构、特征表示、loss landscape 和 decision boundary 等角度深入研究。**

而对你现在的研究来说，**最值得精读的不是整篇论文所有 Backdoor 内容，而是第 VII 节的 Transfer-based Adversarial Examples，尤其是 Eq.(21)–Eq.(25)、MI-FGSM/VMI-FGSM/EMI-FGSM、SGM/LinBP、FDA/ILAP/FIA/NAA、AdvFlow/CG-Attack/MCG/DRA 这一条线**。这部分实际上已经把你现在研究的“多粒度特征对齐 + 迁移攻击”放进了一个非常清楚的研究谱系里。

## DS总结
这篇综述《Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective》提出了一个**统一的对抗机器学习（AML）定义和数学框架**，并按照机器学习系统的**生命周期**将攻击分为三大范式：**后门攻击**、**权重攻击**和**对抗样本**。文章系统梳理了每个范式下的具体攻击方法，并给出了详细的分类。以下是对全文的总结，涵盖文中提到的所有攻击方法（按分类整理，每个方法附简要说明）。

---

## 一、统一框架与三大范式

**AML定义**：基于三个条件——**隐蔽性（Stealthiness）**、**良性一致性（Benign Consistency）**、**对抗不一致性（Adversarial Inconsistency）**。统一优化目标为：

$$ \arg\min_{x_e,w_e} \mathcal{S} + \mathcal{C} + \mathcal{I} $$

不同范式通过具体化这些项得到。

**三大范式**（按生命周期阶段）：

1. **后门攻击**：发生在预训练、训练、推理阶段。攻击者通过操纵训练数据或训练过程，使模型在良性样本上表现正常，但在含触发器的样本上预测为目标标签。
    
2. **权重攻击**：发生在后训练、部署、推理阶段。攻击者直接修改模型参数（连续空间）或翻转比特（离散空间），使模型对特定样本预测错误。
    
3. **对抗样本**：仅发生在推理阶段。攻击者微调良性样本，使模型预测错误。
    

---

## 二、后门攻击

### 2.1 数据投毒型后门攻击（预训练阶段）

攻击者只能操纵训练数据集，无法控制训练过程。形式化：$x_e = g_3\big(g_1(\epsilon), g_2(D_0)\big),\quad y_e = g_4(y_0).$

#### 2.1.1 触发器生成（$g1$​）

- **可见 vs 不可见**：
    
    - **可见触发器**：BadNets（黄色方块/贴纸）、TrojanNN、以及3D点云中的可见点。
        
    - **不可见触发器**：
        
        - **Alpha blending**：Blended（将触发器透明融合）。
            
        - **数字隐写**：Li et al.（LSB替换）、SSBA（双循环自编码器）。
            
        - **对抗扰动**：AdvDoor（目标通用对抗扰动）、label-consistent attacks（如Shafahi et al.、Hidden Trigger）。
            
        - **轻微变换**：WaNet（图像扭曲）、DFST（风格迁移）。
            
- **非语义 vs 语义**：
    
    - **非语义**：大多数早期方法（如网格、随机噪声）。
        
    - **语义**：NLP中的特定词/句（如BadNL）、视觉中的“赛车条纹汽车”（如VSSC）。
        
- **手动设计 vs 可学习**：
    
    - **手动**：BadNets、Blended、SIG、随机噪声等。
        
    - **可学习**：Label-consistent attacks（优化触发器）、AdvDoor（优化通用扰动）。
        
- **数字 vs 物理**：
    
    - **数字**：绝大多数。
        
    - **物理**：Wenger et al.（人脸识别物理触发器）、PTB（物理变换）、Han et al.（车道检测）、VSSC。
        

#### 2.1.2 样本选择策略（$g2$​）

- **随机选择**：最常用。
    
- **非随机选择**：
    
    - FUS（基于遗忘事件过滤更新）
        
    - LPS（基于min-max的可学习选择）
        
    - FUS++（结合遗忘事件和曲率）
        
    - RD（表示距离评分）
        
    - Wu et al.（基于置信度的贪婪搜索）
        
    - PFS（代理自由策略，基于余弦相似度）
        

#### 2.1.3 触发器融合策略（$g3$​）

- **加性 vs 非加性**：
    
    - **加性**：大多数方法（$x_e = \alpha x_0 + (1-\alpha)g_1$）。
        
    - **非加性**：
        
        - 颜色/风格/属性变换：FaceHack（面部属性）、DFTS（图像风格）。
            
        - 空间变换：WaNet（基于扭曲场）。
            
- **样本无关 vs 样本特定**：
    
    - **样本无关**：大多数（触发器固定）。
        
    - **样本特定**：Li et al.（LSB隐写）、SSBA（自编码器隐写）、WaNet、DFTS、Poison Ink（边缘着色）。
        
- **静态 vs 动态**：
    
    - **静态**：BadNets、Blended、SIG。
        
    - **动态**：Random Backdoor（随机位置/模式）、DeHiB（随机位置）、Refool（随机反射参数）、Composite Attack（组合现有物体）。
        

#### 2.1.4 目标标签生成（g4g4​）

- **单目标 vs 多目标**：
    
    - **单目标**：大多数（all-to-one）。
        
    - **多目标**：
        
        - all-to-all：BadNets（同一触发器，不同源类→不同目标类）。
            
        - 多触发器多目标：c-BaN（条件触发器生成）、Marksman（任意目标类）。
            
- **标签一致 vs 标签不一致**：
    
    - **标签不一致**：大多数（源类≠目标类）。
        
    - **标签一致（clean-label）**：
        
        - Zhao et al.（视频识别）
            
        - Refool（反射图像）
            
        - Hidden Trigger（特征空间约束）
            
        - Invisible Poison（自编码器噪声）
            
        - Sleeper Agent（双层优化）
            

---

### 2.2 训练可控型后门攻击（训练阶段）

攻击者完全控制训练过程，可同时优化xexe​和wewe​。

#### 2.2.1 一阶段 vs 两阶段训练

- **两阶段**：先投毒再训练，可复用数据投毒策略。
    
- **一阶段**：联合优化$x_e$​和$w_e$​。
    
    - Input-Aware（联合学习触发器生成器和模型）
        
    - LIRA（双层优化，$\ell_\infty$约束）
        
    - WB（双层优化，Wasserstein距离）
        
    - Zhong et al.（U-Net触发器生成器 + 多分类分布）
        
    - BaN（随机噪声→触发器）
        
    - c-BaN（类条件触发器生成器）
        

#### 2.2.2 训练数据访问：全部 vs 部分

- **全部访问**：集中式学习。
    
- **部分访问**：联邦学习场景。
    
    - Bhagoji et al.（放大恶意更新）
        
    - Bagdasaryan et al.（模型替换）
        
    - DBA（分布式后门，各客户端局部触发器）
        
    - Wang et al.（边缘案例后门）
        
    - Chen et al.（联邦元学习）
        
    - Fung et al.（女巫攻击）
        
    - Neurotoxin（限制梯度重叠，持久后门）
        

#### 2.2.3 训练过程控制：全部 vs 部分

- **全部控制**：大多数训练可控攻击。
    
- **部分控制**：
    
    - 预训练+微调：Shen et al.（CLS token映射）、Poisoned Prompt Tuning（软提示）、LWP（逐层权重中毒）。
        
    - 训练+压缩：Tian et al.（压缩后门）。
        

#### 2.2.4 控制训练组件

- **控制损失**：Zhong et al.（增加隐蔽性项）。
    
- **控制算法**：
    
    - BpPAttack（量化+抖动，对比学习）
        
    - DFST（数据投毒与解毒迭代）
        
    - WaNet & Input-Aware（交叉触发训练）
        
- **控制样本顺序**：
    
    - Data-efficient backdoor（过滤更新）
        
    - BOB（控制批次顺序，模仿代理梯度）
        

---

## 三、权重攻击

### 3.1 后训练阶段：参数修改注入

攻击者直接修改良性模型参数$w_0 \rightarrow w_e$​。

#### 3.1.1 白盒权重攻击

- SBA（单偏置攻击，放大目标类偏置）
    
- GDA（梯度下降攻击，修改部分权重）
    
- Zhao et al.（ADMM框架，最小化参数修改且保持其他样本分类）
    

#### 3.1.2 黑盒权重攻击

- SRA（子网替换攻击，训练对触发器敏感的子网并替换）
    

### 3.2 部署阶段：比特翻转注入

攻击者通过物理故障注入翻转内存中的权重比特。

#### 3.2.1 无触发器比特翻转

- T-BFA（目标比特翻转攻击，二值优化搜索）
    
- TA-LBF（有限比特翻转目标攻击，$\ell_p$-Box ADMM）
    
- FSA（故障潜入攻击）
    
- Robustness attack（降低鲁棒性）
    

#### 3.2.2 有触发器比特翻转

- Trojaning Attack（三阶段：生成触发器→逆向训练数据→重训练）
    
- TBT（目标比特木马，识别重要神经元→生成触发器→翻转关键比特）
    
- ProFlip（类似TBT，不同算法）
    
- AWP（对抗权重扰动，微调权重注入后门）
    
- Anchoring Attack（锚定攻击，logit一致性）
    
- Handcrafted Backdoor（逐层权重修改）
    
- SRA（子网替换，仅需架构信息）
    
- TSA（触发样本攻击，混合整数规划）
    

---

## 四、对抗样本

### 4.1 白盒对抗样本

攻击者知道模型架构和权重。

#### 4.1.1 扰动类型

- **优化型 vs 学习型**：
    
    - **优化型**：
        
        - $\ell_\infty$ + 梯度符号：FGSM、I-FGSM（PGD）、MI-FGSM、NI-FGSM、AutoPGD。
            
        - $\ell_2$ + 梯度：DeepFool、C&W-ℓ2ℓ2​。
            
    - **学习型**：AdvGAN、PhysGAN、CGAN-Adv、AP-GAN、AC-GAN、MAG-GAN、LG-GAN、AdvFaces。
        
- **数字 vs 物理**：
    
    - **数字**：大多数。
        
    - **物理**：
        
        - D2P鲁棒：NPS、AGNs、SLAP、CVAE、CMML、CAA。
            
        - P2D鲁棒：EOT、RP2、TPS、UPC、ShapeShifter、随机渲染、ERCG、AdvPattern、AdvHat、PhysGAN、音频EOT。
            
- **样本特定 vs 样本无关（UAP）**：
    
    - **样本特定**：大多数。
        
    - **样本无关（UAP）**：
        
        - 数据依赖：Moosavi-Dezfooli et al.、CD-UAP。
            
        - 数据无关：Fast Feature Fool、GD-UAP、PD-UA。
            
- **加性 vs 非加性**：
    
    - **加性**：大多数。
        
    - **非加性**：
        
        - 几何变换：Manifold、stAdv、GUAP、Wasserstein距离。
            
        - 风格变换：ReColorAdv、AdvCam。
            
- **稠密 vs 稀疏**：
    
    - **稠密**：大多数。
        
    - **稀疏**：
        
        - 手动：LaVAN。
            
        - 启发式搜索：JSMA、CornerSearch、LocSearchAdv。
            
        - 优化型：One-Pixel、$\ell_0$+ADMM、Pointwise、GreedyFool、SparseFool、$\ell_{2,1}$​组稀疏、SAPF（混合整数规划）。
            

#### 4.1.2 输出类型

- **无目标 vs 有目标**：
    
    - 无目标：最小化真实类概率。
        
    - 有目标：最大化目标类概率。近期提升目标迁移性的方法：Poincare距离+三元组损失、TTP（GAN）、logit损失+充分迭代。
        
- **因子化 vs 结构化输出**：
    
    - **完整目标输出**：Dense Captioning、图像描述、音频到文本、序列替换。
        
    - **部分目标输出**：Target Keyword Attack、Exact Structured Attack（结构化SVM/EM）。
        

### 4.2 黑盒对抗样本

#### 4.2.1 基于分数（连续反馈）

- **查询型**：
    
    - **随机搜索**：SimBA、ECO、Square Attack、PRFA、PPBA。
        
    - **梯度估计**：NES、Bandit、NNAttack、AdvFlow、ZO-signSGD、SignHunter。
        
- **组合型**：
    
    - **静态代理先验**：Subspace Attack、P-RGF、TREMBA、CG-Attack、MSA、EigenBA。
        
    - **自适应代理先验**：Hybrid Batch Attack、LeBA、CSEA、BASES、Simulator Attack、MCG。
        

#### 4.2.2 基于决策（离散反馈）

- **随机搜索**：Boundary、Evolutionary、CISA、GeoDA、SFA、RayS。
    
- **梯度估计**：Ilyas et al.、Opt-attack、Sign-OPT、HopSkipJumpAttack、QEBA、qFool。
    

### 4.3 基于迁移的对抗样本

#### 4.3.1 样本级迁移性

- 共同扰动：UAP。
    
- 共同参数模型：GAP。
    
- 共同边缘分布：AdvFlow。
    
- 共同条件分布：CG-Attack、MCG。
    
- 相似梯度：Meta Attack。
    

#### 4.3.2 模型级迁移性

- **数据角度**：DI²-FGSM、随机缩放、Admix、随机平移、高斯噪声、ODI、S²I-FGSM。
    
- **优化角度**：MI-FGSM、VMI-FGSM、EMI-FGSM、SVRE-MI-FGSM、NI-FGSM、PI-FGSM、SGM、LinBP、MGAA。
    
- **损失角度**：FDA、ILAP、FIA、NAA、logit损失、IR、RAP。
    
- **代理模型角度**：稍微鲁棒的代理、Ghost Networks、IAA、DRA。
    

---

## 五、其他场景

### 5.1 扩散模型攻击

- BadDiffusion（在扩散过程中植入后门）
    
- TrojDiff（扩散目标分布偏移）
    
- TPA / TAA（攻击文本编码器，字符触发）
    
- VillanDiffusion（统一后门框架）
    

### 5.2 大语言模型攻击

- Xu et al.（预训练后门，微调后仍有效）
    
- PPT（毒化提示调优）
    
- PromptAttack（梯度搜索恶意提示）
    
- PromptInject（目标劫持/提示泄露）
    
- BadPrompt（连续提示后门，自适应触发器）
    
- BadGPT（攻击RLHF奖励模型）
    
- Wan et al.（指令调优数据投毒）
    
- HOUYI（SQL注入式提示注入）
    
- PoisonPrompt（双层优化后门）
    
- AutoDAN（自动生成可解释越狱提示）
    
- TrojLLM（黑盒API后门，通用触发器发现）
    

---

## 六、应用

- **后门用于版权保护**：Adi et al.（水印）、Sommer et al.（机器遗忘验证）、Li et al.（数据集版权）、ROW BACK（鲁棒水印）、MIB（成员推断）。
    
- **对抗样本用于隐私保护**：Oh et al.（博弈论框架）、P-FEAT（对抗训练隐私保护）、APF（端云协同隐私过滤）、AaaD（文本空间属性混淆）、AVIH（视觉信息隐藏）。
    

---

## 七、讨论与总结

- **后门攻击趋势**：未来倾向于不可见/语义/可学习触发器、非加性/动态/样本特定融合、多目标设置。
    
- **权重攻击**：仍停留在理论阶段，实际部署需物理故障注入技术（Rowhammer、激光束），跨学科合作是关键。
    
- **对抗样本**：
    
    - 白盒已成熟，多作为其他攻击工具。
        
    - 黑盒：组合型优于查询型，但需防御感知；决策型缺乏组合型方法；目标攻击仍具挑战。
        
    - 迁移型：跨数据集/架构迁移仍是难点，缺乏理论理解。
        
- **三大范式交互**：后门与对抗样本相互启发；防御设计需考虑跨范式鲁棒性。
    

---

**总结**：本文通过生命周期视角统一了AML领域，系统分类并回顾了后门攻击、权重攻击和对抗样本的几乎所有代表性方法，为研究者提供了全面的路线图。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/领域总览/对抗攻击 - 领域分类|对抗攻击 - 领域分类]]：对照生命周期视角与现有领域分类。
- [[30_对抗机器学习/后门攻击/Backdoor Attack - 后门攻击发展历程|Backdoor Attack - 后门攻击发展历程]]：从总览进入后门攻击的发展脉络。
- [[30_对抗机器学习/领域总览/Threat Model - 威胁模型详解|Threat Model - 威胁模型详解]]：补充攻击者能力与约束的统一描述。

## 论文出处

- [Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective](https://arxiv.org/abs/2302.09457)：arXiv，2023，预印本。 核实到 2023 年首发、2024 年修订的 arXiv 记录；正式期刊发表记录未核实。
- [Universal Adversarial Perturbations](https://openaccess.thecvf.com/content_cvpr_2017/html/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.html)：CVPR，2017，会议论文。

<!-- knowledge-links:end -->
