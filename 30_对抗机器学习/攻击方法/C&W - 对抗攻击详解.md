---
title: "C&W - 对抗攻击详解"
aliases:
  - "CW攻击方法详解"
分类: "攻击方法"
笔记类型: "概念笔记"
整理日期: "2026-10-27"
tags:
  - "知识库/笔记"
  - "主题/攻击方法"
状态: "已整理"
摘要: "围绕“C&W - 对抗攻击详解”，整理C&W 对抗攻击方法详解、C&W 是什么？、先统一符号。"
更新日期: "2026-10-30"
上级目录: "[[30_对抗机器学习/攻击方法/总结 - 攻击方法]]"
相关笔记:
  - "[[30_对抗机器学习/攻击方法/论文 - Explaining and Harnessing Adversarial Examples]]"
  - "[[30_对抗机器学习/防御与鲁棒性/Gradient Masking - 梯度掩蔽详解]]"
  - "[[30_对抗机器学习/通用与稀疏扰动/CornerSearch - 稀疏对抗攻击]]"
  - "[[30_对抗机器学习/通用与稀疏扰动/论文 - The Limitations of Deep Learning in Adversarial Settings]]"
论文条目:
  - "[[论文来源目录#fgsm]]"
  - "[[论文来源目录#pgd]]"
  - "[[论文来源目录#cw]]"
论文标题:
  - "Explaining and Harnessing Adversarial Examples"
  - "Towards Deep Learning Models Resistant to Adversarial Attacks"
  - "Towards Evaluating the Robustness of Neural Networks"
发表载体:
  - "ICLR"
  - "ICLR"
  - "IEEE Symposium on Security and Privacy (S&P)"
发表年份:
  - "2015"
  - "2018"
  - "2017"
发表类型:
  - "会议论文"
  - "会议论文"
  - "会议论文"
发表链接:
  - "https://iclr.cc/archive/www/2015.html"
  - "https://openreview.net/forum?id=rJzIBfZAb"
  - "https://doi.org/10.1109/SP.2017.49"
预印本链接:
  - "https://arxiv.org/abs/1412.6572"
  - "https://arxiv.org/abs/1706.06083"
  - "https://arxiv.org/abs/1608.04644"
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[30_对抗机器学习/攻击方法/总结 - 攻击方法|攻击方法总结]]

# C&W - 对抗攻击详解

# C&W 对抗攻击方法详解

> 阅读目标：理解 C&W 在优化什么、损失函数为什么这样写，以及如何通过迭代得到对抗样本。
>
> 本文以图像分类中的 C&W $L_2$ 为主。核心定义来自原论文与作者代码；数值例子、等价变形和梯度推导为本文的教学说明，不是论文实验结果。

## 1. C&W 是什么？

**C&W（Carlini & Wagner Attack）是一组基于优化的对抗攻击方法，其目标是在诱导模型误分类的同时尽量减小输入扰动。**

提出者是 Nicholas Carlini 和 David Wagner。论文 *Towards Evaluating the Robustness of Neural Networks* 于 2016 年发布预印本，并发表于 IEEE Symposium on Security and Privacy（IEEE S&P）2017。[1][2]

| 分类维度 | C&W 的位置 |
| --- | --- |
| 攻击发生阶段 | 测试／推理阶段的逃逸攻击 |
| 典型知识条件 | 白盒：能够获取模型输出与输入梯度 |
| 求解方式 | 迭代优化输入 |
| 攻击目标 | 原论文重点研究有目标攻击；作者代码也支持无目标攻击 |
| 扰动度量 | 原论文包含 $L_0$、$L_2$、$L_\infty$ 三种版本 |

原论文的重要贡献之一是攻破防御蒸馏（Defensive Distillation），说明防御挡住某些攻击，并不足以证明模型鲁棒。[1]

## 2. 先统一符号

假设输入像素缩放至 $[0,1]$，模型完成 $K$ 分类。

| 符号 | 含义 |
| --- | --- |
| $x\in[0,1]^n$ | 原始输入，展开后有 $n$ 个标量分量 |
| $y$ | 原始输入的真实类别；下文默认原图分类正确 |
| $x'$ | 对抗输入 |
| $\delta=x'-x$ | 添加的扰动 |
| $t\ne y$ | 有目标攻击指定的错误类别 |
| $Z_i(x')$ | 模型在 Softmax 之前对类别 $i$ 输出的分数，即 logit |
| $C(x')=\arg\max_i Z_i(x')$ | 模型最终预测的类别 |
| $c>0$ | 扰动项与攻击项之间的权衡系数 |
| $\kappa\ge0$ | 希望目标类别领先的 logit 间隔 |
| $w$ | 通过变量替换引入的可优化变量 |

**攻击期间固定模型参数，改变的是输入。**反向传播经过模型，是为了计算输入该如何修改。

## 3. 从攻击目标得到优化问题

以“原图是猫，希望模型判为狗”为例，令 $t$ 表示狗。有目标攻击希望同时满足：

1. 图片修改尽量小；
2. 模型预测为指定类别 $t$；
3. 修改后像素仍然合法。

可以写成：

$$
\begin{aligned}
\min_{x'}\quad &\|x'-x\|_2^2\\
\text{s.t.}\quad &C(x')=t,\\
&x'\in[0,1]^n.
\end{aligned}
$$

其中：

$$
\|x'-x\|_2^2=\sum_{j=1}^{n}(x'_j-x_j)^2.
$$

例如，两个分量分别改动 $0.01$ 和 $-0.02$，平方距离为 $0.0005$，实际 $L_2$ 距离为 $\sqrt{0.0005}\approx0.02236$。

平方距离与距离具有相同的大小排序，但不能在带权损失中随意互换而保持所有结果不变，因为它们的梯度和与攻击项的权衡关系不同。

直接优化 $C(x')=t$ 比较困难：$\arg\max$ 输出的是离散类别。因此，C&W 用基于 logits 的损失 $f$ 表达攻击目标，求解：[1]

$$
\boxed{\min_{x'\in[0,1]^n}\ \|x'-x\|_2^2+c\,f(x')}
$$

第一项把图片往原图拉，第二项把预测往目标类别推。它们共同决定最终扰动。

## 4. 攻击损失为什么这样写？

### 4.1 先定义“竞争者减目标”的分差

令：

$$
m(x')=\max_{i\ne t}Z_i(x')-Z_t(x').
$$

它先找出所有非目标类别中分数最高的类别，再减去目标类别分数。

- $m>0$：还有其他类别的分数比目标类别高。
- $m<0$：目标类别分数已经超过所有其他类别。
- $m=0$：发生并列，实际预测取决于并列处理规则。

因此，减小 $m$ 就是在推动目标类别成为模型预测。

### 4.2 原论文的写法

原论文的有目标损失为：[1]

$$
\boxed{f(x')=\max\big(m(x'),-\kappa\big)}
$$

损失达到下限 $-\kappa$ 对应：

$$
Z_t(x')\ge\max_{i\ne t}Z_i(x')+\kappa.
$$

也就是目标类别领先最强竞争者至少 $\kappa$。当 $\kappa=0$ 时，仍需直接检查预测，避免将并列误认为成功。

### 4.3 用假设分数看懂它

设三个类别分别为猫、狗、鸟，攻击目标是狗，取 $\kappa=2$。

| 状态 | 猫 | 狗（目标） | 鸟 | $m$ | $f=\max(m,-2)$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 8 | 2 | 1 | 6 | 6 |
| B | 6 | 5 | 1 | 1 | 1 |
| C | 5 | 6 | 1 | −1 | −1 |
| D | 4 | 6 | 1 | −2 | −2 |
| E | 3 | 7 | 1 | −4 | −2 |

从 A 到 C，模型已经改判为狗；从 C 到 D，进一步满足指定间隔。从 D 到 E，攻击项不再改善，因为它已经达到下限。此时更大的改动若增加距离，就会受到第一项惩罚。

这并不表示达到间隔后算法立刻停止；后续更新仍可能缩小距离，或再次使攻击项变为活跃状态。

### 4.4 为什么代码里经常写成非负形式？

作者代码采用等价的 hinge 写法：[2]

$$
g(x')=\max\big(m(x')+\kappa,0\big).
$$

由最大值的性质：

$$
g(x')=\max\big(m(x'),-\kappa\big)+\kappa=f(x')+\kappa.
$$

因此，在 $c$ 和 $\kappa$ 固定的一轮优化中：

$$
\|x'-x\|_2^2+c\,g(x')
=\|x'-x\|_2^2+c\,f(x')+c\kappa.
$$

两者只相差一个与 $x'$ 无关的常数，具有相同的最优解与梯度（在可微处）。不过，不同 $c$ 下的总损失值不适合直接拿来挑选最佳样本，应该比较成功样本的扰动。

### 4.5 $\kappa$ 不是概率

$\kappa$ 约束的是 logits 的间隔。例如 $\kappa=2$ 不代表“置信度为 2”或“概率为 200%”。它也不能跨不同模型直接解释为相同概率强度。

对两个类别，若目标领先 $2$，则目标 Softmax 概率为：

$$
p_t=\frac{1}{1+e^{-2}}\approx0.881.
$$

但如果总共有三个类别，另外两个类别都落后目标 $2$，则：

$$
p_t=\frac{1}{1+2e^{-2}}\approx0.787.
$$

同样的最高竞争者间隔，可以对应不同概率。

## 5. $c$ 与 $\kappa$ 分别控制什么？

| 参数 | 作用 | 需要避免的误解 |
| --- | --- | --- |
| $c$ | 决定优化时有多重视攻击项 | 它不是学习率，也不是扰动预算 |
| $\kappa$ | 规定希望达到的 logit 间隔 | 它不是概率阈值 |
| 学习率 $\eta$ | 控制优化变量每次更新的尺度 | 增大学习率不等于增大目标间隔 |

$c$ 太小时，优化可能更愿意保持原图而攻击失败；$c$ 较大时，优化更重视满足攻击条件。作者实现通过外层搜索调整 $c$。[2]

可以用一个**本文构造的一维例子**理解这种权衡。假设扰动大小为 $d\ge0$，分差随扰动变化为 $m(d)=1-2d$，且 $\kappa=0$：

$$
J(d)=d^2+c\max(1-2d,0).
$$

当 $0\le d<1/2$ 时：

$$
J'(d)=2d-2c.
$$

于是其最小点为 $d^*=\min(c,1/2)$。若 $c=0.1$，最优点还没有到分类边界；若 $c\ge0.5$，最优点到达边界。严格误分类还需要越过边界，并检查实际预测。

这个例子仅说明权衡关系。真实神经网络的优化通常是非凸的，不能据此保证增大 $c$ 就一定成功，也不能保证找到全局最小扰动。

## 6. 为什么引入 $\tanh$ 变量替换？

像素必须位于 $[0,1]$，而普通优化器更新一个无约束变量时可能越界。C&W $L_2$ 使用：[1]

$$
x'=h(w)=\frac{\tanh(w)+1}{2}.
$$

对有限实数 $w$，$-1<\tanh(w)<1$，因此映射后的像素位于 $(0,1)$，满足盒约束。此时优化问题成为：

$$
\boxed{
\min_w J(w)=\|h(w)-x\|_2^2+c\,g(h(w))
}
$$

为了从原图附近开始，可先取：

$$
w_0=\operatorname{arctanh}\big(\operatorname{clip}(2x-1,-1+\epsilon_{\rm num},1-\epsilon_{\rm num})\big).
$$

这里 $\epsilon_{\rm num}$ 是避免反双曲正切在端点发散的微小数值常量，**不是攻击允许的扰动预算**。端点裁剪会使初始重建与原图存在微小差异。

如果合法输入范围是 $[a,b]$，对应的一般形式为：

$$
h(w)=\frac{b-a}{2}\tanh(w)+\frac{a+b}{2}.
$$

作者代码的默认像素范围与本文示例不同，因此阅读代码时应同时检查 `boxmin`、`boxmax`。[2]

## 7. 梯度究竟在推动什么？

下面是对上述目标的直接推导。

设当前分数最高的非目标类别唯一，为：

$$
j^*=\arg\max_{i\ne t}Z_i(x').
$$

在攻击项活跃，即 $m(x')+\kappa>0$ 时：

$$
\nabla_{x'}J
=2(x'-x)+c\big(\nabla_{x'}Z_{j^*}(x')-\nabla_{x'}Z_t(x')\big).
$$

第一部分对应缩小扰动；第二部分对应降低竞争者相对目标的优势。由于目标是最小化 $J$，更新要沿负梯度方向走。

变量替换后，利用链式法则：

$$
\nabla_wJ
=\nabla_{x'}J\odot\frac{1-\tanh^2(w)}{2},
$$

其中 $\odot$ 表示逐元素乘法。在最大值并列或 hinge 转折处，使用自动微分选择的次梯度。

若用普通梯度下降说明更新：

$$
w_{k+1}=w_k-\eta\nabla_wJ(w_k).
$$

作者实现使用 Adam 更新优化变量。[2] Adam 会利用梯度的一阶、二阶统计调整更新；它不会把模型参数当作本次攻击的优化对象。

另外，$|w|$ 很大时，$1-\tanh^2(w)$ 会很小。因此，变量替换满足了范围约束，但并不意味着所有位置都容易优化。

## 8. 完整算法怎么运行？

以下是单个样本的教学伪代码，概括作者实现的内外层结构，不包含批处理和全部加速细节。[2]

```text
输入：原图 x、目标 t、固定模型 Z、间隔 kappa
      外层搜索次数 S、内层迭代次数 T、初始 c

best = 无
best_distance = 正无穷
lower = 0
upper = 正无穷

重复 S 轮：
    从原图附近初始化 w，重置 Adam 状态
    round_success = False

    重复 T 次：
        x_adv = (tanh(w) + 1) / 2
        logits = Z(x_adv)
        m = max(logits[i]，其中 i != t) - logits[t]
        attack_loss = max(m + kappa, 0)
        distance = sum((x_adv - x)^2)
        loss = distance + c * attack_loss

        若实际预测为 t，且达到要求的间隔：
            round_success = True
            若 distance < best_distance：
                保存 x_adv 和 distance

        对 loss 关于 w 反向传播，用 Adam 更新 w

    若 round_success：
        upper = min(upper, c)
    否则：
        lower = max(lower, c)

    若 upper 有限：
        c = (lower + upper) / 2
    否则：
        c = 10 * c

输出：best；若从未成功，则明确返回攻击失败
```

外层试探合适的 $c$，内层优化图片。尚未找到成功上界时先增大 $c$，找到后再收缩区间。由于内层求解并非精确全局优化，这种外层搜索是实用策略，不是全局最优证明。

应保存整个过程中的最佳成功样本：最后一次迭代的图片未必最好，甚至可能已不满足攻击条件。

## 9. 无目标攻击怎么改？

有目标攻击要求 $C(x')=t$；无目标攻击只要求 $C(x')\ne y$。作者代码也支持后一种情况。[2]

对应的非负损失为：

$$
g_{\rm untargeted}(x')=
\max\left(Z_y(x')-\max_{i\ne y}Z_i(x')+\kappa,0\right).
$$

这个符号顺序可以从目标直接推出来：想让真实类别输给某个错误类别，就要减小“真实类别分数减最高错误类别分数”。

因此，比较两种损失时，先明确哪个类别应该获胜，比机械记忆正负号更可靠。

## 10. $L_0$、$L_2$、$L_\infty$ 有什么不同？

原论文提出三个版本，不能仅替换距离符号就认为求解过程完全相同。[1]

| 度量 | 数学含义 | 直观问题 |
| --- | --- | --- |
| $\|\delta\|_0$ | 非零分量数量；严格说不是范数 | 改了多少个位置？ |
| $\|\delta\|_2$ | $\sqrt{\sum_j\delta_j^2}$ | 总体改动有多大？ |
| $\|\delta\|_\infty$ | $\max_j|\delta_j|$ | 单个分量最多改了多少？ |

例如 $\delta=(0.1,0,0,-0.1)$，则 $L_0=2$、$L_2=\sqrt{0.02}\approx0.1414$、$L_\infty=0.1$。

对 RGB 图像，“修改像素数量”和“修改标量通道数量”可能采用不同计数口径，比较论文结果前应核对定义。

## 11. 与 FGSM、PGD 怎么比较？

以下比较典型的无目标 $L_\infty$ FGSM／PGD 与 C&W $L_2$；不同方法也存在其他变体。[1][3][4]

| 方面 | FGSM | PGD | C&W $L_2$ |
| --- | --- | --- | --- |
| 基本思路 | 沿损失梯度符号走一步 | 多次更新，并投影回允许集合 | 优化距离与攻击项的组合 |
| 典型问题 | 给定预算内快速寻找攻击 | 给定预算内寻找高损失输入 | 搜索满足攻击条件的小扰动 |
| 更新次数 | 一次 | 多次 | 内层多次，外层搜索 $c$ |
| 常见范围处理 | 裁剪 | 投影和裁剪 | $\tanh$ 变量替换 |
| 计算开销 | 通常低 | 依赖迭代和重启数 | 依赖内外层迭代，通常较高 |

典型 FGSM 更新为：

$$
x'=\operatorname{clip}_{[0,1]}\left(x+\epsilon\operatorname{sign}(\nabla_x\ell(x,y))\right).
$$

典型 PGD 迭代为：

$$
x_{k+1}=\Pi_{\mathcal S}\left(x_k+\alpha\operatorname{sign}(\nabla_{x_k}\ell(x_k,y))\right),
$$

其中 $\mathcal S=\{u\in[0,1]^n:\|u-x\|_\infty\le\epsilon\}$，通常还使用随机起点。

**C&W 风格损失也可以配合 PGD 更新。**损失函数和优化器是两个不同选择，不能把“用了 C&W loss”直接等同于“完整复现了原始 C&W 算法”。

评价时应统一范数、预算、攻击目标和计算量。不能仅凭方法名字断言哪种攻击一定更强。

## 12. 用在 ViT 上时怎么理解？

从计算图角度，只要能够获取图像输入的梯度，就可以将上述目标用于可微的 ViT 分类器。这是由攻击目标得到的适用性判断，不是原论文对 ViT 的实验结论。

反向传播经过分类头、Transformer 编码器和 patch embedding，最终把梯度传到图像像素。优化像素时，扰动约束仍应在约定的图像空间中计算。

如果模型接收标准化输入 $N(x')$，应写清楚：

$$
J(w)=\|h(w)-x\|_2^2+c\,g\big(N(h(w))\big),
$$

这里 $g$ 内部包含模型对标准化输入的前向计算。距离仍在原始 $[0,1]$ 像素空间中计算，标准化留在可微计算图内。

修改像素、修改 patch token、修改注意力矩阵是不同的攻击设定，不能混用扰动度量或成功率解释。

## 13. 复现与阅读时要检查什么？

- **是否使用 logits？**先 Softmax 再代入上述分差，会改变目标函数。
- **$c$ 是否搜索？**固定一个 $c$ 的简化实现可能不能代表完整方法的表现。
- **距离是求和还是平均？**把平方距离的求和改成均值，会改变其与攻击项的相对尺度。
- **是否保留最佳成功样本？**不能仅返回最后一步结果。
- **是否真正满足预算？**若评测要求 $\|\delta\|_2\le\epsilon$，必须额外检查；经典惩罚目标没有自动施加这一硬约束。
- **攻击失败如何处理？**应明确标记失败，不能将原图混入成功样本的平均扰动。
- **成功率的分母是什么？**说明是否只对原始分类正确的样本攻击；平均扰动说明是否只统计成功样本。
- **保存后是否复核？**量化为 8 位图片会改变输入，需要在实际保存或预处理后的图片上重新检查。

读懂 C&W 的关键，是区分三个层面：**目标类别应该怎样获胜、用什么损失表达这一目标、用什么算法优化输入。**掌握这三点后，损失公式、变量替换与内外层搜索就能连起来理解。

## 参考资料

1. Nicholas Carlini, David Wagner. **Towards Evaluating the Robustness of Neural Networks**. IEEE Symposium on Security and Privacy, 2017, pp. 39–57. DOI: `10.1109/SP.2017.49`。重点阅读第 VI-A 节的 $L_2$ 目标。
   - [作者提供的论文 PDF](https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf)
   - [arXiv:1608.04644](https://arxiv.org/abs/1608.04644)
2. [作者官方代码：carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks)，重点查看 [l2_attack.py](https://github.com/carlini/nn_robust_attacks/blob/master/l2_attack.py) 中的损失、变量范围和外层搜索。
3. Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. **Explaining and Harnessing Adversarial Examples**. ICLR 2015.
   - [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
4. Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, Adrian Vladu. **Towards Deep Learning Models Resistant to Adversarial Attacks**. ICLR 2018.
   - [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)

> 本文没有复现实验。所有示例分数与一维模型均为教学构造，不代表真实模型的攻击成功率或扰动水平。

<!-- knowledge-links:start -->

## 笔记联系

- [[30_对抗机器学习/攻击方法/论文 - Explaining and Harnessing Adversarial Examples|论文 - Explaining and Harnessing Adversarial Examples]]：比较优化式攻击与梯度符号攻击。
- [[30_对抗机器学习/防御与鲁棒性/Gradient Masking - 梯度掩蔽详解|Gradient Masking - 梯度掩蔽详解]]：结合强攻击与梯度问题理解鲁棒性评估。
- [[30_对抗机器学习/通用与稀疏扰动/CornerSearch - 稀疏对抗攻击|CornerSearch - 稀疏对抗攻击]]：比较稀疏搜索与优化式攻击。
- [[30_对抗机器学习/通用与稀疏扰动/论文 - The Limitations of Deep Learning in Adversarial Settings|论文 - The Limitations of Deep Learning in Adversarial Settings]]：对照基于显著图选择特征与优化式攻击的思路。

## 论文出处

- [Explaining and Harnessing Adversarial Examples](https://iclr.cc/archive/www/2015.html)：ICLR，2015，会议论文。 早期 ICLR 论文以官方会议接收目录为发表记录，论文全文见预印本链接。
  [预印本](https://arxiv.org/abs/1412.6572)。
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://openreview.net/forum?id=rJzIBfZAb)：ICLR，2018，会议论文。
  [预印本](https://arxiv.org/abs/1706.06083)。
- [Towards Evaluating the Robustness of Neural Networks](https://doi.org/10.1109/SP.2017.49)：IEEE Symposium on Security and Privacy (S&P)，2017，会议论文。
  [预印本](https://arxiv.org/abs/1608.04644)。

<!-- knowledge-links:end -->
