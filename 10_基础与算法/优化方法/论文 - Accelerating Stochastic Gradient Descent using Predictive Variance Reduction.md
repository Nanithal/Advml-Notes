---
title: "论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
aliases:
  - "Accelerating Stochastic Gradient Descent using  Predictive Variance Reduction论文"
分类: "优化方法"
笔记类型: "论文阅读"
整理日期: "2026-10-27"
tags:
  - "知识库/论文阅读"
  - "主题/优化方法"
状态: "已整理"
摘要: "围绕“论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction”，整理研究背景与问题、SVRG 算法、核心思想。"
更新日期: "2026-10-30"
上级目录: "[[10_基础与算法/优化方法/总结 - 优化方法]]"
相关笔记:
  - "[[10_基础与算法/优化方法/SVRG - 随机方差缩减梯度]]"
论文条目:
  - "[[论文来源目录#svrg]]"
论文标题:
  - "Accelerating Stochastic Gradient Descent using Predictive Variance Reduction"
发表载体:
  - "NIPS"
发表年份:
  - "2013"
发表类型:
  - "会议论文"
发表链接:
  - "https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html"
预印本链接: []
来源核实日期: "2026-10-30"
---

[[首页|首页]] · [[10_基础与算法/优化方法/总结 - 优化方法|优化方法总结]]

# 论文 - Accelerating Stochastic Gradient Descent using Predictive Variance Reduction


这篇论文《Accelerating Stochastic Gradient Descent using Predictive Variance Reduction》（NIPS 2013）由 Rie Johnson 和 Tong Zhang 撰写，提出了 **SVRG（Stochastic Variance Reduced Gradient，随机方差缩减梯度）** 算法。它通过显式地减少 SGD 中的梯度方差，在不存储全部梯度的情况下，达到了与 SDCA 和 SAG 相同的快速线性收敛率，并且分析更简单直观，适用于非凸问题（如神经网络）。

以下是全文总结。

---

## 1. 研究背景与问题

机器学习中常遇到有限和优化问题：

$$\min_{w} P(w) = \frac{1}{n}\sum_{i=1}^{n}\psi_i(w)$$

- **梯度下降（GD）**：每步需计算全部 n 个梯度，计算量大，但收敛快（线性收敛）。
    
- **随机梯度下降（SGD）**：每步只随机选一个样本 i，计算 $\nabla \psi_i(w)$，计算量为 GD 的 1/n，但引入了方差。为了收敛，学习率需逐渐衰减（如 $\eta_t = O\left(\frac{1}{t}\right)$），导致次线性收敛 $O(1/t)$。因此 SGD 存在“每步快但收敛慢”与 GD“每步慢但收敛快”的权衡。
    

**已有方差缩减方法**：

- **SAG**（Le Roux et al., 2012）：随机平均梯度，需存储所有历史梯度。
    
- **SDCA**（Shalev-Shwartz & Zhang, 2012）：随机对偶坐标上升，需存储所有对偶变量。
    
- 两者都能达到线性收敛，但都需要存储全部梯度或对偶变量，内存开销大，难以用于结构化预测和神经网络等复杂问题。
    

**本文目标**：提出一种不需要存储全部梯度的显式方差缩减方法，达到同样的快速收敛率，且分析更简单，可推广到非凸问题。

---

## 2. SVRG 算法

### 2.1 核心思想

SGD 方差大的原因是：用单个样本梯度 $\nabla \psi_i(w)$ 近似全梯度$\Delta P(w)$。SVRG 通过维护一个“快照” $\tilde{w}$（接近最优解）和该点的全梯度平均 $\tilde{\mu} = \nabla P(\tilde{w})$，构造一个方差更小的随机梯度估计：

$$g_t = \nabla \psi_{i_t}\big(w^{(t-1)}\big) - \nabla \psi_{i_t}(\tilde{w}) + \tilde{\mu}$$

其期望仍为$\nabla P\big(w^{(t-1)}\big)$，但当 $w^{(t-1)}$ 和 $\tilde{w}$ 都趋近最优解 $w_*$ 时，$\nabla \psi_{i_t}\big(w^{(t-1)}\big)-\nabla \psi_{i_t}(\tilde{w})\to 0$，方差趋于零。

### 2.2 算法流程（Procedure SVRG，图1）

**参数**：更新频率 $m$，学习率 ηη  
**初始化**：$\tilde{w}_0$
**迭代**：对 $s=1,2,…$

1. 计算全梯度平均：$\tilde{\mu}=\frac{1}{n}\sum_{i=1}^{n}\nabla\psi_{i}(\tilde{w}) \quad (\text{其中 } \tilde{w}=\tilde{w}_{s-1})$
    
2. 令 $w_0 = \tilde{w}$
    
3. 内层循环：对 $=1,2,…,m$
    
    - $\text{随机选 } i_t \in \{1,\dots,n\}$
        
    - $\text{更新：} w_t = w_{t-1} - \eta\big(\nabla\psi_{i_t}(w_{t-1}) - \nabla\psi_{i_t}(\tilde{w}) + \tilde{\mu}\big)$
        
4. 选择下一个快照：
    
    - $\text{Option I: } \tilde{w}_s = w_m$​
        
    - $\text{Option II: } \tilde{w}_s = w_t \quad (\text{随机选一个 } t\in\{0,\dots,m-1\})$
        
5. 重复直到收敛。
    

**计算成本**：每个阶段（s）需计算 $n$ 个全梯度（用于 $\tilde{\mu}$）和 $m$ 次内层迭代（每次 2 个梯度）。对于凸问题，可只存标量，内存与 SDCA/SAG 相当；非凸问题则需额外存储。

---

## 3. 理论分析

**假设**：每个 $\psi_i(w)$ 是凸且光滑的（满足式(5)），$P(w)$ 是强凸的（满足式(6)），参数 $L \ge \gamma > 0$。

**定理 1**：若 $m$ 足够大，使得

$$\alpha=\frac{1}{\gamma\eta(1-2L\eta)m}+\frac{2L\eta}{1-2L\eta}<1$$

则 SVRG（Option II）在期望上以几何速率收敛：

$$\mathbb{E}\big[P(\tilde{w}_s)\big] - P(w_*) \le \alpha^s \big[P(\tilde{w}_0)-P(w_*)\big]$$

**证明要点**：定义 $v_t=\nabla\psi_{i_t}(w_{t-1})-\nabla\psi_{i_t}(\tilde{w})+\tilde{\mu}v_t$​，利用光滑性和强凸性推导出 $\mathbb{E}\|v_t\|^2$的上界，并证明其随 $w$ 接近 $w_∗$​ 而减小，最终得到收敛率。证明比 SAG/SDCA 更简单直观。

**复杂度对比**：

- 当条件数 $L/\gamma = n$时，标准 GD 需 $n\ln(1/\epsilon)$ 次迭代，即 $n^2\ln(1/\epsilon)$ 。
    
- SVRG 取 $\eta = 0.1/L,\; m = O(n),$，可得 $\alpha = 1/2$，达到精度 $\epsilon$ 只需处理 $n\ln(1/\epsilon)$个样本梯度，与 SAG/SDCA 相同。
    
- 对于非强凸光滑问题，SVRG 可达 $O(1/T)$，优于 SGD 的 $O(1/T​)$。
    
- 对于非凸问题（如神经网络），若初始点接近局部极小，SVRG 可加速局部收敛。
    

---

## 4. SDCA 作为方差缩减方法

论文还展示了 SDCA 也可以从方差缩减的角度理解。对于带 L2 正则的问题：

$$P(w) = \frac1n \sum_{i=1}^n \phi_i(w) + \frac{\lambda}{2}\|w\|^2$$

在最优解处有对偶表示$w^* = \sum_{i}\alpha_i^*$,其中 $\alpha_i^* = -\frac{1}{\lambda n}\nabla\phi_i(w^*)$。SDCA 的对偶坐标上升更新可保持 $\nabla\phi_i(w) + \lambda n \alpha_i \to 0$，因此即使学习率不衰减，方差也趋于零。这从另一个角度解释了 SDCA 的快速收敛。

---

## 5. 实验

**凸问题**：在 MNIST 上做 L2 正则多分类逻辑回归，在 rcv1、covtype、protein、CIFAR-10 上做二分类/多分类逻辑回归。

- 对比 SGD（固定学习率、最佳调参的学习率调度）、SDCA。
    
- 结果：SVRG 使用单个较大学习率即可平滑快速下降，训练损失残差指数下降，与 SDCA 相当，明显优于最佳调参的 SGD。SVRG 的更新方差随优化过程递减，而固定学习率的 SGD 方差保持高位。
    

**非凸问题**：在 MNIST 和 CIFAR-10 上训练单隐层神经网络（100 个 sigmoid 隐藏节点，10 个 softmax 输出，L2 正则）。

- 结果：SVRG 同样比最佳调参的 SGD 收敛更快，训练损失和测试误差都更低。由于 SDCA/SAG 需要存储全部梯度，不适用于神经网络，SVRG 在这方面具有优势。
    

---

## 6. 结论

- 提出 **SVRG**，一种显式方差缩减的 SGD 加速方法。
    
- 对于光滑强凸函数，达到与 SDCA、SAG 相同的线性收敛率，但证明更简单直观。
    
- 不需要存储全部梯度，内存开销小，适用于结构化预测和神经网络等复杂问题。
    
- 实验验证了 SVRG 在凸和非凸问题上的有效性和快速收敛性。
    
- 论文也指出，SDCA 可被视为另一种方差缩减方法，这为理解随机优化算法提供了统一视角。
    

---

## 核心贡献总结

|贡献|说明|
|---|---|
|**显式方差缩减**|通过快照点全梯度与随机点梯度之差构造低方差估计，使 SGD 可用大学习率并线性收敛。|
|**无需存储全梯度**|相比 SAG/SDCA，内存友好，可扩展到神经网络等非凸问题。|
|**更简单的分析**|证明简洁直观，揭示方差缩减本质。|
|**统一视角**|将 SDCA 也解释为方差缩减方法，促进算法理解与改进。|
|**实验验证**|在多个凸/非凸数据集上优于最佳调参 SGD，与 SDCA 相当。|

<!-- knowledge-links:start -->

## 笔记联系

- [[10_基础与算法/优化方法/SVRG - 随机方差缩减梯度|SVRG - 随机方差缩减梯度]]：结合算法笔记复习控制变量与方差缩减。

## 论文出处

- [Accelerating Stochastic Gradient Descent using Predictive Variance Reduction](https://papers.nips.cc/paper_files/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html)：NIPS，2013，会议论文。

<!-- knowledge-links:end -->
