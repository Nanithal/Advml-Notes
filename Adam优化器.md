在理解了 **Momentum（一阶动量，管方向）** 和 **RMSprop（二阶动量，管步长）** 之后，我们就迎来了现代深度学习中最著名、最常用的优化器——**Adam（Adaptive Moment Estimation，自适应矩估计）**。

Adam 的本质非常直白：**Adam = Momentum + RMSprop + 偏差修正**。

它同时利用了历史梯度的均值（一阶矩）**来指引方向，以及历史梯度平方的**均值（二阶矩）来自适应调节学习率，并完美解决了一个常常被忽略的“冷启动”问题。

### 一、 Adam 的标准核心公式

假设当前是第 $t$ 次迭代（从 $t=1$ 开始），当前梯度为 $g_t$。Adam 的更新流程包含 5 个明确的步骤：

#### 1. 计算当前梯度

$$g_t = \nabla_\theta J(\theta_{t-1})$$

#### 2. 更新一阶动量（Momentum 部分）

$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t$$

_(记录历史梯度的平滑方向)_

#### 3. 更新二阶动量（RMSprop 部分）

$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2$$

_(记录历史梯度的平方大小，用于后续步长自适应缩放)_

#### 4. 偏差修正（Bias Correction）—— Adam 的独门秘籍

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

_(消除由于初始值设为 0 带来的前期预估偏小问题)_

#### 5. 更新模型参数

$$\theta_t = \theta_{t-1} - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

### 二、 Adam 的核心超参数

Adam 的作者在论文中给出了极具统治力的默认参数，这些参数在 99% 的框架（如 PyTorch、TensorFlow）中都是默认值，且通常不需要修改：

- **$\alpha$（学习率 Learning Rate）**：通常设为 $0.001$。
    
- **$\beta_1$（一阶动量衰减系数）**：默认 **$0.9$**（和标准 Momentum 一致）。
    
- **$\beta_2$（二阶动量衰减系数）**：默认 **$0.999$**（意味着对历史梯度平方的记忆非常长）。
    
- **$\epsilon$（防除零常数）**：默认 **$10^{-8}$**。
    

### 三、 深度解析：为什么要进行“偏差修正”？

在 Momentum 和 RMSprop 中，动量的初始值 $m_0$ 和 $v_0$ 都是 $0$向量。这种**初始化为 0** 的操作，会导致在训练初期（$t$ 较小时），算出来的移动平均值严重偏向 0（**冷启动问题**）。

我们以一阶动量为例算一笔账：

假设第 $1$ 步时算出的真实梯度 $g_1 = 10$，$\beta_1 = 0.9$。

如果不做修正：

$$m_1 = 0.9 \cdot m_0 + 0.1 \cdot g_1 = 0 + 0.1 \times 10 = 1$$

**问题出现了**：真实的梯度是 $10$，但算出来的平滑动量 $m_1$ 只有 $1$！这使得模型在训练最开始的几步，更新步长极其微小，白白浪费算力。

**加入偏差修正后：**

$$\hat{m}_1 = \frac{m_1}{1 - 0.9^1} = \frac{1}{0.1} = 10$$

完美还原了真实的梯度量级！

随着迭代次数 $t$ 不断增大（比如 $t=1000$），$\beta^t$ 会无限趋近于 $0$（如 $0.9^{1000} \approx 0$），那么分母 $1 - \beta^t \approx 1$。

也就是说，**偏差修正只在训练的前期起放大作用，在训练中后期会自动“隐身退场”**。

### 四、 Adam 为什么这么强？

1. **傻瓜式调参**：因为有二阶动量自适应调节步长，Adam 对初始学习率 $\alpha$ 不敏感。哪怕你随便给个 $10^{-3}$，它也能在绝大多数网络上立刻跑出不错的结果（被称为“开箱即用”）。
    
2. **结合了双重优势**：既有 Momentum 冲出局部极小值和减弱震荡的能力，又有 RMSprop 对极度平缓或极度陡峭地形的自适应处理能力。
    
3. **参数更新有界限**：在 Adam 的更新公式中，真实的步长增量大约被限制在 $\alpha$ 附近，不会发生像普通 GD 那样梯度爆炸导致步长无限大的问题。
    

### 五、 面向对象代码实现 (Python/NumPy)

以下是符合工业标准的纯代码实现，你可以清晰看到迭代次数 $t$ 是如何参与运算的：

Python

```
import numpy as np

class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # 一阶动量
        self.v = None  # 二阶动量
        self.t = 0     # 迭代步数记录器

    def update(self, params, grads):
        # 延迟初始化
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1  # 更新步数

        # 1. 更新一阶和二阶动量
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grads ** 2)

        # 2. 计算偏差修正 (Bias Correction)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)

        # 3. 更新参数
        adaptive_lr = self.lr / (np.sqrt(v_hat) + self.epsilon)
        updated_params = params - adaptive_lr * m_hat

        return updated_params

# --- 验证逻辑 ---
if __name__ == "__main__":
    opt = AdamOptimizer(lr=0.1)
    w = np.array([5.0, -4.0]) 
    
    for epoch in range(3):
        grad = w * 2.0  # 假设梯度
        w = opt.update(w, grad)
        print(f"Step {epoch+1}: w = {np.round(w, 4)}")
```