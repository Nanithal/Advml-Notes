# 张量（tensor）

PyTorch（以及所有深度学习模型）的底层计算单元是张量——一个多维数组。
## 分类

| 维度  | 阶数  | 英文名    | 数学对象 |     |
| --- | --- | ------ | ---- | --- |
| 0维  | 0   | scalar | 标量   |     |
| 1维  | 1   | vector | 向量   |     |
| 2维  | 2   | matrix | 矩阵   |     |
|     |     |        |      |     |
## 创建
采用torch里面的函数创建
```py
x = torch.tensor(1., requires_grad=True)#创建0维张量，值为1.0
y = torch.tensor([1.],requires_grad=True)#创建一维张量，长度为1.0
#......
#参数表里有几个括号就是几维张量

```

# 梯度
pytorch可以实现自动计算梯度
在创建张量的时候，参数表里若指明``requires_grad=True``,则pytorch会自动记录所有针对该张量的运算历史，构建一张计算图
## 函数实现

```py
y.backward()#沿着计算图反向遍历，利用逐层计算导数
print(x.grad)#输出梯度（偏导）
```

