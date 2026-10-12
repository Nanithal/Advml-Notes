神经网络（为了模仿人的大脑而提出的名词）
大脑中有神经元neuron等等，传递脉冲信号等等，生物知识
神经元接受和传递电信号（输入与输出）
在神经网络中，将输出记作a（activation激活）
多个/一个神经网络在一起是层layer，输入层，隐藏层，输出层
![[Pasted image 20260604160333.png]]

为何神经网络在当下流行----其对大数据处理较牛逼
神经网络不需要手动设计特征，他可以学习
![[Pasted image 20260717172313.png]]

隐藏层可能具有多个（取决于神经网络架构），神经网络架构的选择将影响到算法的正确预测等等
![[Pasted image 20260721142552.png]]

复杂神经网路架构的流程
![[Pasted image 20260721232338.png]]

前向传播：输入->隐藏->输出的流程，最终得到预测值（从左到右）
![[Pasted image 20260721233231.png]]

Numpy（array)和Tensorflow（Tensor）存储多维数据
![[Pasted image 20260723143333.png]]
![[Pasted image 20260723194526.png]]

![[Pasted image 20260723195455.png]]

![[Pasted image 20260723195638.png]]
单层前向传播
![[Pasted image 20260728142130.png]]
python代码实现
![[Pasted image 20260728142920.png]]