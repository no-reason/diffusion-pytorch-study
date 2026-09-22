# Pytorch学习

## python中的面向对象编程实现方法


```python
python 中也可以用class来实现面向对象编程，但是两者对于构造函数的使用方法不同
在C++中,构造函数必须与类的名称相同,不可以有返回值，不需要声明返回值类型

在python中，构造函数通常用__init__来实现，不可以有返回值，不需要声明返回值类型

C++中this指针隐式定义,访问类的成员可以直接写成员名，也可以用this->成员名来访问

python中，__init__的第一个参数必须是self,其他参数可以通过需要来定义,
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

# 当你写：
r = Rectangle(10, 20)

# Python 实际执行：
# 1. 创建空对象
# 2. Rectangle.__init__(r, 10, 20)  # self=r, width=10, height=20
class MyClass:
    def method(self, x):
        print(f"self={self}, x={x}")

obj = MyClass()

# 这两种调用是等价的：
obj.method(10)              # 推荐写法
MyClass.method(obj, 10)     # 等价写法（不常用）
```

### python中的类的继承

```python
class Net(nn.Module):# 表示Net这个类继承自nn.Module这个类
    def __init__(self):
        super().__init__()    #这是python中继承的语法，表示调用父类的构造函数
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

## python中的类型标注

```python
def project_topk_linf(delta: torch.Tensor, eps: Optional[float], k: Optional[int]) -> torch.Tensor:
    projected = delta
    if eps is not None:
        projected = projected.clamp(-float(eps), float(eps))
    if k is None or k >= projected.numel():
        return projected
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return torch.zeros_like(projected)
    flat = projected.flatten()
    indices = flat.abs().topk(k, sorted=False).indices
    sparse = torch.zeros_like(flat)
    sparse[indices] = flat[indices]
    return sparse.view_as(projected)

这里的delta:torch.Tensor和这个函数-> torch.Tensor都是类型标注，对于python解释器没有影响，只是可以让IDE帮助进行代码提示，然后可以提升一定的可读性
```


## Tensor和tensor

```python
Tensor:是pytorch中Tensor这个类的名称
tensor:是用来声明Tensor对象的一个工厂函数
tensor()可以实现数据类型自动推断
Tensor()会默认把数据类型存储成float32
```

## tensor、Parameter、Buffer

```python

普通Tensor: 不可训练, .to(device)不可自动移动 , state_dict()不默认保存

nn.Parameter: 可训练, .to(device)可自动移动 , state_dict()默认保存

register_buffer: 不可训练, .to(device)可自动移动 , state_dict()默认保存


普通Tensor: 临时/辅助数据

Parameter : 模型要学习的参数

Buffer : 模型运行必须但不学习的状态
```

## backward

### 标量loss

```python
# 标量loss 的话就可以直接调用backward()，不需要传入参数

# 下面的例题是计算一个线性问题的loss

w=torch.tensor([2.0, 3.0], requires_grad=True)

x=torch.tensor([1.0, 2.0])

target=torch.tensor(1.0)

pred=torch.dot(w,x)

loss = (pred-target)**2

loss.backward()

print(w.grad)


```

### 向量loss
此时loss不是一个数而是一个向量,
这时loss调用需要带一个gradient参数
L.backward(v)计算的是$J_{L}(x)^{T}v$
```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

L1 = x[0] ** 2 + x[1]
L2 = x[0] * x[1]
L3 = 3 * x[0] + x[1] ** 2

L = torch.stack([L1, L2, L3])

v = torch.tensor([1.0, 2.0, 3.0])

L.backward(v)

print(x.grad)
```

## checkpoint

```python
y = checkpoint(block, x, use_reentrant=False)
# block内部的中间activation会按照checkpoint的逻辑进行处理
```
## 把tensor转化为numpy中的矩阵

```python
tensor.detach().cpu().numpy()
```

## torch.nn
torch.nn是pytorch中专门用来搭建神经网络模型的工具库

在通常的计算神经网络有多少层中，只有带可训练参数的层才被算进来

Relu就没有自己的参数，所以Relu层就不计入层数
```python
# 基类
torch.nn.Module# :所有模型和网络层的基类


#各种层
torch.nn.Linear# :线性层
torch.nn.Conv2d# :二维卷积层
torch.nn.ReLU# :ReLU激活层

#参数管理
torch.nn.Parameter# :可训练参数层

# 损失函数
torch.nn.MSELoss# :均方误差损失函数
torch.nn.CrossEntropyLoss# :交叉熵损失函数


# 利用nn.Module来搭建模型
# 搭建一个两层网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
```

### Dropout

```python
pytorch中的model具有两种模式:eval()和train()

在train()模式下,dropout层会随机将一些神经元设置为0,从而减少过拟合的风险

在eval()模式下自动关闭dropout层

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(p=0.3)# p表示每个神经元被dropout的概率
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### BatchNorm

在pytorch中train()模式除了Dropout以外还有另一个和eval()模式不同的点:就是在与BatchNorm



### freeze_model

```python
# 可以实现model中的参数冻结，但是导致梯度没有办法经过model

def freeze_model(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
```

## torch.max()

```python
# 求整个tensor的最大值
import torch

x = torch.tensor([
    [1, 5, 3],
    [4, 2, 6]
])

m = torch.max(x)

print(m)

# 沿某个维度求最大值
values, indices = torch.max(x, dim=1)

# keepdim=True

被压缩的维度仍然保留,只是元素个数变成1
```

## detach()

```python
detach()得到一个和原tensor共享数据、但从当前计算图中切断梯度传播的新tensor
y=f(x)
z=y.detach()


import torch 

x = torch.tensor(1.0,requires_grad=True)

a=2*x

b=3*a

b.backward()

print(x.grad)
```

## x.view()

在不改变Tensor元素总数和数据内容的前提下，重新解释x的形状


## x.squeeze()

吧Tensor中长度为1的维度删掉


## torch.linspace

torch.linspace(a,b,steps=n)
在a,b,之间包括端点生成n个数

```python
torch.linspace(0.1, 0.5, steps=5)
#得到tensor([0.1, 0.2, 0.3, 0.4, 0.5])
```

## torch.cat

张量拼接

```python
torch.cat([tensors],dim=k)

将tensors中的张量沿着第k个维度拼接起来

A.shape = (2, 3)
B.shape = (4, 3)

C = torch.cat([A, B], dim=0)
# C.shape = (6, 3)

A.shape = (2, 3)
B.shape = (2, 5)

C = torch.cat([A, B], dim=1)
# C.shape = (2, 8)

betas = torch.cat([torch.zeros([1]), betas], dim=0)
# 可以实现下标从0开始到下标从1开始
```

