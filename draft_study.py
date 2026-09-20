# %%

import torch 

# %% tensor 和Tensor


a = torch.Tensor([1.0, 2.0, 3.0])

b = torch.tensor([1,2,3])

print(a.dtype)#float32
print(b.dtype)#int64

# %% backward()

# %% 标量loss

# 标量loss 的话就可以直接调用backward()，不需要传入参数

# 下面的例题是计算一个线性问题的loss

w=torch.tensor([2.0, 3.0], requires_grad=True)

x=torch.tensor([1.0, 2.0])

target=torch.tensor(1.0)

pred=torch.dot(w,x)

loss = (pred-target)**2

loss.backward()

print(w.grad)


# %% 向量loss


import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

L1 = x[0] ** 2 + x[1]
L2 = x[0] * x[1]
L3 = 3 * x[0] + x[1] ** 2

L = torch.stack([L1, L2, L3])

v = torch.tensor([1.0, 2.0, 3.0])

L.backward(v)

print(x.grad)

# %% detach()

x=torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

a=2*x

b=3*a

loss = b.sum()

loss.backward()

print(x.grad)


# %%
