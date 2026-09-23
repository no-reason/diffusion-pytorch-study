# PDM_study


## PointNet.py
在PDM中,我们首先由PointNet.py对输入点云进行编码

为了满足后面生成任务的需要我们让输出层输出之后正太分布的均值和方差两个量


```python
import torch
import torch.nn.functional as F
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v//这里的m对应均值,v对应方差


```

## diffusion.py

这里的diffusion.py主要实现了diffusion过程

主要包括三个类:

1.VarianceSchedule:这里主要是计算一些register_buffer的值

而且这里需要注意一点,

在diffusion模型利用神经网络来进行预测的时候,其实分布的方差并不是一个可以学习的量。而是通过后验分布可以使用Bayes公式直接算出来与x_0无关的。

所以这里的方差计算用了一种 线性插值的方法来进行计算
```python
    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
```


2.PointwiseNet

这个类主要是用来实现用神经网络对\epsilon(t,z,x)进行预测

这里值得积累的一个点是这里使用神经网络对\epsilon (t,z,x)进行预测的时候，采用了标准的神经网络和残差神经网络两种方式:

```python
    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:#如果我这个表示残差连接开关的bool参数为true，那么就是残差神经网络
            return x + out
        else:
            return out#否则就是标准的神经网络
```

```python
class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(3, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 3, context_dim+3)
        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


```

3.DiffusionPoint

这一部分实现了diffusion.py的核心步骤：loss的计算和推理过程的采样.

这里需要注意PDM在实现的时候并没有真的去算$\hat{x_{0}}和x_{0}$之间的loss

而是选择了用\hat{\epsilon}和\epsilon 之间的loss来进行替换


```python
    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss
```

这里有一个调用的时候有没有提供初始噪声的点:
```python
 if initial_x_T is None:
            x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        else:
            x_T = initial_x_T.to(context.device)
            assert x_T.shape == (batch_size, num_points, point_dim)
'''
如果调用的时候没有提供初始噪声，那么就从Guass分布中随机采样生成一个初始噪声点
如果调用的时候提供了初始噪声点，那么就使用这个初始噪声点作为初始点,讲这个初始点转移到gpu上，并且确保这个初始点是正确的维度
'''
```


去噪过程

```python
        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            if return_trace and t == self.var_sched.num_steps:
                trace["first_reverse_input"] = x_t.detach().cpu()

            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            
            if return_trace and t == self.var_sched.num_steps:
                trace["first_reverse_output"] = x_next.detach().cpu()

            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if return_trace:
            trace["final_x_0"] = traj[0].detach().cpu()
            return trace

        if ret_traj:
            return traj
        else:
            return traj[0]
```


需要注意的是在DiffusionPoint这个类中我们可以发现虽然上面的VarianceSchedule类中对于预测分布的方差sigma使用了线性插值。但是实际上的默认值还是和bayes后验的结果是一致的

```python

class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, t=None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False, initial_x_T=None, return_trace=False):
        batch_size = context.size(0)
        if initial_x_T is None:
            x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device)
        else:
            x_T = initial_x_T.to(context.device)
            assert x_T.shape == (batch_size, num_points, point_dim)

        traj = {self.var_sched.num_steps: x_T}
        trace = {}
        if return_trace:
            trace["initial_x_T"] = x_T.detach().cpu()

        for t in range(self.var_sched.num_steps, 0, -1):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            if return_trace and t == self.var_sched.num_steps:
                trace["first_reverse_input"] = x_t.detach().cpu()

            beta = self.var_sched.betas[[t]*batch_size]
            e_theta = self.net(x_t, beta=beta, context=context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            
            if return_trace and t == self.var_sched.num_steps:
                trace["first_reverse_output"] = x_next.detach().cpu()

            traj[t-1] = x_next.detach()     # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]
        
        if return_trace:
            trace["final_x_0"] = traj[0].detach().cpu()
            return trace

        if ret_traj:
            return traj
        else:
            return traj[0]

```

## vae_guassion.py

这个文件实现的是把Pointnet中传入的PointNetEncoder进行采样并计算loss的 ELBO损失。

### 重参数化技巧

在VAE中,编码器输出的是一个概率分布的参数。

我们需要从分布中采样得到隐变量z。

但是问题就在于我们如何进行采样：

```python
z = torch.distributions.Normal(z_mu, z_sigma).sample()  # ❌ 不可导！
#采样操作会阻断梯度流，因为随机数生成器不是可微函数。这意味着梯度无法反向传播到编码器，VAE 就无法训练。

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)    # σ = exp(0.5 * logvar)
    eps = torch.randn(std.size())    # ε ~ N(0, I)
    return mean + std * eps          # z = μ + σ·ε
    



编码器 → μ, σ ──→ 加法/乘法 ──→ z
                    ↑
              ε（固定噪声，不反向传播）
```

### ELBO损失的计算

```python
    def get_loss(self, x, writer=None, it=None, kl_weight=1.0):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, z)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

```

### 采样

```python
    def sample(self, z, num_points, flexibility, truncate_std=None, initial_x_T=None, return_trace=False):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(
            num_points, 
            context=z, 
            flexibility=flexibility,
            initial_x_T=initial_x_T,
            return_trace=return_trace
        )
        return samples
```