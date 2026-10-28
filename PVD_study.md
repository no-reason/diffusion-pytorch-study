# PVD Study

                       时间嵌入 t → MLP
                              │
                              ▼
输入带噪点云 Xₜ (N×3)
       │
       ▼
┌────────────────────────────────── Encoder ──────────────────────────────────┐
│                                                                            │
│ [PVConv+SA1] ──→ [PVConv+SA2] ──→ [PVConv+SA3] ──→ [SA4]                   │
│  2048→1024        1024→256         256→64          64→16                  │
│       │               │               │                │                  │
└───────┼───────────────┼───────────────┼────────────────┼──────────────────┘
        │               │               │                ▼
        │               │               │         [Global Attention]
        │               │               │                │
        │               │               │                ▼
        │               │               └───→ [FP+PVConv] 16→64
        │               │                          │
        │               └──────────────────→ [FP+PVConv] 64→256
        │                                          │
        └──────────────────────────────────→ [FP+PVConv] 256→1024
                                                   │
                                                   ▼
                                         [FP+PVConv] 1024→2048
                                                   │
                                                   ▼
                                               输出 MLP
                                                   │
                                                   ▼
                                           εθ(Xₜ,t) ∈ R^(N×3)

## 上采样和下采样


### 下采样 （SA）
(可以理解成一种特征聚合,可能原本有很多点，做完下采样后就是用更少的点可以描述原本的结构)

但是很多时候会额外附带特征聚合(就是虽然点的数量减少，但是每个点的特征感受视野增大了（存储的特征的维数增大了）)



1. FPS （Farthest Point Sampling，最远点采样）:
**找出中心点**

尽可能让选出的点代表整个点云而不是集中在一小块区域。
假设我们已经选中了点集 \(S\)，那么下一次选择
\[
\boxed{
x_{\mathrm{new}}
=\arg\max_{x_i\notin S}
\min_{c\in S}\|x_i-c\|_2
}
\]

2. Ball Query(找到需要被聚合信息的点):

找到一个球形邻域，将邻域内的点都视为需要被聚合信息的点.

3. PVCNN++进行特征聚合:

两条分支

1. PointNet++

2. PVconv
把点云根据位置关系转化为体素,体素的特征聚合则通过3D CNN实现。



### 上采样

将第分辨率的特征重新传播会高分辨率位置

1. 通过线性插值将对应点的位置找到

2. 恢复每个点的信息

在下采样阶段就将具体的特征信息存储下来，然后在这个时候直接将存储的特征和下采样的分支做合并。

再通过MLP将两类特征融合


除了上下采样之外PVD还有一个实现特征更加深入融合的方法:Attention

假设当前层的特征矩阵为：
\[
F\in\mathbb R^{N\times C}.
\]

先计算：
\[
Q=FW_Q,\qquad
K=FW_K,\qquad
V=FW_V.
\]

再计算：
\[
\boxed{
\operatorname{Attention}(F)=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}
\right)V
}
\]

它的含义是：对于某个位置 \(i\)，通过注意力权重对其他位置的特征进行加权聚合。
\[
f_i'=
\sum_j a_{ij}v_j.
\]

## 




## PVCNN2Base

```python

class PVCNN2Base(nn.Module):

    def __init__(self, num_classes, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.sa_layers = nn.ModuleList(sa_layers)

        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier
        )
        self.fp_layers = nn.ModuleList(fp_layers)


        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, dropout, num_classes], # was 0.5
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)

        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def get_timestep_embedding(self, timesteps, device):
        assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, inputs, t):

        temb =  self.embedf(self.get_timestep_embedding(t, inputs.device))[:,:,None].expand(-1,-1,inputs.shape[-1])

        # inputs : [B, in_channels + S, N]
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        for i, sa_blocks  in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks ((features, coords, temb))
            else:
                features, coords, temb = sa_blocks ((torch.cat([features,temb],dim=1), coords, temb))
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        if self.global_att is not None:
            features = self.global_att(features)
        for fp_idx, fp_blocks  in enumerate(self.fp_layers):
            features, coords, temb = fp_blocks((coords_list[-1-fp_idx], coords, torch.cat([features,temb],dim=1), in_features_list[-1-fp_idx], temb))

        return self.classifier(features)



```


# SPVD


                             SPVD 噪声预测网络

输入带噪点云 Xₜ (N×3)
        │
        ▼
 [时间嵌入 t]
        │
        ▼
 [PointTensor] → [Initial Voxelize]
                         │
                         ▼
┌────────────────────────────── Encoder ──────────────────────────────────────┐
│                                                                            │
│ [Sparse Conv] ──→ [Strided Conv] ──→ [Strided Conv] ──→ [Strided Conv]      │
│  + ResBlock           ↓                 ↓                 ↓                │
│   分辨率 R           R/2               R/4               R/8              │
│       │               │                 │                 │                │
└───────┼───────────────┼─────────────────┼─────────────────┼────────────────┘
        │               │                 │                 ▼
        │               │                 │             [Mid Block]
        │               │                 │                 │
        │               │                 │                 ▼
        │               │                 └──→ [Sparse Transposed Conv] R/8→R/4
        │               │                                    │
        │               └─────────────────→ [Sparse Transposed Conv] R/4→R/2
        │                                                    │
        └─────────────────────────────────→ [Sparse Transposed Conv] R/2→R
                                                             │
                                                             ▼
                                                    [Sparse ResBlocks]
                                                             │
                                                             ▼
                                                      [Voxel-to-Point]
                                                             │
                                                             ▼
                                                   恢复 N 个点的特征
                                                             │
                                                             ▼
                                                       [Linear Head]
                                                             │
                                                             ▼
                                                   εθ(Xₜ,t) ∈ R^(N×3)