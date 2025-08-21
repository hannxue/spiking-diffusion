from argparse import ZERO_OR_MORE
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T

import global_v as glv

from .snn_layers import *


class PriorBernoulliSTBP(nn.Module):
    def __init__(self, latent_dim, n_steps,k) -> None:
        """
        modeling of p(z_t|z_<t)建立 p(z_t|z_<t) 的模型
        """
        super().__init__()
        #self.channels = glv.network_config['latent_dim']# 获取潜在维度的通道数
        self.channels = latent_dim
        self.k = k
        #self.n_steps = glv.network_config['n_steps'] # 时间步数
        self.n_steps = n_steps

        # 构建包含时序线性层、批归一化和LIF脉冲激活的序列模块
        self.layers = nn.Sequential(
            tdLinear(self.channels,
                    self.channels*2,
                    bias=True,
                    bn=tdBatchNorm(self.channels*2, alpha=2), #自定义的批量归一化层
                    spike=LIFSpike()),
            tdLinear(self.channels*2,
                    self.channels*4,
                    bias=True,
                    bn=tdBatchNorm(self.channels*4, alpha=2),
                    spike=LIFSpike()),
            tdLinear(self.channels*4,
                    self.channels*k,
                    bias=True,
                    bn=tdBatchNorm(self.channels*k, alpha=2),
                    spike=LIFSpike())
        )
        # 注册一个初始输入的缓冲区，用于保存形状为 (1, C, 1) 的零张量
        self.register_buffer('initial_input', torch.zeros(1, self.channels, 1))# (1,C,1)


    def forward(self, z, scheduled=False, p=None):#前向传播函数，支持计划采样
        if scheduled:
            return self._forward_scheduled_sampling(z, p)
        else:
            return self._forward(z)
    
    def _forward(self, z):
        """
        input z: (B,C,T) # latent spike sampled from posterior 输入 z: (B, C, T)，后验中采样的潜在脉冲
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T) 输出 : (B, C, k, T)，表示 p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (B,C,T)
        batch_size = z_shape[0]
        z = z.detach() # 不进行梯度跟踪

        z0 = self.initial_input.repeat(batch_size, 1, 1) # (B,C,1)
        # 拼接 z0 和 z 的前 T-1 个时间步，得到输入 (B, C, T)
        inputs = torch.cat([z0, z[...,:-1]], dim=-1) # (B,C,T)从张量 z 中选择除了最后一个时间步之外的所有时间步，保留其他维度（批大小和通道数）不变。例如，如果 z 的形状是 (B, C, T)，那么 z[..., :-1] 的形状将是 (B, C, T-1)。
        outputs = self.layers(inputs) # (B,C*k,T)
        # 将输出形状调整为 (B, C, k, T)
        p_z = outputs.view(batch_size, self.channels, self.k, self.n_steps) # (B,C,k,T)
        return p_z

    def _forward_scheduled_sampling(self, z, p):#此函数主要用于训练阶段的“计划采样” (scheduled sampling)。
        """
        use scheduled sampling
        input 
            z: (B,C,T) # latent spike sampled from posterior z: (B, C, T)，后验中采样的潜在脉冲
            p: float # prob of scheduled sampling计划采样的概率
        output : (B,C,k,T) # indicates p(z_t|z_<t) (t=1,...,T)
        """
        z_shape = z.shape # (B,C,T)
        #print(f"z_shape: {z_shape}")
        batch_size = z_shape[0]
        z = z.detach() # 不进行梯度跟踪
        # 创建初始输入 z_t_minus，其形状为 (B, C, 1)
        z_t_minus = self.initial_input.repeat(batch_size,1,1) # z_<t, z0=zeros:(B,C,1)
        if self.training:
            with torch.no_grad():
                 # 遍历每个时间步
                for t in range(self.n_steps-1): 
                    #print(f"t = {t}, z_t_minus.shape = {z_t_minus.shape}")
                    if t>=5 and random.random() < p: # scheduled sampling  # 使用计划采样   #t=6时，
                        #当时间步 t 大于等于 5 且一个随机数小于概率 p 时，才进行“计划采样”  这通常是一个训练阶段后期的阶段
                        # 如果随机数小于 p，模型将使用自身预测进行采样；否则，它使用真实值。  引入随机性以避免过拟合           
                        outputs = self.layers(z_t_minus.detach()) #binary (B, C*k, t+1) z_<=t  #z_t_minus=（B,C,5+2)-(B, C*k, 6+1)
                        #print(f"outputs: {outputs.shape}")
                        p_z_t = outputs[...,-1] # (B, C*k, 1)#提取最后一个时间步
                        #print(f"p_z_t: {p_z_t.shape}")
                        # sampling from p(z_t | z_<t)
                        #将 p_z_t 调整形状为 (batch_size, channels, k)，表示每个样本的每个通道有 k 个候选概率值。对 k 这个维度取平均值后，得到 (batch_size, channels) 的张量 prob1
                        #每个元素 prob1[b, c] 表示在样本 b 的通道 c 上 的激活平均概率。
                        prob1 = p_z_t.view(batch_size, self.channels, self.k).mean(-1) # (B,C)求均值会将 (B, C, k) 中的 k 个值压缩成一个标量，这样每个 (B, C) 位置就只剩下一个均值。
                        #prob1 上添加小的随机噪声 1e-3 * torch.randn_like(prob1) 是为了在激活决策时引入一些随机性，使得模型更具鲁棒性。
                        prob1 = prob1 + 1e-3 * torch.randn_like(prob1)#torch.randn_like(prob1) 是 PyTorch 中用于生成一个与 prob1 形状相同的张量，且其中的元素为从标准正态分布（均值为 0，标准差为 1）中随机采样的值。 
                        #z_t = (prob1 > 0.5).float() 将 prob1 中大于 0.5 的值置为 1（激活），小于等于 0.5 的值置为 0（未激活），实现二值化采样。
                        z_t = (prob1>0.5).float() #二值化采样 (B,C)
                        z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
                        z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)#（B,C,6+2)
                    else:#从后验中选取
                        z_t_minus = torch.cat([z_t_minus, z[...,t].unsqueeze(-1)], dim=-1) # (B,C,t+2)#t=0时，z_t_minus的维度（B,C,1+1)=(B,C,0+2)
        else:
            z_t_minus = torch.cat([z_t_minus, z[:,:,:-1]], dim=-1) # (B,C,T)

        z_t_minus = z_t_minus.detach() # (B,C,T) z_{<=T-1} 
        p_z = self.layers(z_t_minus) # (B,C*k,T),p_z 表示整个时间序列上的潜在分布 p(z1,T),即对于每一个时间步的全部输出分布。
        #print(f"p_z shape before view: {p_z.shape}, expected total elements: {batch_size * self.channels * self.k * self.n_steps}")
        #print(f"batch_size: {batch_size}, channels: {self.channels}, k: {self.k}, n_steps: {self.n_steps}")

        p_z = p_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)这里有问题
        
        return p_z
    # 此函数主要用于推理阶段的采样，它逐步从条件分布中生成潜在变量序列 sampled_z，作为模型在测试阶段的输出。
    #用于推理时的纯采样，通过逐步递归生成潜在变量序列，不依赖真实样本
    def sample(self, batch_size=64):
        z_minus_t = self.initial_input.repeat(batch_size, 1, 1) # (B, C, 1)
        for t in range(self.n_steps):
            outputs = self.layers(z_minus_t) # (B, C*k, t+1)#表示每个时间步t上，条件分布p(Z_t|Z<t)的分布
            p_z_t = outputs[...,-1] # (B, C*k, 1)

            random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) pick one from k
            random_index = random_index.to(z_minus_t.device)

            z_t = p_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)
            z_minus_t = torch.cat([z_minus_t, z_t], dim=-1) # (B,C,t+2)

        
        sampled_z = z_minus_t[...,1:] # (B,C,T)提取 z_minus_t 中的有效部分（去掉第一个时间步）

        return sampled_z

