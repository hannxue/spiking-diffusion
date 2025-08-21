import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import global_v as glv

from .snn_layers import *

# 定义后验分布的 Bernoulli 估计器类，使用脉冲神经网络的反向传播（STBP）
class PosteriorBernoulliSTBP(nn.Module):
    def __init__(self, latent_dim, n_steps,k) -> None:
        """
        modeling of q(z_t | x_<=t, z_<t)#建立 q(z_t | x_<=t, z_<t) 的模型
        """
        super().__init__()
        #self.channels = glv.network_config['latent_dim']# 获取潜在维度的通道数，
        self.channels = latent_dim
        self.k = k # 离散采样的数量
        #self.n_steps = glv.network_config['n_steps']# 时间步数
        self.n_steps = n_steps
       # 构建包含时序线性层、批归一化和LIF脉冲激活的序列模块
        self.layers = nn.Sequential(
            tdLinear(self.channels*2,
                    self.channels*2,
                    bias=True,
                    bn=tdBatchNorm(self.channels*2, alpha=2), 
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

        self.is_true_scheduled_sampling = True# 是否使用计划采样

    def forward(self, x):
        """
        input: 
            x:(B,C,T) x: (B, C, T)  输入张量，B为批次大小，C为通道数，T为时间步数
        returns: 
            sampled_z:(B,C,T)采样后的潜变量
            q_z: (B,C,k,T) # indicates q(z_t | x_<=t, z_<t) (t=1,...,T)
        """
        x_shape = x.shape # (B,C,T) # 获取输入张量的形状
        batch_size=x_shape[0]
        random_indices = []
        # sample z inadvance without gradient# 预先采样 z，且不计算梯度
        with torch.no_grad():#initial_input(1, C, 1) 
            z_t_minus = self.initial_input.repeat(x_shape[0],1,1) # z_<t z0=zeros:(B,C,1)初始为全零
            for t in range(self.n_steps-1):
                 # 拼接 x_<=t 和 z_<t，形状为 (B, C + C, t+1)；[..., :t+1] 是一种切片方式，表示取所有批次和通道的数据，但时间步只取从 0 到 t（共 t+1 个时间步）
                 #.detach() 方法用于从计算图中分离出张量，使其不再参与后续的反向传播。
                inputs = torch.cat([x[...,:t+1].detach(), z_t_minus.detach()], dim=1) #如果同t=0,，x取前t+1=1个时间步，x维度为(B,C,1)，z_t_minus 初始化为全零，形状为 (B, C, 1)。最后(B, 2 * C, 1)
                # 通过网络计算输出，形状为 (B, C*k, t+1)
                outputs = self.layers(inputs) #(B, C*k, t+1) #后验输出
                # 取最后一个时间步的输出 (B, C*k, 1)
                q_z_t = outputs[...,-1] # (B, C*k, 1) q(z_t | x_<=t, z_<t) 
                
                # sampling from q(z_t | x_<=t, z_<t)# 从 q(z_t | x_<=t, z_<t) 中采样
                #生成一个形状为 (batch_size * self.channels,) 的张量，其中每个元素是从 [0, self.k-1] 区间内随机选取的整数。
                #生成一个形状为 (batch_size * self.channels,) 的张量，起始值是 0，结束值是 batch_size * self.channels * self.k，步长为 self.k
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k) #(B*C,) select 1 from every k value每个样本的每个通道中，从 k 个值中随机选取一个值
                random_index = random_index.to(x.device)#random_index 张量移动到与输入张量 x 相同的设备
                random_indices.append(random_index)

                z_t = q_z_t.view(batch_size*self.channels*self.k)[random_index] # (B*C,)#将 q_z_t 展平为一维张量，形状为 (batch_size * channels * k,),使用了 random_index 进行索引
                z_t = z_t.view(batch_size, self.channels, 1) #(B,C,1)

                z_t_minus = torch.cat([z_t_minus, z_t], dim=-1) # (B,C,t+2)#(B,C,2)

        z_t_minus = z_t_minus.detach() # (B,C,T) z_0,...,z_{T-1}
        #这种方式适合一次性地生成所有时间步的后验分布 q(z_{0:T-1} | x, z)，而不是逐步生成。
        q_z = self.layers(torch.cat([x, z_t_minus], dim=1)) # (B,C*k,T)#这可以理解为最终的预测或编码结果
        #拼接 x 和 z_t_minus 是为了在时间序列模型中使用历史信息来建模序列的潜在表示。将 x 和 z_t_minus 连接起来后，模型可以基于输入数据和过去的采样来生成潜在分布 q(z_t | x_<=t, z_<t)。
        
        # input z_t_minus again to calculate tdBN 重新输入 z_t_minus 以计算 tdBN
        sampled_z = None
        for t in range(self.n_steps):
            
            if t == self.n_steps-1:
                # when t=T # 当 t=T 时，生成新的随机索引
                #可能是因为需要为最终的潜在变量生成一个新的分布采样，让最后的输出与之前时间步的采样有稍微不同的倾向。
                random_index = torch.randint(0, self.k, (batch_size*self.channels,)) \
                            + torch.arange(start=0, end=batch_size*self.channels*self.k, step=self.k)
                random_indices.append(random_index)
            else:
                # when t<=T-1  # 当 t<=T-1 时，使用预先生成的随机索引
                random_index = random_indices[t]

            # sampling采样 z_t，形状为 (B, C, 1)
            sampled_z_t = q_z[...,t].view(batch_size*self.channels*self.k)[random_index] # (B*C,)
            sampled_z_t = sampled_z_t.view(batch_size, self.channels, 1) #(B,C,1)
            if t==0:
                sampled_z = sampled_z_t
            else:
                sampled_z = torch.cat([sampled_z, sampled_z_t], dim=-1)# (B,C,T)
        # 调整 q_z 的形状为 (B, C, k, T)        
        q_z = q_z.view(batch_size, self.channels, self.k, self.n_steps)# (B,C,k,T)

        return sampled_z, q_z