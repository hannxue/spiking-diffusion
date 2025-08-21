
import torch
import torch.nn as nn
from .snn_layers import *# 导入自定义的SNN（脉冲神经网络）相关层
from .fsvae_prior import * # 导入 FS-VAE 的先验模块
from .fsvae_posterior import *  # 导入 FS-VAE 的后验模块
import torch.nn.functional as F

import global_v as glv# 导入全局变量配置文件

# 定义脉冲序列生成的变分自编码器（FSVAE）类
class FSVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, n_steps, k):
        super().__init__()

        #in_channels = glv.network_config['in_channels'] # 输入通道数
        #latent_dim = glv.network_config['latent_dim']# 潜在空间的维度
        #self.latent_dim = latent_dim
        #self.n_steps = glv.network_config['n_steps']# 时间步数

        #self.k = glv.network_config['k'] # 用于先验和后验分布的参数k
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.k = k

        hidden_dims = [32, 64, 128, 256]# 隐藏层的维度
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        is_first_conv = True# 标识是否是第一个卷积层
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike(),
                        is_first_conv=is_first_conv)
            )
            in_channels = h_dim
            is_first_conv = False# 设置为False，表明后续层不是第一个卷积层，这个变量名暗示它可能是用来指示某个操作是否是第一次进行卷积，在第一次进行卷积时，可能会执行一些初始化步骤（如权重初始化），而后续的卷积层则跳过这些步骤。
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,#4表示的是2*2，2x2x256=1024-128
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())# 编码器的最后线性层

        self.prior = PriorBernoulliSTBP(self.latent_dim,self.n_steps,self.k)  # 先验分布
        
        self.posterior = PosteriorBernoulliSTBP(self.latent_dim,self.n_steps,self.k) # 后验分布

        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()# 反转隐藏层的顺序

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
             # 第一层 tdConvTranspose：执行反卷积，将特征图尺寸放大一倍
                            tdConvTranspose(hidden_dims[-1],# 输入通道数为 hidden_dims 列表的最后一个元素
                                            hidden_dims[-1], # 输出通道数与输入通道数相同，维持通道数不变
                                            kernel_size=3,   # 反卷积核大小为 3x3
                                            stride=2,        # 步长为 2，用于放大特征图
                                            padding=1,       # 使用填充为 1，以保持特征图中心位置
                                            output_padding=1, # 在输出上额外填充，以精确控制特征图大小
                                            bias=True,        # 使用偏置项
                                            bn=tdBatchNorm(hidden_dims[-1]),  # 应用批量归一化层
                                            spike=LIFSpike()),                # 使用 LIFSpike 激活函数
             # 第二层 tdConvTranspose：将特征图还原为输入的通道数大小
                            tdConvTranspose(hidden_dims[-1],  # 输入通道数为上一层的输出通道数
                                            out_channels=1,# 输出通道数为网络配置中的输入通道数（恢复到原始输入大小
                                            kernel_size=3,  # 卷积核大小为 3x3
                                            padding=1,# 填充为 1，保持特征图中心位置
                                            bias=True,# 使用偏置项
                                            bn=None,# 最后一层不使用批量归一化
                                            spike=None)  # 最后一层不使用激活函数
        )

        self.p = 0# 初始概率p

        self.membrane_output_layer = MembraneOutputLayer() # 脉冲神经网络的膜电位输出层

        self.psp = PSP()# 突触后电位模块

    def forward(self, x, scheduled=False):
        sampled_z, q_z, p_z = self.encode(x, scheduled)
        x_recon = self.decode(sampled_z)#重构，
        return x_recon, q_z, p_z, sampled_z
    
    def encode(self, x, scheduled=False):
        #print(f"encofder前x_shape: {x.shape}")
        x = self.encoder(x) # (N,C,H,W,T)
        #start_dim=1: 指定展平的起始维度。这里设置为 1，表示从第二个维度开始展平（第一个维度通常是 batch size）。
        #end_dim=3: 指定展平的结束维度。这里设置为 3，表示在第四个维度结束展平。
        #print(f"encoder后x_shape: {x.shape}")
        x= torch.flatten(x, start_dim=1, end_dim=3) # (N,C*H*W,T)
        latent_x = self.before_latent_layer(x) # (N,latent_dim,T)#即每个时间步都有一个 latent_dim 的特征向量。
        #print(f"flatten后latent_x_shape: {latent_x.shape}")
        sampled_z, q_z = self.posterior(latent_x) # sampled_z:(B,C,T)(B,C,1,1,T), q_z:(B,C,k,T)
        #print(f"sampled_z: {sampled_z.shape}")
        #print(f"q_z: {q_z.shape}")
        p_z = self.prior(sampled_z, scheduled, self.p)#z只有计划采样的p_z(B,C,k,T)
        return sampled_z, q_z, p_z

    def decode(self, z):
        result = self.decoder_input(z) # (B,C,T)-(N,C*H*W,T)C*H*W=256*2*2
        result = result.view(result.shape[0], self.hidden_dims[-1], 2, 2, self.n_steps) # (N,C,H,W,T)
        result = self.decoder(result)# (N,C,H,W,T)
        result = self.final_layer(result)# (N,C,H,W,T)
        out = torch.tanh(self.membrane_output_layer(result)) #spiking to image decoding       
        return out

    def sample(self, batch_size=64):
        sampled_z = self.prior.sample(batch_size)
        sampled_x = self.decode(sampled_z)
        return sampled_x, sampled_z
        
    def loss_function_mmd(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        q_z_ber = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        p_z_ber = torch.mean(p_z, dim=2) # (N, latent_dim, T)

        #kld_loss = torch.mean((q_z_ber - p_z_ber)**2)
        mmd_loss = torch.mean((self.psp(q_z_ber)-self.psp(p_z_ber))**2)
        loss = recons_loss + mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': mmd_loss}

    def loss_function_kld(self, input_img, recons_img, q_z, p_z):
        """
        q_z is q(z|x): (N,latent_dim,k,T)
        p_z is p(z): (N,latent_dim,k,T)
        """
        recons_loss = F.mse_loss(recons_img, input_img)
        prob_q = torch.mean(q_z, dim=2) # (N, latent_dim, T)
        prob_p = torch.mean(p_z, dim=2) # (N, latent_dim, T)
        
        kld_loss = prob_q * torch.log((prob_q+1e-2)/(prob_p+1e-2)) + (1-prob_q)*torch.log((1-prob_q+1e-2)/(1-prob_p+1e-2))
        kld_loss = torch.mean(torch.sum(kld_loss, dim=(1,2)))

        loss = recons_loss + 1e-4 * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'Distance_Loss': kld_loss}
    def weight_clipper(self):
        with torch.no_grad():
            for p in self.parameters():
                p.data.clamp_(-4,4)

    def update_p(self, epoch, max_epoch):
        init_p = 0.1
        last_p = 0.3
        self.p = (last_p-init_p) * epoch / max_epoch + init_p
        

class FSVAELarge(FSVAE):
    def __init__(self):
        super(FSVAE, self).__init__()
        in_channels = glv.network_config['in_channels']
        latent_dim = glv.network_config['latent_dim']
        self.latent_dim = latent_dim
        self.n_steps = glv.network_config['n_steps']

        self.k = glv.network_config['k']

        hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                tdConv(in_channels,
                        out_channels=h_dim,
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        bias=True,
                        bn=tdBatchNorm(h_dim),
                        spike=LIFSpike())
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.before_latent_layer = tdLinear(hidden_dims[-1]*4,
                                            latent_dim,
                                            bias=True,
                                            bn=tdBatchNorm(latent_dim),
                                            spike=LIFSpike())

        self.prior = PriorBernoulliSTBP(self.k)
        
        self.posterior = PosteriorBernoulliSTBP(self.k)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = tdLinear(latent_dim, 
                                        hidden_dims[-1] * 4, 
                                        bias=True,
                                        bn=tdBatchNorm(hidden_dims[-1] * 4),
                                        spike=LIFSpike())
        
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                    tdConvTranspose(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=3,
                                    stride = 2,
                                    padding=1,
                                    output_padding=1,
                                    bias=True,
                                    bn=tdBatchNorm(hidden_dims[i+1]),
                                    spike=LIFSpike())
            )
        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            tdConvTranspose(hidden_dims[-1],
                                            hidden_dims[-1],
                                            kernel_size=3,
                                            stride=2,
                                            padding=1,
                                            output_padding=1,
                                            bias=True,
                                            bn=tdBatchNorm(hidden_dims[-1]),
                                            spike=LIFSpike()),
                            tdConvTranspose(hidden_dims[-1], 
                                            out_channels=glv.network_config['in_channels'],
                                            kernel_size=3, 
                                            padding=1,
                                            bias=True,
                                            bn=None,
                                            spike=None)
        )

        self.p = 0

        self.membrane_output_layer = MembraneOutputLayer()

        self.psp = PSP()