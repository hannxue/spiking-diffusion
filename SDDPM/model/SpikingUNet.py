import math
import torch
from abc import abstractmethod
import numpy as np
from torch import nn
import torch as th
from torch.nn import init
from torch.nn import functional as F
from spikingjelly.activation_based import functional, surrogate
# from spikingjelly.activation_based.neuron import LIFNode as IFNode
# from spikingjelly.activation_based.neuron import ParametricLIFNode as IFNode
from spikingjelly.activation_based.neuron import IFNode as IFNode

import os



class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x


class Swish(nn.Module):
    def forward(self, x):
        sig = torch.sigmoid(x)
        return x * sig

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):  ## T: total step of diff; d_model: base channel num; dim:d_model*4
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb




class Spk_DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(in_ch)
        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape
        x = self.neuron(x)
        x = x.flatten(0, 1)  ## [T*B C H W]
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        _, C, H, W = x.shape
        x = self.bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class Spk_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.neuron1 = IFNode(surrogate_function=surrogate.ATan())
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.neuron2 = IFNode(surrogate_function=surrogate.ATan())
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)


        self.in_ch = in_ch
        self.out_ch = out_ch

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        #if attn:
            #self.attn = Spike_SelfAttention(out_ch)
        #else:
            #self.attn = nn.Identity()


        functional.set_step_mode(self, step_mode='m')
        #functional.set_backend(self, backend='cupy')
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        # init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        T, B, C, H, W = x.shape

        h = self.neuron1(x)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv1(h)
        h = self.bn1(h).reshape(T, B, -1, H, W).contiguous()

        #temp = self.temb_proj(temb).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, h.shape[-2], h.shape[-1])
        #，意思是「复制一份新的干净张量，不要改动原来的版本」，就能避免 PyTorch 的自动求导机制报错了。
        #h = torch.add(h, temp)
        # time embedding proj
        temb_out = self.temb_proj(temb.clone())                        # [B, C]
        temb_out = temb_out.unsqueeze(-1).unsqueeze(-1)         # [B, C, 1, 1]
        temb_out = temb_out.repeat(1, 1, H, W).contiguous().clone()  # [B, C, H, W]
        temb_out = temb_out.unsqueeze(0)                        # [1, B, C, H, W]
        h = h + temb_out 


        h = self.neuron2(h)
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv2(h)
        h = self.bn2(h).reshape(T, B, -1, H, W).contiguous()


        #h = h + self.shortcut(x.flatten(0, 1)).reshape(T, B, -1, H, W).contiguous()
        # shortcut (residual)
        shortcut_out = self.shortcut(x.flatten(0, 1))            # [T*B, C, H, W]
        shortcut_out = shortcut_out.reshape(T, B, -1, H, W).contiguous().clone()

        h = h + shortcut_out

        #h = self.attn(h)

        return h


class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    """

    def __init__(self, timestep=4) -> None:
        super().__init__()
        self.n_steps = timestep

    def forward(self, x):
        """
        x : (T,N,C,H,W)
        """

        arr = torch.arange(self.n_steps - 1, -1, -1)
        coef = torch.pow(0.8, arr).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(x.device)
        out = torch.sum(x * coef, dim=0)
        return out


class Spk_UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout, timestep, img_in_ch=3, img_out_ch=3):
        super().__init__()
        # assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        #扩散时间步 t 嵌入为高维特征
        tdim = ch * 4#128*4
        self.img_in_ch = img_in_ch
        self.img_out_ch = img_out_ch
        self.time_embedding = TimeEmbedding(T, ch, tdim)  ## T: total step of diff; ch: base channel num; tdim:ch*4
        self.timestep = timestep  ## SNN timestep
        #输入层与脉冲神经元
        self.conv = nn.Conv2d(img_in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(ch)

        self.neuron = IFNode(surrogate_function=surrogate.ATan())
        #恒等路径（残差连接）
        self.conv_identity = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        self.bn_identity = nn.BatchNorm2d(ch)
        # Downsampling 阶段
        self.downblocks = nn.ModuleList()
        chs = [ch]  #  保存每层输出通道数，用于上采样跳连
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(Spk_ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(Spk_DownSample(now_ch))
                chs.append(now_ch)
        # 中间瓶颈块 middleblocks
        self.middleblocks = nn.ModuleList([
            Spk_ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            Spk_ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])
        #Upsampling 阶段
        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(Spk_ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(Spk_UpSample(now_ch))
        assert len(chs) == 0
        #输出层（Conv + 膜输出）
        self.tail_bn = nn.BatchNorm2d(now_ch)
        self.tail_swish = Swish()
        self.tail_conv = nn.Conv2d(now_ch, img_out_ch, kernel_size=3, stride=1, padding=1)

        #跨时间建模与膜输出整合
        #Conv3D 跨时间建模：将 [T, B, C, H, W] 视作 [B, C, H, W, T] 做 3D 卷积
        self.T_output_layer = nn.Conv3d(img_out_ch, img_out_ch, kernel_size=(3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.last_bn = nn.BatchNorm2d(img_out_ch)
        self.swish = Swish()
        self.membrane_output_layer = MembraneOutputLayer(timestep=self.timestep)
        
        #多步模式（multi-step）
        functional.set_step_mode(self, step_mode='m')

        self.initialize()

    #告诉 SpikingJelly 框架使用多步输入（时间序列）
    def initialize(self):
        # init.xavier_uniform_(self.head.weight)
        # init.zeros_(self.head.bias)
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)
        init.xavier_uniform_(self.tail_conv.weight, gain=1e-5)
        init.zeros_(self.tail_conv.bias)

    def forward(self, x, t):
        #将静态图像展开成 T 帧动态脉冲序列，适配 SNN 处理
        x = x.unsqueeze(0).repeat(self.timestep, 1, 1, 1, 1)  # [T, B, C, H, W]

        # 获取扩散步嵌入（用于传入每个 ResBlock）
        temb = self.time_embedding(t)

        # Downsampling 输入编码 + 残差
        T, B, C, H, W = x.shape
        h = x.flatten(0, 1)  ## [T*B C H W]
        h = self.conv(h)
        h = self.bn(h).reshape(T, B, -1, H, W).contiguous()
        h = self.neuron(h)

        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.conv_identity(h)
        h = self.bn_identity(h).reshape(T, B, -1, H, W).contiguous()

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, Spk_ResBlock):
                h = torch.cat([h, hs.pop()], dim=2)
            h = layer(h, temb)

        T, B, C, H, W = h.shape
        h = h.flatten(0, 1)  ## [T*B C H W]
        h = self.tail_bn(h)
        h = self.tail_swish(h)
        h = self.tail_conv(h).reshape(T, B, -1, H, W).contiguous()

        h_temp = h.permute(1, 2, 3, 4, 0)  # [ B, C, H, W, T]
        h_temp = self.T_output_layer(h_temp).permute(4, 0, 1, 2, 3)   # [ T, B, C, H, W]
        h_temp = self.last_bn(h_temp.flatten(0,1)).reshape(T, B, -1, H, W).contiguous()
        h = self.swish(h_temp) + h  # [ T, B, C, H, W]

        h = self.membrane_output_layer(h)


        assert len(hs) == 0
        return h
class SuperResModel(Spk_UNet):
    """
    A UNetModel that performs super-resolution.
    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, img_ch, T, ch, ch_mult, attn, num_res_blocks, dropout, timestep, **kwargs):
        super().__init__(T=T, ch=ch, ch_mult=ch_mult, attn=attn, num_res_blocks=num_res_blocks, dropout=dropout, timestep=timestep,img_in_ch=img_ch*2, img_out_ch=img_ch, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):# 输入是高分辨率图像 x、时间步 timesteps 和低分辨率图像 low_res
        _, _, new_height, new_width = x.shape

        if low_res is not None:#低分辨率图像首先被上采样到与高分辨率图像相同的尺寸。这一步骤的目的是使低分辨率图像与高分辨率图像在空间维度上对齐，以便可以将它们合并。
                                #使用插值方法（如 F.interpolate）进行上采样，确保低分辨率图像的尺寸与高分辨率图像一致。
            upsampled = F.interpolate(low_res, (new_height, new_width), mode="nearest")
            x = th.cat([x, upsampled], dim=1)# 拼接高分辨率图像和上采样的低分辨率图像
        return super().forward(x, timesteps, **kwargs)#UNetModel 的输入是拼接后的高分辨率图像和低分辨率图像，输出是生成的高分辨率图像。
                                                     #输出是经过 U-Net 网络处理后的图像，格式为 [N x C x H x W]

if __name__ == '__main__':
    batch_size = 2
    model = Spk_UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 4], attn=[8],
        num_res_blocks=2, dropout=0.1, timestep=4).cuda()

    ## Load model
    # ckpt = torch.load(os.path.join('/home/jiahang/jiahang/Diffusion_with_spk/pytorch-ddpm/logs/threshold_test', 'snnbest.pt'))
    # model.load_state_dict(ckpt['net_model'])

    ckpt = torch.load(os.path.join('/home/jiahang/jiahang/Diffusion_with_spk/pytorch-ddpm/logs/final_thres_log/45000ckpt.pt'))
    model.load_state_dict(ckpt['net_model'])

    x = torch.randn(batch_size, 3, 32, 32).cuda()
    t = torch.randint(1000, (batch_size,)).cuda()
    # print(model)
    y = model(x, t)
    print(y.shape)
    model_size = 0
    for param in model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    functional.reset_net(model)
