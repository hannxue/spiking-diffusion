import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from .spiking_layer import Spiking, last_Spiking, IF, Spiking_TimeEmbed
from .quant_layer import QuantSwish

class S_TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim, bit, spike_time: int = 3):
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

        self.timembedding = Spiking_TimeEmbed(nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            IF(),
            nn.Linear(dim, dim),
        ), spike_time, alpha_loc = 2)
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()
        self.idem = False

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        if self.idem:
            return x
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.main(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()
        self.idem = False

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        if self.idem:
            return x
        B, T, C, H, W = x.shape
        x = x.flatten(0, 1)  ## [T*B C H W]
        # _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        _, C, H, W = x.shape
        x = x.reshape(B, T, C, H, W).contiguous()
        return x

class S_ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, bit, spike_time: int = 3):
        super().__init__()
        self.idem = False
        self.block1 = Spiking(nn.Sequential(
            nn.GroupNorm(32, in_ch),
            IF(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        ), spike_time, alpha_loc=1)
        self.temb_proj = Spiking(nn.Sequential(
            IF(),
            nn.Linear(tdim, out_ch),
        ),spike_time, alpha_loc=0)
        self.block2 = Spiking(nn.Sequential(
            nn.GroupNorm(32, out_ch),
            IF(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        ), spike_time, alpha_loc=1)
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2.block[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        if self.idem:
            return x
        B, T, C, H, W = x.shape
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x.flatten(0,1)).reshape(B, T, -1, H, W).contiguous()
        return h
# ----------------------------
# 脉冲版注意力模块
# ----------------------------
class S_AttentionBlock(nn.Module):
    """
    脉冲版注意力模块，用于对空间位置进行注意力计算。
    输入为形状 [B, T, C, H, W]，内部先将 B 与 T 合并，计算注意力后恢复。
    """
    def __init__(self, channels, num_heads=1, spike_time=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.spike_time = spike_time
        # 使用 GroupNorm 对输入做归一化
        self.norm = nn.GroupNorm(32, channels)
        # 使用 Spiking 包装的 1x1 卷积来生成 QKV，输出通道数为 3 * channels
        self.qkv = Spiking(nn.Conv2d(channels, channels * 3, kernel_size=1), spike_time, alpha_loc=1)
        # 使用 Spiking 包装的 1x1 卷积做投影
        self.proj_out = Spiking(nn.Conv2d(channels, channels, kernel_size=1), spike_time, alpha_loc=1)

    def forward(self, x, temb=None):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        # 将时间步和 batch 合并，形状变为 [B*T, C, H, W]
        x_reshaped = x.view(B * T, C, H, W)
        h = self.norm(x_reshaped)
        # 生成 QKV，形状为 [B*T, 3C, H, W]
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        # 将空间展平，并将通道分成多头
        q = q.view(B * T, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B * T, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B * T, self.num_heads, C // self.num_heads, H * W)
        # 缩放 q
        scale = 1 / math.sqrt(C // self.num_heads)
        q = q * scale
        # 计算注意力权重，注意力矩阵形状为 [B*T, num_heads, H*W, H*W]
        attn = torch.matmul(q.transpose(-2, -1), k)
        attn = F.softmax(attn, dim=-1)
        # 加权求和得到输出
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1)
        # 将多头合并，恢复成 [B*T, C, H, W]
        out = out.reshape(B * T, C, H, W)
        # 通过投影层
        out = self.proj_out(out)
        # 残差连接
        out = x_reshaped + out
        # 恢复时序维度，输出形状 [B, T, C, H, W]
        out = out.view(B, T, C, H, W)
        return out

class S_UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout, bit=32, spike_time: int = 8, attn_heads=1):
        super().__init__()
        tdim = ch * 4
        self.bit=bit
        self.spike_time = spike_time
        self.time_embedding = S_TimeEmbedding(T, ch, tdim, self.bit, self.spike_time)

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(S_ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit, spike_time=self.spike_time))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            S_ResBlock(now_ch, now_ch, tdim, dropout, self.bit, self.spike_time),
            S_AttentionBlock(now_ch, num_heads=attn_heads, spike_time=self.spike_time),
            S_ResBlock(now_ch, now_ch, tdim, dropout, self.bit, self.spike_time),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(S_ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, bit=self.bit, spike_time=self.spike_time))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = last_Spiking(nn.Sequential(
            nn.GroupNorm(32, now_ch),
            IF(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)), self.spike_time, alpha_loc=1)
        self.initialize()

        self.timembedding_idem = False
        self.head_idem = False
        self.tail_idem = False
        self.upblocks_target = False
    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail.block[-1].weight, gain=1e-5)
        init.zeros_(self.tail.block[-1].bias)

    def forward(self,x, t, z=None, y=None, **kwargs):
        if self.timembedding_idem:
            return t
        x_in = x
        t_in = t
        x_in = x_in.unsqueeze(1)
        x_in = x_in.repeat(1, self.spike_time, 1, 1, 1)
        # Timestep embedding
        temb = self.time_embedding(t_in)
        # Downsampling
        if self.head_idem:
            return temb
        train_shape = [x_in.shape[0], x_in.shape[1]]
        x_in = x_in.flatten(0, 1)
        h = self.head(x_in)
        train_shape.extend(h.shape[1:])
        h = h.reshape(train_shape)

        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for (idx,layer) in enumerate(self.upblocks):
            if layer.idem:
                if self.middleblocks[1].idem:
                    h = layer(h, temb)
                else:
                    if idx == 0 or (not self.upblocks[idx-1].idem and layer.idem):
                        if not self.upblocks_target:
                            if isinstance(layer, S_ResBlock):
                                h = torch.cat([h, hs.pop()], dim=2)
                            h = layer(h, temb)
                        else:
                            h = layer(h, temb)
                    else:
                        h = layer(h, temb)

            else:
                if isinstance(layer, S_ResBlock):
                    h = torch.cat([h, hs.pop()], dim=2)
                h = layer(h, temb)
        if self.tail_idem:
            return h
        h = self.tail(h)
        return h

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantSwish):
                m.show_params()
class S_SuperResModel(S_UNet):
    """
    A spiking UNet that performs super-resolution.
    与原 SuperResModel 类似：需要一个额外的 low_res 输入，形状 [B,3,H_low,W_low]。
    不增加额外的采样或残差模块，而是复用父类 S_UNet 的时序逻辑。
    """
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout, bit=32, spike_time=8):
        """
        :param T: 总扩散步数(与 S_UNet 相同)
        :param ch: 基础通道数
        :param ch_mult: 通道倍数 (比如 [1,2,2,2])
        :param num_res_blocks: 每个level的残差块数量
        :param dropout: dropout比例
        :param bit: 量化位宽(如果你需要量化，可保留)
        :param spike_time: 脉冲时间步
        """
        # 先调用 S_UNet 的构造函数来创建完整的脉冲U-Net结构
        super().__init__(T, ch, ch_mult, num_res_blocks, dropout, bit, spike_time)

        # 由于要在输入阶段拼接高分辨率与低分辨率(3+3=6 通道)，
        # 这里将 head 改成从 6 通道 -> ch
        self.head = nn.Conv2d(6, ch, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, t, low_res=None, **kwargs):
        """
        :param x: 高分辨率图像 [B,3,H,W]
        :param t: 时间步张量 [B]
        :param low_res: 低分辨率图像 [B,3,H_low,W_low] (可选)
        :return: 超分结果 [B,3,H,W]
        """
        # 如果给了 low_res，就先上采样到与 x 相同的大小，然后通道维拼接
        if low_res is not None:
            B, _, H, W = x.shape
            upsampled = F.interpolate(low_res, size=(H, W), mode="nearest")
            # 拼接得到 [B,6,H,W]
            x = torch.cat([x, upsampled], dim=1)
        # 然后调用父类(S_UNet)的 forward：其内部会做多时刻脉冲处理，并在最后返回 [B,3,H,W]
        return super().forward(x, t,**kwargs)

if __name__ == '__main__':
    import re
    import sys
    batch_size = 1
    model = S_UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2],num_res_blocks=2, dropout=0.1, bit=3)
    # load model and evaluate

    new_state_dict = {}
    old_state_dict = torch.load('./logs/RELU_QUANT2B_DDPM_CIFAR10_EPS/SNN_2.0bit_ft/ft_blocks_samples/model_best.pt')['ema_model']
    for key in old_state_dict:
        parts = key.split('.')
        if 'main' in key or 'shortcut' in key or 'head' in key:
            new_key = key
        else:
            parts.insert(-2, 'block')
            new_key = '.'.join(parts)
        new_state_dict[new_key] = old_state_dict[key]
    
    model.load_state_dict(old_state_dict, strict=True)
    print("Successfully load weight!")
    print(model.state_dict()['time_embedding.timembedding.block.2.act_alpha'])

    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)
    print(y.shape)
    print(x.shape, t.shape)

    print(model.tail.idem)


    m=getattr(model,'downblocks')
    print(old_state_dict.keys())
    print(model.head_idem,model.time_embedding,model.tail_idem)
