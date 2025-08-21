import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import global_v as glv

dt = 5
a = 0.25
aa = 0.5  
Vth = 0.2
tau = 0.25


class SpikeAct(torch.autograd.Function):
    """ 
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu

class LIFSpike(nn.Module):
    """
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        nsteps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(nsteps):
            u, out[..., step] = self.state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out
    
    def state_update(self, u_t_n1, o_t_n1, W_mul_o_t1_n, tau=tau):
        u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
        o_t1_n1 = SpikeAct.apply(u_t1_n1)
        return u_t1_n1, o_t1_n1  

class tdLinear(nn.Linear):
    def __init__(self, 
                in_features,
                out_features,
                bias=True,
                bn=None,
                spike=None):
         # 确保输入和输出特征的维度为整数
        assert type(in_features) == int, 'inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape)
        assert type(out_features) == int, 'outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape)
         # 调用父类 nn.Linear 的构造函数
        super(tdLinear, self).__init__(in_features, out_features, bias=bias)
        
        # 可选的批量归一化层和脉冲激活层
        self.bn = bn
        self.spike = spike
        

    def forward(self, x):
        """
        x : (N,C,T) x : 输入张量 (N, C, T)，其中 N 是批次大小，C 是通道数，T 是时间步数
        """        
        x = x.transpose(1, 2) #  (N, C, T) -> (N, T, C)
        y = F.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)# (N, T, C) -> (N, C, T)
        
        # 如果有批量归一化层，则应用批量归一化
        if self.bn is not None:
            y = y[:,:,None,None,:]# 添加两个维度，变为 (N, C, 1, 1, T)
            y = self.bn(y) # 执行批量归一化
            y = y[:,:,0,0,:]# 恢复原始形状 (N, C, T)
        # 如果有脉冲激活层，则应用脉冲激活    
        if self.spike is not None:
            y = self.spike(y)
        return y

class tdConv(nn.Conv3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                dilation=1,# 膨胀系数,意味着卷积核的每个元素之间没有额外的间隔
                groups=1,# 分组数
                bias=True, # 是否使用偏置
                bn=None,# 批归一化层
                spike=None, # 脉冲层
                is_first_conv=False): # 是否为第一个卷积层

        # kernel设置卷积核大小
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)# 将二维核大小扩展为三维
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)# 将2维核扩展为3维
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride设置步长
        if type(stride) == int:
            stride = (stride, stride, 1)# 扩展为三维步长
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding# 设置填充
        if type(padding) == int:
            padding = (padding, padding, 0)# 扩展为三维填充
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation设置膨胀系数
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)# 扩展为三维膨胀
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))
        # 调用父类的初始化方法
        super(tdConv, self).__init__(in_channels, out_channels, kernel, stride, padding, dilation, groups,
                                        bias=bias)
        self.bn = bn            # 批归一化层
        self.spike = spike      # 脉冲层
        self.is_first_conv = is_first_conv# 标记是否为第一层卷积

    def forward(self, x):
       # 前向传播函数，定义输入数据 x 的处理方式
       # 应用 3D 卷积

        x = F.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
        
        # 如果存在批归一化层，则应用批归一化
        if self.bn is not None:
            x = self.bn(x)

         # 如果存在脉冲激活层，则应用脉冲激活
        if self.spike is not None:
            x = self.spike(x)
        return x
        

class tdConvTranspose(nn.ConvTranspose3d):
    def __init__(self, 
                in_channels, 
                out_channels,  
                kernel_size,
                stride=1,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=1,
                bias=True,
                bn=None,
                spike=None):

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))


        # output padding
        if type(output_padding) == int:
            output_padding = (output_padding, output_padding, 0)
        elif len(output_padding) == 2:
            output_padding = (output_padding[0], output_padding[1], 0)
        else:
            raise Exception('output_padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        super().__init__(in_channels, out_channels, kernel, stride, padding, output_padding, groups,
                                        bias=bias, dilation=dilation)

        self.bn = bn
        self.spike = spike

    def forward(self, x):
        x = F.conv_transpose3d(x, self.weight, self.bias,
                        self.stride, self.padding, 
                        self.output_padding, self.groups, self.dilation)

        if self.bn is not None:
            x = self.bn(x)
        if self.spike is not None:
            x = self.spike(x)
        return x

class tdBatchNorm(nn.BatchNorm2d):
    """
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
        assert input is not None, "Input tensor is None"
        assert len(input.shape) >= 5, f"Expected 5D input, but got shape {input.shape}"
        #print(f"Input shape: {input.shape}")#[N, C, H, W, T]
        #print(f"Input device: {input.device}")

        #print(input.shape)
        assert input.is_cuda, "Input tensor is not on CUDA"
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            #print("Before mean calculation")
            #mean = input.mean([0, 2, 3, 4])  # 检查这一步是否出错
            #print("After mean calculation")
            input_cpu = input.to('cpu')  # 或 input_cpu = input.cpu()

            # Perform the same operation on CPU
            mean = input_cpu.mean([0, 2, 3, 4])
            #print(f"mean shape: {mean.shape}")
            #print("Mean calculated on CPU:", mean)

            # use biased var in train
            #var = input.var([0, 2, 3, 4], unbiased=False)
            var = input_cpu.var([0, 2, 3, 4],unbiased=False)

            #print("var calculated on CPU:", var)
            # 将 mean 移动回 GPU
            mean = mean.to('cuda:0')
            #print(f"mean shape: {mean.shape}")
            var = var.to('cuda:0')
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]
            #print(input.shape)
        
        return input


class PSP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau_s = 2

    def forward(self, inputs):
        """
        inputs: (N, C, T)
        """
        syns = None
        syn = 0
        n_steps = inputs.shape[-1]
        for t in range(n_steps):
            syn = syn + (inputs[...,t] - syn) / self.tau_s
            if syns is None:
                syns = syn.unsqueeze(-1)
            else:
                syns = torch.cat([syns, syn.unsqueeze(-1)], dim=-1)

        return syns

class MembraneOutputLayer(nn.Module):
    """
    outputs the last time membrane potential of the LIF neuron with V_th=infty
    输出 LIF 神经元在最后时间步的膜电位，阈值电压 V_th=∞
    """
    def __init__(self) -> None:
        super().__init__()
        n_steps = 16 # 时间步数
        # 创建一个从 n_steps-1 到 0 的递减序列
        arr = torch.arange(n_steps-1,-1,-1)
         # 将系数 coef 注册为 buffer，形状为 (1,1,1,1,T)
         # 系数为 0.8 的递减幂次，每个元素计算 0.8 的相应次幂，以模仿膜电位的时间衰减特性
         #[None,None,None,None,:]用于将张量 coef 扩展到与输入张量 x 的形状兼容的维度。None 是 Python 中的语法，用于在特定位置添加新的维度，相当于 PyTorch 的 .unsqueeze(dim) 操作。
        self.register_buffer("coef", torch.pow(0.8, arr)[None,None,None,None,:]) # 最终的 coef 形状是 (1,1,1,1,T)

    def forward(self, x):
        """
        x : (N,C,H,W,T)
        """
        # 计算输出，将输入张量与系数相乘后，在时间维度上求和
        out = torch.sum(x*self.coef, dim=-1)
        return out