import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import matplotlib.pyplot as plt


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def extract2(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


# class GaussianDiffusionTrainer(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T):
#         super().__init__()

#         self.model = model
#         self.T = T

#         self.register_buffer(
#             'betas', torch.linspace(beta_1, beta_T, T).double())
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

#     def forward(self, x_0, y=None):
#         """
#         Algorithm 1.
#         """
#         t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
#         noise = torch.randn_like(x_0)
#         x_t = (
#             extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
#             extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
#         #print(self.model(x_t, t).size())
#         #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
#         print(y)
#         if y != None:
#             loss = F.mse_loss(self.model(x_t, t, torch.argmax(y)), noise, reduction='none')
#         else:
#             loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
#         return loss
#训练高斯扩散模型（Gaussian Diffusion Model）的类 GaussianDiffusionTrainer       
class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model# 将模型作为参数传入并保存
        self.T = T
        # 创建一个线性空间，从 beta_1 到 beta_T，包含 T 个点，表示扩散过程中每一步的 beta 值
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas# 计算 alpha 值，等于 1 - beta
        alphas_bar = torch.cumprod(alphas, dim=0)#计算累积乘积 alphas_bar，用于后续计算
        # 为扩散过程中的 q(x_t | x_{t-1}) 计算预处理常数
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))# sqrt(alphas_bar) 用于计算每一步的标准差
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))# sqrt(1 - alphas_bar) 用于计算噪声项

    def forward(self, x_0):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)# 随机选择扩散步骤 t
        noise = torch.randn_like(x_0)# # 生成与输入图像大小相同的随机噪声
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +# 根据 alpha_bar 计算 x_t
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)# 加入噪声，计算 x_t
        #print(self.model(x_t, t).size())
        #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss


class LatentGaussianDiffusionTrainer(nn.Module):
    def __init__(self, model,vae, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.vae = vae
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self,x_0):
        """
        Algorithm 1. Latent Diffusion
        """
        weight_dtype = torch.float32
        latents = self.vae.encode(
                x_0.to(dtype=weight_dtype)
            ).latent_dist.sample()
        latents = latents * 0.18215
        t = torch.randint(self.T, size=(latents.shape[0], ), device=latents.device)
        noise = torch.randn_like(latents)

        x_t = (
            extract(self.sqrt_alphas_bar, t, latents.shape) * latents +
            extract(self.sqrt_one_minus_alphas_bar, t, latents.shape) * noise)
        #print(self.model(x_t, t).size())
        #loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none')
        return loss
    

class GaussianDiffusionLogger(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    
    def forward(self, x_0):
        '''
        Evaluate the loss through time
        '''

        print(f'{x_0.shape[0]} images start computing loss through time')
        t_list = torch.linspace(start=0, end=self.T-1, steps=self.T, dtype=torch.int64, device=x_0.device)
        # t_list = t_list.view(len(t_list))
        loss_list = []
        with torch.no_grad():
            for t in t_list:
                t = t.unsqueeze(0).repeat(x_0.shape[0])
                noise = torch.randn_like(x_0)
                x_t = (
                    extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
                    extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
                loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
                loss_list.append(loss.item())

                functional.reset_net(self.model)
        
        return t_list.cpu().numpy(), loss_list
        # fig = plt.figure()
        # plt.plot(t_list.cpu().numpy(),loss_list)
        # plt.title('Loss through Time')
        # plt.xlabel('t')
        # plt.ylabel('loss')

        # return fig


#高斯扩散采样的类 GaussianDiffusionSampler
class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000):
        print(mean_type)
        assert mean_type in ['xprev','xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        assert sample_type in ['ddpm', 'ddim','ddpm2']
        super().__init__()

        self.model = model # 模型，用于生成图像的神经网络
        self.T = T # 总的扩散步骤数
        self.img_size = img_size# 图像的尺寸
        self.mean_type = mean_type# 模型预测的类型：`xprev`（预测前一个图像）、`xstart`（预测初始图像）、`epsilon默认`（预测噪声）
        self.var_type = var_type # 变量类型：`fixedlarge`（固定的大方差）默认、`fixedsmall`（固定的小方差）
        self.sample_steps = sample_steps # 采样的步数
        self.sample_type = sample_type# 采样类型：`ddpm`（经典的扩散模型）、`ddim`（变种扩散模型）、`ddpm2`（第二种扩散模型）

        #假设 self.T = 1000（总的扩散步数），self.sample_steps = 100（采样的步数），那么 self.ratio_raw = 1000 / 100 = 10。这意味着每 10 步就会选择一个新的时间步进行采样。
        self.ratio_raw = self.T/self.sample_steps # 扩散步长与采样步数的比例
        #self.t_list 会是一个包含 100 个时间步的列表，表示我们在 1000 个时间步中，选出哪些时间步用于采样。999-989-979...0
        #在实际操作中，生成图像的过程通常会通过 较少的采样步 来加速推理。通常，每隔一定步数采样一次，可以减少不必要的计算，同时保持生成过程的稳定性。
        self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]# 计算反向扩散过程中的采样时间步
        logging.info(self.t_list)
        if self.t_list[-1] != 0:
            self.t_list.append(0) # 确保最后一个时间步为0
        # print(self.t_list)

        # beta_t
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # alpha_t
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)#alphas_bar 是 alphas 的累积乘积（cumulative product）
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]#alphas_bar_prev 是 alphas_bar 的偏移版本，具体来说，它是 alphas_bar 的 前一个时间步 的值
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))#√(αt )

        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))#1−αt
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))#√1/1-(α_t )

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))#√1/(α_t )
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))#√1/(α_t )-1

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',# 计算后验方差
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar)) 
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',# 对数后验方差,取对数来计算。对数变换可以提高计算的稳定性，特别是对于方差非常小的情况。
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1', # 计算后验均值系数1,对应x0
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2', # 计算后验均值系数2,对应xt
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) # 计算后验均值
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape) # 获取后验方差的对数,对数更稳定
        return posterior_mean, posterior_log_var_clipped
    #从噪声预测初始图像
    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        ) # 使用噪声预测初始图像 x_0
    #该方法使用当前的图像 x_t 和前一时刻的图像 xprev 来预测初始图像 x_0。
    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    # 生成模型的均值和方差（p(x_t | t)），表示在给定当前图像 x_t 和时间步 t 的条件下，模型生成前一时刻的图像的分布。
    # model_log_var表示生成模型的对数方差。在 ddpm 采样方法中，方差的选择可能会有两种：固定大方差或固定小方差。通过选择不同的方差策略，模型可以调整生成图像的噪声强度。
    #model_mean：根据 mean_type 的不同，模型会生成不同的预测：epsilon默认`（预测噪声）,当 mean_type == 'epsilon' 时，模型的目标是预测图像中的噪声 epsilon，然后从噪声恢复出图像。
    def p_mean_variance(self, x_t, t):
        # below: only log_variance is used in the KL computations
        # Mean parameterization
        if self.sample_type=='ddpm':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)
            if self.mean_type == 'xprev':       # the model predicts x_{t-1}
                x_prev = self.model(x_t, t)
                x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
                model_mean = x_prev
            elif self.mean_type == 'xstart':    # the model predicts x_0
                x_0 = self.model(x_t, t)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            elif self.mean_type == 'epsilon':   # the model predicts epsilon
                eps = self.model(x_t, t)#：模型根据当前图像 x_t 和时间步 t 预测噪声 epsilon。
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)#使用 epsilon 和当前图像 x_t 来恢复原始图像 x_0。
                #print(x_0.dtype)
                x_0 = x_0.clamp(-1.,1.)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                raise NotImplementedError(self.mean_type)
            #(model_mean)
            x_0 = torch.clip(x_0, -1., 1.)
            
            functional.reset_net(self.model)

            return model_mean, model_log_var
        elif self.sample_type=='ddim':
            eps = self.model(x_t, t)
            a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)# 获取当前时间步的系数 a_t
            sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))  # 当前时间步的噪声强度 sigma_t=√1-(α_t )
            sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))# 前一时间步的噪声强度 sigma_s=√1-(α_tratio )
            a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)# 前一时间步的系数 a_s

            a_ts = a_t/a_s#比率，调整噪声与信息比例
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2 # 计算新的 beta_ts
             # x0_t 是通过去噪模型生成的初始图像估计
            x0_t = (x_t - eps*sigma_t)/(a_t)
            x0_t = x0_t.clamp(-1.,1.)# 将结果裁剪到 [-1, 1] 范围内
            eta = 0 # 可选噪声控制参数
            c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
            c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
            mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t) # 合成最终的去噪结果
            functional.reset_net(self.model)
            return mean
        elif self.sample_type=='ddpm2':
            model_log_var = {
                # for fixedlarge, we set the initial (log-)variance like so to
                # get a better decoder log likelihood
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)

            eps = self.model(x_t, t)

            a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
            a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
            sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
            sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
            a_ts = a_t/a_s
            beta_ts = sigma_t**2-a_ts**2*sigma_s**2
            mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
            mean_x0 = mean_x0.clamp(-1.,1.)
            mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0

            functional.reset_net(self.model)
            return mean_xs, model_log_var
        else:
            pass

    def forward(self, x_T):
        x_t = x_T
        #for time_step in reversed(range(self.T)):
        for n_count1,time_step in enumerate(self.t_list):#这里我们迭代 self.t_list 中的每个时间步（即扩散过程的时间步）。
            if n_count1 < len(self.t_list)-1:
                self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])#self.ratio 记录了时间步间隔的大小，这对采样过程很重要。

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step#这里的 t 是当前时间步的张量，它与 x_t 的 batch size 相同，x_t 是输入图像，它的形状通常是 (batch_size, channels, height, width)，代表一个批次的图像。
            if self.sample_type =='ddpm' or self.sample_type =='ddpm2':
                #print(x_t.dtype)
                # no noise when t == 0
                if time_step > 0:#我们只会在 time_step > 0 时才进行噪声的添加和去噪，当 time_step == 0 时，图像已经恢复到原始图像（即没有噪声了）。
                    #self.p_mean_variance(x_t=x_t, t=t)：通过当前图像 x_t 和时间步 t 计算生成模型的 均值（mean）和 对数方差（log_var）。这些值描述了图像在当前时间步的去噪程度。
                    mean, log_var = self.p_mean_variance(x_t=x_t, t=t)# 计算均值和方差mean是xt-1
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.exp(0.5 * log_var) * noise#加入噪声并更新图像
                else:#当 time_step == 0 时，我们已经恢复到原始图像 x_0
                    eps = self.model(x_t, t)#通过模型 self.model 预测当前图像的噪声 eps。
                    a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)#计算与当前时间步相关的系数，控制噪声和信息的比例。
                    sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                    beta_ts = (1-a_ts**2)
                    x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)#通过 x_t 和噪声 eps 恢复原始图像 x_0。
                    return torch.clip(x_0, -1, 1)
            else:
                if time_step == 0: return x_t
                x_t = self.p_mean_variance(x_t=x_t, t=t)


# class GaussianDiffusionSampler(nn.Module):
#     def __init__(self, model, beta_1, beta_T, T, img_size=32,
#                  mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000,cond=False):
#         print(mean_type)
#         assert mean_type in ['xprev','xstart', 'epsilon']
#         assert var_type in ['fixedlarge', 'fixedsmall']
#         assert sample_type in ['ddpm', 'ddim','ddpm2','analyticdpm']
#         super().__init__()
#         self.ms_pred = torch.load('./score/cifar10_ema_eps_400000.ms_eps.pth')
#         self.model = model
#         self.T = T
#         self.img_size = img_size
#         self.mean_type = mean_type
#         self.var_type = var_type
#         self.sample_steps = sample_steps
#         self.sample_type = sample_type
#         self.cond = cond

#         self.ratio_raw = self.T/self.sample_steps
#         self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]
#         logging.info(self.t_list)
#         if self.t_list[-1] != 0:
#             self.t_list.append(0)
#         print(self.t_list)

#         # beta_t
#         self.register_buffer(
#             'betas', torch.linspace(beta_1, beta_T, T).double())
#         # alpha_t
#         alphas = 1. - self.betas
#         alphas_bar = torch.cumprod(alphas, dim=0)
#         alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
#         self.register_buffer(
#             'sqrt_alphas_bar', torch.sqrt(alphas_bar))
#         self.register_buffer(
#             'one_minus_alphas_bar', (1.- alphas_bar))
#         self.register_buffer(
#             'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

#         # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.register_buffer(
#             'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
#         self.register_buffer(
#             'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

#         # calculations for posterior q(x_{t-1} | x_t, x_0)
#         self.register_buffer(
#             'posterior_var',
#             self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
#         # below: log calculation clipped because the posterior variance is 0 at
#         # the beginning of the diffusion chain
#         self.register_buffer(
#             'posterior_log_var_clipped',
#             torch.log(
#                 torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
#         self.register_buffer(
#             'posterior_mean_coef1',
#             torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
#         self.register_buffer(
#             'posterior_mean_coef2',
#             torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

#     def q_mean_variance(self, x_0, x_t, t):
#         """
#         Compute the mean and variance of the diffusion posterior
#         q(x_{t-1} | x_t, x_0)
#         """
#         assert x_0.shape == x_t.shape
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_log_var_clipped = extract(
#             self.posterior_log_var_clipped, t, x_t.shape)
#         return posterior_mean, posterior_log_var_clipped

#     def predict_xstart_from_eps(self, x_t, t, eps):
#         assert x_t.shape == eps.shape
#         #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
#         return (
#             extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
#             extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
#         )

#     def predict_xstart_from_xprev(self, x_t, t, xprev):
#         assert x_t.shape == xprev.shape
#         return (  # (xprev - coef2*x_t) / coef1
#             extract(
#                 1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
#             extract(
#                 self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
#                 x_t.shape) * x_t
#         )

#     def p_mean_variance(self, x_t, t):
#         # below: only log_variance is used in the KL computations
#         # Mean parameterization
#         if self.sample_type=='ddpm':
#             model_log_var = {
#                 # for fixedlarge, we set the initial (log-)variance like so to
#                 # get a better decoder log likelihood
#                 'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
#                                                 self.betas[1:]])),
#                 'fixedsmall': self.posterior_log_var_clipped,
#             }[self.var_type]
#             model_log_var = extract(model_log_var, t, x_t.shape)
#             if self.mean_type == 'xprev':       # the model predicts x_{t-1}
#                 x_prev = self.model(x_t, t, self.label)
#                 x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
#                 model_mean = x_prev
#             elif self.mean_type == 'xstart':    # the model predicts x_0
#                 x_0 = self.model(x_t, t, self.label)
#                 model_mean, _ = self.q_mean_variance(x_0, x_t, t)
#             elif self.mean_type == 'epsilon':   # the model predicts epsilon
#                 #eps = self.model(x_t, t, self.label)
#                 eps = self.model(x_t, t)
#                 x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
#                 #print(x_0.dtype)
#                 x_0 = x_0.clamp(-1.,1.)
#                 model_mean, _ = self.q_mean_variance(x_0, x_t, t)
#             else:
#                 raise NotImplementedError(self.mean_type)
#             #(model_mean)
#             x_0 = torch.clip(x_0, -1., 1.)
            
#             functional.reset_net(self.model)

#             return model_mean, model_log_var
#         elif self.sample_type=='ddim':
#             eps = self.model(x_t, t, self.label)
#             a_t  = extract(self.sqrt_alphas_bar, t, x_t.shape)
#             sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_s  = extract(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)

#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2

#             x0_t = (x_t - eps*sigma_t)/(a_t)
#             x0_t = x0_t.clamp(-1.,1.)
#             eta = 0
#             c_1 = eta * torch.sqrt((1-a_t.pow(2)/a_s.pow(2)) * (1-a_s.pow(2))/(1-a_t.pow(2)))
#             c_2 = torch.sqrt((1-a_s.pow(2))-c_1.pow(2))
#             mean = a_s * x0_t + c_2*eps + c_1 * torch.randn_like(x_t)
#             functional.reset_net(self.model)
#             return mean
#         elif self.sample_type=='ddpm2':
#             model_log_var = {
#                 # for fixedlarge, we set the initial (log-)variance like so to
#                 # get a better decoder log likelihood
#                 'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
#                                                 self.betas[1:]])),
#                 'fixedsmall': self.posterior_log_var_clipped,
#             }[self.var_type]
#             model_log_var = extract(model_log_var, t, x_t.shape)

#             eps = self.model(x_t, t, self.label)

#             a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
#             a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
#             sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2
#             mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
#             mean_x0 = mean_x0.clamp(-1.,1.)
#             mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0
#             return mean_xs, model_log_var
#         elif self.sample_type=='analyticdpm':
#             eps = self.model(x_t.float(), t, self.label)

#             a_t  = extract2(self.sqrt_alphas_bar, t, x_t.shape)
#             a_s  = extract2(self.sqrt_alphas_bar, t-self.ratio, x_t.shape)
#             sigma_t = torch.sqrt(extract2(self.one_minus_alphas_bar, t, x_t.shape))
#             sigma_s = torch.sqrt(extract2(self.one_minus_alphas_bar, t-self.ratio, x_t.shape))
#             a_ts = a_t/a_s
#             beta_ts = sigma_t**2-a_ts**2*sigma_s**2
#             mean_x0 = ((1/a_t).float()*x_t - (sigma_t/a_t).float() * eps)
#             mean_x0 = mean_x0.clamp(-1.,1.)
#             mean_xs = (a_ts*sigma_s.pow(2)/(sigma_t.pow(2))).float() * x_t + (a_s*beta_ts/(sigma_t.pow(2))).float() * mean_x0

#             sigma2_small = (sigma_s**2*beta_ts)/(sigma_t**2)
#             ms_pred_temp = ((torch.tensor(self.ms_pred[1+int(t[0].cpu())])).float()).to(x_t.device)

#             cov_x0_pred = sigma_t.pow(2)/a_t.pow(2) * (1-ms_pred_temp)
#             cov_x0_pred = cov_x0_pred.clamp(0., 1.)
#             offset = a_s.pow(2)*beta_ts.pow(2)/sigma_t.pow(4) * cov_x0_pred
#             model_var  = sigma2_small + offset
#             model_var  = model_var.clamp(0., 1.)
#             functional.reset_net(self.model)
#             return mean_xs,torch.log(model_var)
#         else:
#             pass

#     def forward(self, x_T, label=None):
#         self.label = label
#         x_t = x_T
#         #for time_step in reversed(range(self.T)):
#         for n_count1,time_step in enumerate(self.t_list):
#             if n_count1 < len(self.t_list)-1:
#                 self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])

#             t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
#             if self.sample_type =='ddpm' or self.sample_type =='ddpm2' or self.sample_type =='analyticdpm':
#                 #print(x_t.dtype)
#                 # no noise when t == 0
#                 if time_step > 0:
#                     mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
#                     noise = torch.randn_like(x_t)
#                     if time_step-self.ratio <= 0:
#                         var_threshold = (2 * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
#                         var = torch.exp(log_var)
#                         var = var.clamp(0., var_threshold)
#                         x_t = mean + var**0.5 * noise
#                         continue
#                     x_t = mean + torch.exp(0.5 * log_var) * noise
#                 else:
#                     #eps = self.model(x_t.float(), t, self.label)
#                     eps = self.model(x_t.float(), t)
#                     a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
#                     sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
#                     beta_ts = (1-a_ts**2)
#                     x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)
#                     return torch.clip(x_0, -1, 1)
#             else:
#                 if time_step == 0: return x_t
#                 x_t = self.p_mean_variance(x_t=x_t, t=t)



class LatentGaussianDiffusionSampler(nn.Module):
    def __init__(self, model,vae, beta_1, beta_T, T, img_size=32,
                 mean_type='xstart', var_type='fixedlarge',sample_type='ddpm',sample_steps=1000):
        print(mean_type)
        assert mean_type in ['xprev','xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        assert sample_type in ['ddpm', 'ddim','ddpm2','analyticdpm']
        super().__init__()
        self.ms_pred = torch.load('./score/cifar10_ema_eps_400000.ms_eps.pth')
        self.model = model
        self.vae   = vae
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        self.sample_steps = sample_steps
        self.sample_type = sample_type

        self.ratio_raw = self.T/self.sample_steps
        self.t_list = [max(int(self.T-1-self.ratio_raw*x),0) for x in range(self.sample_steps)]
        logging.info(self.t_list)
        if self.t_list[-1] != 0:
            self.t_list.append(0)
        print(self.t_list)

        # beta_t
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        # alpha_t
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'one_minus_alphas_bar', (1.- alphas_bar))
        self.register_buffer(
            'sqrt_recip_one_minus_alphas_bar', 1./torch.sqrt(1.- alphas_bar))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        #print((extract(self.sqrt_recip_alphas_bar, t, x_t.shape)).dtype)
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t):
        if self.sample_type=='ddpm':
            model_log_var = {
                'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                                self.betas[1:]])),
                'fixedsmall': self.posterior_log_var_clipped,
            }[self.var_type]
            model_log_var = extract(model_log_var, t, x_t.shape)
            if self.mean_type == 'xprev':       # the model predicts x_{t-1}
                with torch.no_grad():
                    x_prev = self.model(x_t, t, self.label)
                x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
                model_mean = x_prev
            elif self.mean_type == 'xstart':    # the model predicts x_0
                with torch.no_grad():
                    x_0 = self.model(x_t, t, self.label)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            elif self.mean_type == 'epsilon':   # the model predicts epsilon
                with torch.no_grad():
                    eps = self.model(x_t, t, self.label)
                x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
                #print(x_0.dtype)
                x_0 = x_0.clamp(-1.,1.)
                model_mean, _ = self.q_mean_variance(x_0, x_t, t)
            else:
                raise NotImplementedError(self.mean_type)
            x_0 = torch.clip(x_0, -1., 1.)
            
            functional.reset_net(self.model)
            return model_mean, model_log_var
        else:
            pass

    def forward(self, x_T):
        x_t = x_T
        #for time_step in reversed(range(self.T)):
        for n_count1,time_step in enumerate(self.t_list):
            if n_count1 < len(self.t_list)-1:
                self.ratio = int(self.t_list[n_count1] - self.t_list[n_count1+1])

            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long)  * time_step
            if self.sample_type =='ddpm' or self.sample_type =='ddpm2' or self.sample_type =='analyticdpm':
                #print(x_t.dtype)
                # no noise when t == 0
                if time_step > 0:
                    mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
                    noise = torch.randn_like(x_t)
                    if time_step-self.ratio <= 0:
                        var_threshold = (2 * 2. / 255. * (math.pi / 2.) ** 0.5) ** 2
                        var = torch.exp(log_var)
                        var = var.clamp(0., var_threshold)
                        x_t = mean + var**0.5 * noise
                        continue
                    x_t = mean + torch.exp(0.5 * log_var) * noise
                else:
                    with torch.no_grad():
                        eps = self.model(x_t.float(), t, self.label)
                    a_ts = extract(self.sqrt_alphas_bar, t, x_t.shape)
                    sigma_t = torch.sqrt(extract(self.one_minus_alphas_bar, t, x_t.shape))
                    beta_ts = (1-a_ts**2)
                    x_0 = 1/a_ts*( x_t - eps * beta_ts/sigma_t)

                    weight_dtype = torch.float32
                    latents = 1 / 0.18215 * x_0.detach()
                    self.vae = self.vae.to(dtype=weight_dtype)
                    with torch.no_grad():
                        image = self.vae.decode(latents)['sample']
                    return torch.clip(image, -1, 1)
            else:
                if time_step == 0: return x_t
                x_t = self.p_mean_variance(x_t=x_t, t=t)