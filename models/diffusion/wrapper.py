import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from util import space_timesteps


class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        fsvae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]                         # 检查损失函数类型，只支持 "l1" 或 "l2"
        assert eval_mode in ["sample", "recons"]            # 检查评估模式，只支持 "sample" 或 "recons"
        assert resample_strategy in ["truncated", "spaced"] # 检查重采样策略，只支持 "truncated" 或 "spaced"
        assert sample_method in ["ddpm", "ddim"]            # 检查采样方法，只支持 "ddpm" 或 "ddim"
        assert skip_strategy in ["uniform", "quad"]          # 检查跳跃策略，只支持 "uniform" 或 "quad"
        # 初始化参数
        self.z_cond = z_cond                     # 是否使用 z 作为条件
        self.online_network = online_network     # 在线网络（当前训练网络）
        self.target_network = target_network     # 目标网络（用于评估）
        self.fsvae = fsvae                           # VAE模型，用于条件生成
        self.cfd_rate = cfd_rate                 # 控制无条件生成的概率（用于分类自由引导）

        # Training arguments 
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()# 损失函数选择，默认为 L2 损失
        self.lr = lr                             #损失函数选择，默认为 L2 损失
        self.grad_clip_val = grad_clip_val       # 梯度裁剪的阈值
        self.n_anneal_steps = n_anneal_steps     # KL退火步数（用于调节学习率）

        # Evaluation arguments
        self.sample_from = sample_from              # 选择采样源（"target" 或 "online"）
        self.conditional = conditional              # 是否使用条件生成
        self.sample_method = sample_method          # 采样方法（"ddpm" 或 "ddim"）
        self.resample_strategy = resample_strategy  # 重采样策略（"spaced" 或 "truncated"）
        self.skip_strategy = skip_strategy          # 跳跃策略（"uniform" 或 "quad"）
        self.eval_mode = eval_mode                  # 评估模式（"sample" 或 "recons"）
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps # # 预测步数
        self.pred_checkpoints = pred_checkpoints    # 预测检查点
        self.temp = temp                            # 温度系数
        self.guidance_weight = guidance_weight      # 引导权重
        self.ddpm_latents = ddpm_latents            # DDPM潜在变量

        # Disable automatic optimization禁用自动优化
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)初始化 Spaced Diffusion（用于基于时间步的重采样）
        self.spaced_diffusion = None

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):  
        # 选择样本网络（在线网络或目标网络）
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        # 根据在线网络类型选择重采样方法
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # 如果使用基于时间步的重采样
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            # 计算采样步数
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            # 如果尚未初始化 SpacedDiffusion，则初始化
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)
            
            # 使用 DDIM 采样方法
            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            # 使用 DDPM 采样方法
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # # 对于截断重采样，只支持 DDPM 采样
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
         # 使用目标或在线网络进行常规 DDPM 采样
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        #  获取优化器和学习率调度器
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:
            x = batch
            # 使用 VAE 编码器生成条件信号
            with torch.no_grad():
                spike_input = x.unsqueeze(-1).repeat(1, 1, 1, 1, 16) # (N,C,H,W,T)时间步维度上重复图像数据 ，spike_input还要进行SNN Encoder
                cond, q_z, p_z, sampled_z = self.fsvae(spike_input, scheduled=True) # sampled_z(B,C,1,1,T)
                #mu, logvar = self.vae.encoder(x * 0.5 + 0.5)
                #z = self.vae.reparameterize(mu, logvar)
                #cond = self.vae.decode(z)
                cond = 2 * cond - 1#带有一些“模糊”信息，是一个较弱的条件

            # 根据分类自由引导率调整条件信号
            if torch.rand(1)[0] < self.cfd_rate:#随机生成一个 [0,1) 之间的浮点数；如果这个数小于 self.cfd_rate（比如 0.1），就执行下面的“条件丢弃”逻辑
                cond = torch.zeros_like(x)#也就是说：这个判断表示“有一定概率不使用条件信息”。
                z = torch.zeros_like(z)
        else:
            x = batch

        # 随机生成时间步
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # # 生成噪声
        eps = torch.randn_like(x)

        # 使用在线网络预测噪声
        eps_pred = self.online_network(
            x, eps, t, low_res=cond)

        # 计算损失
        loss = self.criterion(eps, eps_pred)

        # 梯度裁剪和优化
        optim.zero_grad()
        self.manual_backward(loss)
        # 在 batch 结束时重置 SNN 内部状态
        from spikingjelly.activation_based import functional
        functional.reset_net(self.online_network)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # 学习率调度
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch #x_t（B,1，32，32）z(B,128,1,1)
            z_num=z.shape[0]
            recons,sample_z = self.fsvae.sample(z_num)
            recons = 2 * recons - 1#我觉得严格一点这里应该是sample
            


            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
