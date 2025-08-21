# Helper script to sample from a conditional DDPM model
# Add project directory to sys.path
import os
import sys
# 将当前目录的 "main" 目录添加到 sys.path 中，使得后续可以导入 "main" 目录中的模块
#p = os.path.join(os.path.abspath("."), "main")
#sys.path.insert(1, p)
# 导入所需的库和模块
import copy

import hydra
import pytorch_lightning as pl
import torch
from datasets.latent import LatentDataset # 自定义的数据集模块
from models.callbacks import ImageWriter  # 用于写图像的回调函数
from models.diffusion import DDPM, DDPMv2, DDPMWrapper# DDPM模型相关模块
from SDDPM.model.SpikingUNet import  Spk_UNet, SuperResModel
from models.fsvae_models.fsvae import FSVAE# 引入自定义的 VAE 模型# 引入自定义的 VAE 模型# VAE模型
from pytorch_lightning import seed_everything# 用于设置随机种子
from torch.utils.data import DataLoader# 用于加载数据集的DataLoader
from util import configure_device# 配置设备（如GPU）

from score.both import get_inception_and_fid_score  # 你已有的函数
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm

# 字符串解析函数，将逗号分隔的字符串解析为整数列表
def __parse_str(s):
    split = s.split(",")# 按逗号分割字符串
    return [int(s) for s in split if s != "" and s is not None]# 转为整数并返回
def load_images_from_folder(folder):
    image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")])
    images = []
    for path in tqdm(image_paths, desc="Loading generated images"):
        img = read_image(path).float() / 255.0 # shape: (C, H, W)像素值为 [0, 1]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)  # 灰度 → RGB
        images.append(img)
    images = torch.stack(images)  # (N, 3, H, W)
    return images.numpy()#images.permute(0, 2, 3, 1).numpy()  # → (N, H, W, C)
# 通过Hydra配置管理器加载配置文件
@hydra.main(config_path=r'D:\hxuser\u430201701\Spiking VaeDiffusion\configs\dataset\mnist', config_name="test", version_base="1.2")
def sample_cond(config):
    # 设置随机种子，确保实验的可复现性
    config_ddpm = config.ddpm
    config_vae = config.vae
    seed_everything(config_ddpm.evaluation.seed, workers=True)# 设置种子
     # 从配置中获取批量大小、步骤数、采样数等参数
    batch_size = config_ddpm.evaluation.batch_size
    n_steps = config_ddpm.evaluation.n_steps
    n_samples = config_ddpm.evaluation.n_samples
    image_size = config_ddpm.data.image_size
    ddpm_latent_path = config_ddpm.data.ddpm_latent_path
    ddpm_latents = torch.load(ddpm_latent_path) if ddpm_latent_path != "" else None

    # 加载 VAE 模型
    fsvae = FSVAE(in_channels=config_ddpm.data.n_channels, latent_dim=config_ddpm.model.dim, n_steps=16, k=20)#1,128,16,20
    
    # 加载检查点文件
    checkpoint = torch.load(config_vae.evaluation.chkpt_path, map_location=torch.device('cuda'),weights_only=True)
    
    # 加载模型权重
    fsvae.load_state_dict(checkpoint)
    
    # 将模型移动到 GPU
    fsvae.to('cuda')
    # 加载预训练的VAE（变分自编码器）模型
    #vae = VAE.load_from_checkpoint(
        #config_vae.evaluation.chkpt_path, # 从检查点加载VAE模型
        #input_res=image_size,# 设置输入分辨率
    #)
    fsvae.eval()# 设置为评估模式

    # 加载预训练的超分辨率模型（SuperResModel）
    attn_resolutions = __parse_str(config_ddpm.model.attn_resolutions)# 解析注意力分辨率
    dim_mults = __parse_str(config_ddpm.model.dim_mults)# 解析维度缩放因子

    decoder = SuperResModel(
        T=config_ddpm.model.n_timesteps, 
        ch=config_ddpm.model.dim,
        ch_mult=dim_mults,                      
        attn=config_ddpm.model.attn, 
        num_res_blocks=config_ddpm.model.n_residual,
        dropout=config_ddpm.model.dropout, 
        timestep=config_ddpm.model.timestep,
        img_ch=config_ddpm.data.n_channels, 
    )


    ema_decoder = copy.deepcopy(decoder) # 复制解码器作为EMA模型
    decoder.eval()# 设置解码器为评估模式
    ema_decoder.eval()# 设置EMA解码器为评估模式
    # 根据配置选择使用DDPMv2还是DDPM
    ddpm_cls = DDPMv2 if config_ddpm.evaluation.type == "form2" else DDPM
    # 创建在线DDPM和目标DDPM模型
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config_ddpm.model.beta1,
        beta_2=config_ddpm.model.beta2,
        T=config_ddpm.model.n_timesteps,
        var_type=config_ddpm.evaluation.variance,
    )
    # 创建DDPMWrapper
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config_ddpm.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        fsvae=fsvae,
        conditional=True,
        pred_steps=n_steps,
        eval_mode="sample",
        resample_strategy=config_ddpm.evaluation.resample_strategy,
        skip_strategy=config_ddpm.evaluation.skip_strategy,
        sample_method=config_ddpm.evaluation.sample_method,
        sample_from=config_ddpm.evaluation.sample_from,
        data_norm=config_ddpm.data.norm,
        temp=config_ddpm.evaluation.temp,
        guidance_weight=config_ddpm.evaluation.guidance_weight,
        z_cond=config_ddpm.evaluation.z_cond,
        strict=True,
        ddpm_latents=ddpm_latents,
    )

    # 创建潜在变量数据集
    z_dataset = LatentDataset(
        (n_samples, config_vae.model.z_dim, 1, 1),#50000,128,1,1
        (n_samples, 1, image_size, image_size),#我把3改成了1 ,50000,1,32,32
        share_ddpm_latent=True if ddpm_latent_path != "" else False,
        expde_model_path=config_vae.evaluation.expde_model_path,
        seed=config_ddpm.evaluation.seed,
    )

    # 配置设备
    test_kwargs = {}
    loader_kws = {}
    device = config_ddpm.evaluation.device
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        test_kwargs["devices"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        test_kwargs["tpu_cores"] = 8

    #创建数据加载器
    val_loader = DataLoader(
        z_dataset,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        num_workers=config_ddpm.evaluation.workers,
        **loader_kws,
    )

    # 配置图像写回调函数
    write_callback = ImageWriter(
        config_ddpm.evaluation.save_path,
        "batch",
        n_steps=n_steps,
        eval_mode="sample",
        conditional=True,
        sample_prefix=config_ddpm.evaluation.sample_prefix,
        save_vae=config_ddpm.evaluation.save_vae,
        save_mode=config_ddpm.evaluation.save_mode,
        is_norm=config_ddpm.data.norm,
    )
    # 配置训练器
    test_kwargs["callbacks"] = [write_callback]
    test_kwargs["default_root_dir"] = config_ddpm.evaluation.save_path
    trainer = pl.Trainer(**test_kwargs)
    trainer.predict(ddpm_wrapper, val_loader)
    
    # === 采样完成后，计算 IS 和 FID ===
    print("✅ 开始计算 Inception Score 和 FID...")

    gen_image_dir = os.path.join(config_ddpm.evaluation.save_path, "1000", "images")
    gen_images = load_images_from_folder(gen_image_dir)

    is_score, fid_score = get_inception_and_fid_score(
        images=gen_images,
        fid_cache=config_ddpm.evaluation.fid_cache,  # 你在 config 里设定的 .npz 文件
        num_images=config_ddpm.evaluation.n_samples,
        use_torch=True,
        verbose=True
    )

    print(f"✅ Inception Score: {is_score[0]:.2f} ± {is_score[1]:.2f}")
    print(f"✅ FID Score: {fid_score:.2f}")

# 运行采样函数
if __name__ == "__main__":
    sample_cond()

