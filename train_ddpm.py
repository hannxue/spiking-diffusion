import torch
import copy # 用于深拷贝对象
import logging# 用于日志记录
import os# 用于操作系统相关的操作，如路径处理
                                                                                         
import hydra# 用于配置管理，方便从配置文件中加载参数
import pytorch_lightning as pl # 用于简化 PyTorch 的训练流程，支持多 GPU 训练、自动保存检查点等功能
from omegaconf import OmegaConf# 用于处理 Hydra 的配置文件
from pytorch_lightning.callbacks import ModelCheckpoint# 用于在训练过程中保存模型检查点
from pytorch_lightning import seed_everything# 用于设置随机种子
from torch.utils.data import DataLoader# 用于创建数据加载器

from models.callbacks import EMAWeightUpdate# 用于模型的 EMA（Exponential Moving Average）权重更新
from models.diffusion import DDPM, DDPMv2, DDPMWrapper # 导入扩散模型相关的类
from SDDPM.model.SpikingUNet import  Spk_UNet, SuperResModel
#from models.vae import VAE# 导入变分自编码器（VAE）模型
from models.fsvae_models.fsvae import FSVAE# 引入自定义的 VAE 模型
from util import configure_device, get_dataset # 自定义函数，配置设备和获取数据集
# 配置日志记录器
logger = logging.getLogger(__name__)


def __parse_str(s):
    # 解析字符串为整数列表，例如 "1,2,3" -> [1, 2, 3]
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path=r'D:\hxuser\u430201701\Spiking VaeDiffusion\configs\dataset\mnist', config_name="train", version_base="1.2")# 从配置文件加载配置
def train(config):
    # 1. 获取配置并设置
    # Get config and setup
    config = config.ddpm # 通过配置文件加载数据集相关参数
    logger.info(OmegaConf.to_yaml(config))# 打印当前配置

    # Set seed # 2. 设置随机种子，确保结果可复现
    seed_everything(config.training.seed, workers=True)

    # Dataset# 3. 加载数据集
    root = config.data.root# 数据集根目录
    d_type = config.data.name# 数据集名称（如 MNIST, CIFAR 等）
    image_size = config.data.image_size# 图像尺寸
    dataset = get_dataset(
        d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip
    )# 获取数据集，支持归一化和水平翻转
    N = len(dataset)# 数据集大小
    batch_size = config.training.batch_size # 每批次的样本数
    batch_size = min(N, batch_size)# 确保批次大小不超过数据集的大小

    # Model# 4. 配置模型参数
    lr = config.training.lr # 学习率
    attn_resolutions = __parse_str(config.model.attn_resolutions)# 解析模型的注意力分辨率
    dim_mults = __parse_str(config.model.dim_mults)# 解析模型的维度倍数#dim_mults: "1,2,2,2"
    ddpm_type = config.training.type# 使用的扩散模型类型（如 uncond, form2）

    # Use the superres model for conditional training 
    # # 5. 根据不同的扩散模型类型，选择解码器（UNet(uncond) 或 SuperResModel(form1,form2)）
    decoder_cls = Spk_UNet if ddpm_type == "uncond" else SuperResModel
    decoder = decoder_cls(
       T=config.model.n_timesteps, ch=config.model.dim, ch_mult=dim_mults, attn=config.model.attn,
       num_res_blocks=config.model.n_residual, dropout=config.model.dropout, timestep=config.model.timestep, img_ch=config.data.n_channels)
    

    # EMA parameters are non-trainable
    # 6. 创建 EMA 解码器（EMA 权重更新的模型，非训练模型）# 复制解码器以用于 EMA 更新
    ema_decoder = copy.deepcopy(decoder) #深度拷贝解码器
    for p in ema_decoder.parameters():
        p.requires_grad = False   # EMA 解码器的权重不参与训练
    # 7. 根据扩散模型类型选择相应的 DDPM 模型
    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )# 创建在线训练的 DDPM 模型
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )# 创建目标（EMA 更新）模型
    # 8. 加载预训练的 VAE 模型，并将其设置为评估模式
    # 加载 VAE 模型in_channels, latent_dim, n_steps, k
    fsvae = FSVAE(in_channels=config.data.n_channels, latent_dim=config.model.dim, n_steps=16, k=20)#1,128,16,20
    
    # 加载检查点文件
    checkpoint = torch.load(config.training.vae_chkpt_path, map_location=torch.device('cuda'))
    
    # 加载模型权重
    fsvae.load_state_dict(checkpoint)
    
    # 将模型移动到 GPU
    fsvae.to('cuda')
    #vae = VAE.load_from_checkpoint(
        #config.training.vae_chkpt_path,
        #input_res=image_size,
    #)# 加载 VAE 模型
    fsvae.eval()# 设置为评估模式

    for p in fsvae.parameters():
        p.requires_grad = False# VAE 模型不参与训练
     # 9. 确保在线模型和目标模型都是扩散模型类型
    assert isinstance(online_ddpm, ddpm_cls)
    assert isinstance(target_ddpm, ddpm_cls)
    logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.data.norm}")
     # 10. 创建扩散模型的包装器
    ddpm_wrapper = DDPMWrapper(
        online_ddpm,
        target_ddpm,
        fsvae,
        lr=lr,
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,#type: 'form1'   # DiffuseVAE type. One of ['form1', 'form2', 'uncond']. `uncond` is baseline DDPM
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond,
    )# 创建 DDPMWrapper，封装扩散模型及其训练细节

    # Trainer # 11. 配置训练相关参数
    train_kwargs = {}
    restore_path = config.training.restore_path# 恢复路径（用于从检查点恢复）
    #if restore_path != "":
        # Restore checkpoint# 如果有恢复路径，加载恢复的检查点
       # train_kwargs["resume_from_checkpoint"] = restore_path
    resume_path = restore_path if restore_path != "" else None

    # Setup callbacks# 12. 设置模型保存的回调
    results_dir = config.training.results_dir# 结果保存目录
    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,# 每隔一定的 epoch 保存一次模型
        save_top_k=-1,   # 关键：不做清理，全部保留，当你没有设置 monitor 时，Lightning 的默认 save_top_k=1（只保留 1 个 ckpt）
        save_on_train_epoch_end=True,
    )# 设置检查点回调

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs # 最大训练 epochs5000
    train_kwargs["log_every_n_steps"] = config.training.log_step # 每隔多少步骤记录一次日志5000
    train_kwargs["callbacks"] = [chkpt_callback]
    # 13. 如果使用 EMA，添加 EMA 回调
    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)# 设置 EMA 衰减
        train_kwargs["callbacks"].append(ema_callback)
    # 14. 配置训练设备
    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):  # 如果使用 GPU 训练
        _, devs = configure_device(device)
        #train_kwargs["gpus"] = devs
        train_kwargs["accelerator"] = "gpu"
        train_kwargs["devices"] = 1

        
        # Disable find_unused_parameters when using DDP training for performance reasons
         # 使用 DDP 训练时禁用 unused_parameters，以提高性能
        
        #from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin
        

        #train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True# 保持数据加载器的工作进程持久化
    elif device == "tpu": # 如果使用 TPU 训练
        train_kwargs["tpu_cores"] = 8

    # Half precision training 15. 使用半精度训练（FP16）
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader# 16. 创建数据加载器
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,# 传递额外的参数
    )

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config.training.grad_clip
    # 17. 配置并启动训练
    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, loader, ckpt_path=resume_path)

if __name__ == "__main__":
    train() # 启动训练
