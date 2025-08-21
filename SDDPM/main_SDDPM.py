import copy
import json
import os
import warnings
import numpy as np
import wandb
# from data import ImageNet,LSUNBed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from tqdm import trange
import random

from diffusion import GaussianDiffusionTrainer,GaussianDiffusionSampler,LatentGaussianDiffusionTrainer,LatentGaussianDiffusionSampler
from model import Spk_UNet
from score.both import get_inception_and_fid_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
## argument parsing ##
import argparse
parser = argparse.ArgumentParser()# 参数配置，命令行传入的参数，将会在脚本中控制不同的实验设置
parser.add_argument('--seed', default=42, type=int, help='seed')# 用于设置随机种子，确保实验的可重复性。
parser.add_argument('--train', action='store_true', default=False, help='train from scratch')#是否从头开始训练模型。
parser.add_argument('--eval', action='store_true', default=False, help='load ckpt.pt and evaluate FID and IS')#是否加载预训练模型进行评估（FID、IS等指标）。
parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')#设置数据集类型，可以是cifar10，celeba等。
parser.add_argument('--sample_type', type=str, default='ddpm', help='Sample Type')#生成样本的类型（例如DDPM或其他类型的扩散模型）
parser.add_argument('--wandb', action='store_true', default=False, help='use wandb to log training')#是否使用wandb进行训练日志记录和可视化。
# Spiking UNet# 网络模型配置，用于定义UNet模型的层数、通道数等。
parser.add_argument('--ch', default=128, type=int, help='base channel of UNet')
parser.add_argument('--ch_mult', default=[1, 2, 2, 4], help='channel multiplier')
parser.add_argument('--attn', default=[], help='add attention to these levels')
parser.add_argument('--num_res_blocks', default=2, type=int, help='# resblock in each level')
parser.add_argument('--img_size', default=32, type=int, help='image size')
parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate of resblock')
parser.add_argument('--timestep', default=4, type=int, help='snn timestep')
parser.add_argument('--img_ch', type=int, default=1, help='image channel')#这里记得改1或3
# Gaussian Diffusion#扩散模型的配置。
parser.add_argument('--beta_1', default=1e-4, type=float, help='start beta value')
parser.add_argument('--beta_T', default=0.02, type=float, help='end beta value')
parser.add_argument('--T', default=1000, type=int, help='total diffusion steps')
parser.add_argument('--mean_type', default='epsilon', help='predict variable:[xprev, xstart, epsilon]')
parser.add_argument('--var_type', default='fixedlarge', help='variance type:[fixedlarge, fixedsmall]')
# Training
parser.add_argument('--resume', default=False, help="load pre-trained model")#扩散模型的配置。
parser.add_argument('--resume_model', type=str, help='resume model path')
parser.add_argument('--lr', default=2e-4, help='target learning rate')
parser.add_argument('--grad_clip', default=1., help="gradient norm clipping")
parser.add_argument('--total_steps', type=int, default=500000, help='total training steps')# 总训练步数。
parser.add_argument('--warmup', default=5000, help='learning rate warmup')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')#批处理大小。
parser.add_argument('--num_workers', type=int, default=4, help='workers of Dataloader')
parser.add_argument('--ema_decay', default=0.9999, help="ema decay rate")
parser.add_argument('--parallel', default=False, help='multi gpu training')#True是否进行多GPU训练，我改成了False
# Logging & Sampling
parser.add_argument('--logdir', default='./log', help='log directory')
parser.add_argument('--sample_size', type=int,default=64, help="sampling size of images")
parser.add_argument('--sample_step', type=int,default=5000, help='frequency of sampling')
# Evaluation
parser.add_argument('--save_step', type=int,default=0, help='frequency of saving checkpoints, 0 to disable during training')#保存模型的频率。
parser.add_argument('--eval_step', type=int,default=0, help='frequency of evaluating model, 0 to disable during training')#评估模型的频率。
parser.add_argument('--num_images', type=int,default=50000, help='the number of generated images for evaluation')
parser.add_argument('--fid_use_torch', default=True, help='calculate IS and FID on gpu')
parser.add_argument('--fid_cache', default='./stats/cifar10.train.npz', help='FID cache')
parser.add_argument('--num_step', type=int,default=1000, help='number of sampling steps')
parser.add_argument('--pre_trained_path', default='./pth/1224_4T.pt', help='FID cache')

args = parser.parse_args()


device = torch.device('cuda:0')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

#该函数实现了模型的EMA更新策略。EMA是一种平滑技术，通过对模型参数的指数加权平均，减小训练过程中的波动，通常用于生成模型的训练中，使得最终生成的图像质量更稳定。
def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, args.warmup) / args.warmup


def evaluate(sampler, model):
    model.eval()# # 将模型设置为评估模式，禁用 dropout 和 batch normalization 等训练时特有的操作
    with torch.no_grad():## 禁用梯度计算，避免在推理过程中计算和存储梯度，从而节省内存
        images = []# 用于存储生成的图像
        desc = "generating images" # 进度条描述
         # trange 用于创建一个进度条，在每个批次生成图像时更新进度
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)# 计算当前批次的大小，确保不超过总图像数量
            # 创建一个随机的高斯噪声张量 x_T，作为扩散模型的初始输入
            x_T = torch.randn((batch_size, args.img_ch, args.img_size, args.img_size))
            # 使用模型的采样器生成图像。将生成的图像移到 CPU 上以便后续处理
            batch_images = sampler(x_T.to(device)).cpu()
            # 将图像的像素值从 [-1, 1] 范围转换到 [0, 1]，并将其添加到 images 列表中
            images.append((batch_images + 1) / 2)
             # 从生成的图像中选取前 64 张图像，制作成网格并保存为图片文件
            grid = (make_grid(batch_images[:64,...]) + 1) / 2
            save_image(grid, 'ddpm.png')# 保存生成的图像网格为文件
        # 将所有生成的图像合并成一个大的 tensor，并转换为 numpy 数组格式
        images = torch.cat(images, dim=0).numpy()
    # 将模型恢复到训练模式，启用训练特有的操作（如 dropout 等）
    model.train()
    # 计算 Inception Score 和 FID，评估生成图像的质量
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True)# 计算指标并输出进度信息
    return (IS, IS_std), FID, images# 返回 Inception Score、标准差、FID 以及生成的图像


def train():
    if args.dataset == 'cifar10': # 检查命令行参数中的数据集名称是否为 'cifar10'
        dataset = CIFAR10(
            root='/home/dataset/Cifar10', train=True, download=False, # 设置数据集的存储路径
            transform=transforms.Compose([ # 定义数据预处理的操作
                transforms.RandomHorizontalFlip(), # 随机水平翻转图像，增强数据
                transforms.ToTensor(),# 将图片转换为 Tensor 格式
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 将图像像素值归一化，均值和标准差是 0.5（范围从 -1 到 1）
            ]))
        
    elif args.dataset == 'celeba':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.CenterCrop(148),
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.ImageFolder(root='/home/dataset/CelebA/celeba', 
                                                   transform=transform)
        
    elif args.dataset == 'fashion-mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.FashionMNIST(root='/home/dataset/FashionMnist', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
    
    elif args.dataset == 'mnist':
        SetRange = torchvision.transforms.Lambda(lambda X: 2 * X - 1.)
        transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        torchvision.transforms.ToTensor(),
        SetRange])
        dataset = torchvision.datasets.MNIST(root='/home/dataset/Mnist', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
        
   
    else:
        raise NotImplementedError

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True)
    #   infiniteloop 是一个自定义函数 ，   通过 infiniteloop(dataloader)，datalooper 会返回一个无限迭代器，用于在训练时不断获取批次数据，直到训练任务手动终止                             
    datalooper = infiniteloop(dataloader)

    print(f'-------Starting loading {args.dataset} Dataset!-------')
    

    # model setup
    #基于SNN（脉冲神经网络）的UNet模型，该模型被用于生成图像。模型的各个参数（如通道数、残差块数等）由命令行参数传入。
    net_model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)
    optim = torch.optim.Adam(net_model.parameters(), lr=args.lr)
    #args.resume：如果启用了继续训练（--resume），则从指定路径加载已保存的模型检查点
    if args.resume:
        ckpt = torch.load(os.path.join(args.resume_model))
        print(f'Loading Resume model from {args.resume_model}')
        net_model.load_state_dict(ckpt['net_model'], strict=True)
    else:
        print('Training from scratch')#如果没有继续训练，则从头开始训练模型。
        
    #GaussianDiffusionTrainer：初始化扩散训练器，负责训练扩散模型。
    trainer = GaussianDiffusionTrainer(
    net_model, float(args.beta_1), float(args.beta_T), args.T).to(device)
    #GaussianDiffusionSampler：初始化扩散模型的采样器，用于生成图像。
    net_sampler = GaussianDiffusionSampler(
        net_model, float(args.beta_1), float(args.beta_T), args.T, args.img_size,
        args.mean_type, args.var_type).to(device)
    #DataParallel：如果启用了多 GPU 训练（args.parallel），则使用 DataParallel 对训练器和采样器进行封装，允许在多个 GPU 上进行训练。
    if args.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler).cuda()




    # log setup#日志与图像保存：创建一个文件夹用于保存生成的示例图像，生成一个随机噪声输入并保存一个样本图像
    if not os.path.exists(os.path.join(args.logdir,'sample')):
        os.makedirs(os.path.join(args.logdir, 'sample'))
    x_T = torch.randn(int(args.sample_size), int(args.img_ch), int(args.img_size), int(args.img_size))
    x_T = x_T.to(device)
    #next(iter(dataloader)) 返回的是 (images, labels) 这样一个元组，[0] 通过索引访问元组的第一个元素，即 images，提取出前 args.sample_size 张图像
    grid = (make_grid(next(iter(dataloader))[0][:args.sample_size]) + 1) / 2
    save_image(grid, os.path.join(args.logdir,'sample','groundtruth.png'))

    # show model size模型参数统计：计算并输出模型的总参数数量，以便了解模型的大小
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    # start training
    with trange(args.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad() # 清除之前梯度的累积
            x_0 = next(datalooper).to(device)# 从数据加载器中获取下一批图像，并将其移动到指定的设备（如GPU）
            loss = trainer(x_0.float()).mean()# 计算当前批次的损失，并取平均
            loss.backward() # 反向传播计算梯度

            if args.wandb:
                wandb.log({'training loss': loss.item()})# 如果启用了 wandb，记录训练过程中的损失

            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), args.grad_clip)# 对模型的梯度进行裁剪，防止梯度爆炸
            optim.step()# 执行优化器步骤，更新模型参数
            pbar.set_postfix(loss='%.3f' % loss)# 更新进度条，显示当前步数的损失值

            ## reset SNN neuron # 重置脉冲神经网络的神经元状态
            functional.reset_net(net_model)

            # sample，args.sample_step=5000，意思是 每 5000 步生成并保存一次样本图像
            # print(f'Sample at {step} step')
            if args.sample_step > 0 and step % args.sample_step == 0:
                net_model.eval()# 切换模型为评估模式
                with torch.no_grad():
                    x_0 = net_sampler(x_T)# 使用生成器从噪声样本中生成图像
                    grid = (make_grid(x_0) + 1) / 2## 将生成的图像拼接成网格并归一化到 [0, 1]
                    path = os.path.join(# 定义保存路径
                        args.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)# 保存生成的图像
                    ## log to wandb 
                    if args.wandb:
                        wandb.log({'sample': [wandb.Image(grid, caption='sample')]})# 使用 wandb 记录生成的样本图像
                    
                net_model.train()# 切换回训练模式

            # save保存模型检查点，每隔 args.save_step 步保存一次模型检查点。args.save_step = 0，则模型检查点将不会保存。
            # print(f'Save model at {step} step')
            if args.save_step > 0 and step % args.save_step == 0 and step > 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                save_path = str(step) +  'ckpt.pt'
                torch.save(ckpt, os.path.join(args.logdir,save_path))

            # evaluate，args.eval_step = 0 的意思是 禁用评估功能。
            # print(f'Evaluate at {step} step')
            if args.eval_step > 0 and step % args.eval_step == 0 and step > 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                }
                pbar.write(
                    "%d/%d " % (step, args.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
               
                with open(os.path.join(args.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    


def eval():
    # model setup
    model = Spk_UNet(
       T=args.T, ch=args.ch, ch_mult=args.ch_mult, attn=args.attn,
       num_res_blocks=args.num_res_blocks, dropout=args.dropout, timestep=args.timestep, img_ch=args.img_ch)
    
    ckpt_path = args.pre_trained_path #ckpt_path：加载的预训练模型文件路径，通过 args.pre_trained_path 获取。
    ckpt1 = torch.load(ckpt_path)['net_model']#从文件中加载保存的模型参数（权重）。文件通常是 .pt 格式，包含模型的状态字典（state_dict）。
    print(f'Successfully load checkpoint!')


    model.load_state_dict(ckpt1)#：将预训练模型的权重（ckpt1）加载到模型实例中。这会更新模型的参数，使其具有预训练时的状态。
    model.eval()    

    sampler = GaussianDiffusionSampler(# 初始化采样器
        model, float(args.beta_1), float(args.beta_T), args.T, img_size=int(args.img_size),
        mean_type=args.mean_type, var_type=args.var_type,sample_type=args.sample_type,sample_steps=args.num_step).to(device)
    if args.parallel:#如果启用了多GPU训练（args.parallel 为 True），这行代码会将采样器模型包装成一个支持数据并行的模块，使其能够在多个GPU上并行运行。
        sampler = torch.nn.DataParallel(sampler)

    with torch.no_grad():
        images = []#用来保存生成的图像。
        desc = "generating images"
        for i in trange(0, args.num_images, args.batch_size, desc=desc):
            batch_size = min(args.batch_size, args.num_images - i)
            x_T = torch.randn((batch_size, int(args.img_ch), int(args.img_size), int(args.img_size)))#生成与图像尺寸相同的随机噪声 x_T，作为扩散模型的输入。
            batch_images = sampler(x_T.to(device))#通过 sampler 对噪声进行采样，生成一批图像
            batch_images = batch_images.cpu()
            images.append((batch_images + 1) / 2)  #将生成的图像进行归一化，使其像素值位于 [0, 1] 之间        
        images = torch.cat(images, dim=0).numpy()
    print(images.shape)
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, args.fid_cache, num_images=args.num_images,
        use_torch=args.fid_use_torch, verbose=True) 
    print(f'IS: {IS}, IS_std: {IS_std}, FID: {FID}')


def main():
    if args.wandb:
        ## wandb init ##
        wandb.init(project="spike_diffusion", name=str(args.dataset)+str(args.sample_type))
        # suppress annoying inception_v3 initialization warning #
        warnings.simplefilter(action='ignore', category=FutureWarning)

    seed_everything(42)
    if args.train:
        train()
    if args.eval:
        eval()
    if not args.train and not args.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    # app.run(main)
    main()
