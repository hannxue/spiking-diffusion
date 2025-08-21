# SPIKEDIFFUSION: A FULLY SPIKING CONDITIONAL DIFFUSION FRAMEWORK FOR ENERGY-EFFICIENT IMAGE GENERATION
# Requirements
To install requirements: python 3.10
                         torch 2.4.1+cu121
                         spikingjelly 0.0.0.0.14
                         datasets 3.6.0
                         clean-fid 0.1.35
                         tensorboard 2.18.0
 # File     
 model/     # spiking vae   
 SDDPM/     # spking UNet
 configs/   #training and test hyperparameters
 train_ddpm.py   # training code
 sample_cond.py   # generate image
  # Train
  python train_ddpm.py
