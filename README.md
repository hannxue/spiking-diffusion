# SPIKEDIFFUSION: A FULLY SPIKING CONDITIONAL DIFFUSION FRAMEWORK FOR ENERGY-EFFICIENT IMAGE GENERATION

## Requirements
To install requirements, please make sure to have the following versions of libraries:

- Python 3.10
- PyTorch 2.4.1+cu121
- SpikingJelly 0.0.0.0.14
- Datasets 3.6.0
- Clean-FID 0.1.35
- TensorBoard 2.18.0

## File Structure

- `model/`         : Contains the spiking VAE model
- `SDDPM/`         : Contains the spiking UNet model
- `configs/`       : Contains training and test hyperparameters
- `train_ddpm.py`  : Training code
- `sample_cond.py` : Code for generating images

## Train

To train the model, run the following command:

```bash
python train_ddpm.py

