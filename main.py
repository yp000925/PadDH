'''
This script is used to validate the PadDH with experimental data
Two examples are provided: USAF and convallaria
You may use your own experimental data by defining your own task within the parse_task function
'''
from functools import partial
import os
import argparse
import yaml
from PIL import  Image
import numpy as np
import json
from torch.fft import fft2, ifft2
import torch
import matplotlib.pyplot as plt
from pad.unet import create_model
from pad.physical_model import DHOperator
from data.dataloader import get_dataset, get_dataloader
from pad.utils import rgb_to_gray_tensor,get_logger,prepross_bg,generate_otf_torch
import torchvision.transforms as transforms
from pad.physical_model import DHCorrector
from pad.script_util import create_gaussian_diffusion
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def psnr(x, im_orig):
    def norm_tensor(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0) * torch.log10(1 / mse)
    return psnr



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config', default='configs/diffusion_config.yaml', type=str)
    parser.add_argument('--task_config', default='configs/task_config.yaml',type=str)
    parser.add_argument('--exp_name', default='USAF', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(args=[])

    # logger
    logger = get_logger()

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    EXP_NAME = args.exp_name

    # Working directory
    out_path = os.path.join(args.save_dir, EXP_NAME)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator
    operator = DHOperator(device=device, **task_config['operator'])
    # noiser = get_noise(**task_config['operator']['noise'])
    logger.info(f"Prop kernel: {task_config['operator']['prop_kernel']} ")
    # # Prepare data
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    for i,measurement in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        measurement = measurement.to(device)
        measurement = measurement / torch.max(measurement)

        # Prepare correcting method
        scale = task_config['correction']['scale']
        corrector = DHCorrector(operator=operator, measurement=measurement,scale=scale)

        # Load diffusion sampler
        diffusion_config['timestep_respacing'] = [500]
        diffusion = create_gaussian_diffusion(**diffusion_config)
        sample_fn = diffusion.p_sample_loop

        # Sampling
        x_start = operator.backward(measurement).requires_grad_()
        plt.imshow(x_start[0,0,:,:].tolist(), cmap='gray')
        plt.show()

        sample = sample_fn(
            model,
            x_start.shape,
            x_start = x_start,
            correction_fn=corrector.correcting,
            save_root=out_path,
            verbose = True,
            progress=True
        )

        plt.imsave(os.path.join(out_path, 'input', fname), rgb_to_gray_tensor(measurement)[0, 0, :, :].tolist(), cmap='gray')
        plt.imsave(os.path.join(out_path, 'recon', fname), rgb_to_gray_tensor(sample)[0, 0, :, :].tolist(), cmap='gray')



if __name__ == '__main__':
    main()
