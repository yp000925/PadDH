'''
This script is for PadDH verification with simulation data
Two examples are provided: USAF and cell images from the online source
The default propogation kernel is used for both examples, which can be modified in configs/task_sim_USAF.yaml and configs/task_sim.yaml
'''
from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from pad.diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from pad.condition_methods import GradientCorrection,Identity
from pad.unet import create_model
from pad.physical_model import DHOperator, get_noise
from pad.utils import rgb_to_gray_tensor,get_logger,clear_color,normalize_np


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
    parser.add_argument('--task_config', default='configs/task_sim_USAF.yaml', type=str)
    parser.add_argument('--exp_name', default='simulation', type=str)
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

    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    operator = DHOperator(device=device, **task_config['operator'])

    noiser = get_noise(**task_config['operator']['noise'])
    logger.info(f"Prop kernel: {task_config['operator']['prop_kernel']} / Noise: {task_config['operator']['noise']['name']}"
                f" / Noise level: {task_config['operator']['noise']['sigma']}")

    # Prepare conditioning method
    scale = task_config['conditioning']['scale']
    if task_config['conditioning']['method'] == 'gc':
        logger.info(f"Conditioning method : sampling with gradient correction")
        logger.info(f"Conditioning scale : {scale}")
        cond_method = GradientCorrection(operator=operator, noiser=noiser, scale=scale)
    else:
        logger.info(f"Conditioning method : vanilla sampling")
        cond_method = Identity(operator=operator, noiser=noiser, scale=scale)
    measurement_cond_fn = cond_method.conditioning

    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config)
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)

    # Working directory
    out_path = os.path.join(args.save_dir, EXP_NAME)
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        y = operator.forward(ref_img)
        y_n = noiser(y)

        # Sampling
        measurement = y_n / torch.max(y_n)
        x_start = operator.backward(measurement).requires_grad_()
        plt.imshow(rgb_to_gray_tensor(x_start)[0, 0, :, :].tolist(), cmap='gray')
        plt.show()
        sample = sample_fn(x_start=x_start, measurement=measurement, record=True, save_root=out_path)

        plt.imsave(os.path.join(out_path, 'input', fname), rgb_to_gray_tensor(y_n)[0,0,:,:].tolist(), cmap='gray')
        plt.imsave(os.path.join(out_path, 'label', fname), rgb_to_gray_tensor(ref_img)[0,0,:,:].tolist(),cmap='gray')
        plt.imsave(os.path.join(out_path, 'recon', fname), rgb_to_gray_tensor(sample)[0,0,:,:].tolist(),cmap='gray')
        print(psnr(rgb_to_gray_tensor(sample), rgb_to_gray_tensor(ref_img)))



if __name__ == '__main__':
    main()
