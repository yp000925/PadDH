from abc import ABC, abstractmethod
import torch

from .utils import generate_otf_torch


class DHOperator(ABC):
    def __init__(self, device, prop_kernel, **kwargs):
        self.device = device
        self.A = generate_otf_torch(**prop_kernel).to(device)
        self.AT = torch.conj(self.A).to(device)
    def forward(self, data, **kwargs):
        fs_out = torch.multiply(torch.fft.fft2(data.to(self.device)), self.A.expand(data.shape))
        f_out = torch.fft.ifft2(fs_out)
        amplitude = f_out.abs()
        # amplitude = amplitude/torch.max(amplitude)
        return amplitude
    def backward(self, data, **kwargs):
        fs_out = torch.multiply(torch.fft.fft2(data.to(self.device)), self.AT.expand(data.shape))
        f_out = torch.fft.ifft2(fs_out)
        amplitude = f_out.abs()
        # amplitude = amplitude/torch.max(amplitude)
        return amplitude

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass

__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data


@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0

        # return data.clamp(low_clip, 1.0)