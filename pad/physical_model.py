from .utils import generate_otf_torch
import torch


class DHCorrector():
    def __init__(self, operator, measurement, **kwargs):
        self.scale = kwargs.get('scale', 1.0)
        self.operator = operator
        self.measurement = measurement

    def correcting(self, x_t_1, x_t, x_0, grad_map=False, **kwargs):
        difference = self.measurement - self.operator.forward(x_0, **kwargs)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t_1)[0]
        x_t -= norm_grad * self.scale
        # check grad_map
        if grad_map:
            return x_t, norm, norm_grad
        return x_t, norm


class DHOperator():
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


class GaussianNoise():
    def __init__(self, sigma):
        self.sigma = sigma

    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma
