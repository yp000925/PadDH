import torch,torchvision
import numpy as np
import logging
def generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance, pad_size=None):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    if pad_size:
        nx = pad_size[0]
        ny = pad_size[1]
    r1 = torch.linspace(-nx / 2, nx / 2 - 1, nx)
    c1 = torch.linspace(-ny / 2, ny / 2 - 1, ny)
    deltaFx = 1 / (nx * deltax) * r1
    deltaFy = 1 / (nx * deltay) * c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx, deltaFy)
    k = 2 * torch.pi / wavelength
    otf = np.exp(1j * k * distance * torch.sqrt(1 - wavelength ** 2 * (mesh_qx ** 2
                                                                       + mesh_qy ** 2)))
    otf = torch.fft.ifftshift(otf)
    return otf



def rgb_to_gray_tensor(tensor_in):
    tensor_out = torchvision.transforms.functional.rgb_to_grayscale(tensor_in)
    return tensor_out




def prepross_bg(img, bg):
    temp = img / bg
    out = (temp - np.min(temp)) / (1 - np.min(temp))
    return out

def get_logger():
    logger = logging.getLogger(name='PadDH')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(name)s] >> %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger



def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    if x.shape[1] == 1:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(x)
    elif x.shape[1] == 3:
        x = x.detach().cpu().squeeze().numpy()
        return normalize_np(np.transpose(x, (1, 2, 0)))
    else:
        raise ValueError("Wrong number of channels")
def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling
def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)
def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

