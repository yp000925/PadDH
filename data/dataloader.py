from glob import glob

import torchvision.transforms
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from pathlib import Path
import os
from torchvision.transforms.transforms import Resize,ToTensor
__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset,
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        # self.transforms =  transforms.Compose([transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='dh')
class DHDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, img_size: int=256):
        super().__init__(root, transforms)
        self.img_size = img_size
        # self.transforms =  transforms.Compose([transforms.ToTensor(),
        #                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        try:
            f = []
            for p in root if isinstance(root, list) else [root]:
                p = Path(root)
                if p.is_dir():
                    f += glob(str(p / '**' / "*.*"), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.fpaths = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png', 'jpg']])
            assert self.fpaths, f'{root}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {root}:{e}\n')

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB').resize([self.img_size, self.img_size])

        if self.transforms is not None:
            img = self.transforms(img)

        return img

@register_dataset(name='dh_1c')
class DH1cDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None, img_size: int=256):
        super().__init__(root, transforms)
        self.img_size = img_size
        self.transforms = ToTensor()

        try:
            f = []
            for p in root if isinstance(root, list) else [root]:
                p = Path(root)
                if p.is_dir():
                    f += glob(str(p / '**' / "*.*"), recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.fpaths = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in ['png', 'jpg']])
            assert self.fpaths, f'{root}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {root}:{e}\n')

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('L').resize([self.img_size, self.img_size])
        img = self.transforms(img)
        # img = self.rescale_data(img, -1, 1)
        return img

    def rescale_data(self,x, min_val, max_val):
        x_mins = x.min()
        x_maxs = x.max()
        rangex = x_maxs - x_mins
        if rangex == 0:
            return x
        else:
            scaled_data = (max_val - min_val) * ((x - x_mins) / rangex) + min_val
            return scaled_data
