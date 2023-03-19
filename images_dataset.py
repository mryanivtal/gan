import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from pathlib import Path
from common_utils.torch_pil_utils import display_image_from_tensor


class ImagesDataset(Dataset):
    def __init__(self, images_dir: str, transformer=None):
        self.images_path = Path(images_dir)
        self.transformer = transformer
        assert self.images_path.exists()

        self.file_list = list(self.images_path.glob('*'))
        self.length = len(self.file_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = int(item.item())

        image_path = Path(self.file_list[item])
        sample = read_image(str(image_path))
        if self.transformer:
            sample = self.transformer(sample)

        return sample
