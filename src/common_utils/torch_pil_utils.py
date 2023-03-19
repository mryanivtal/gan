import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def pil_to_tensor(pil_img):
    tensor_img = transforms.PILToTensor()(pil_img)
    img_shape = tensor_img.shape
    tensor_img = tensor_img.flatten()
    tensor_img = tensor_img / 255

    return tensor_img, img_shape


def display_image_from_tensor(tensor_img, title=None, save_path=None, n_columns=8):
    tensor_img = tensor_img.detach()

    if len(tensor_img.shape) == 3:
        tensor_img = transforms.ToPILImage()(tensor_img)
        if title is not None:
            plt.title(title)
        plt.imshow(tensor_img)

    elif len(tensor_img.shape) == 4:
        n_images = tensor_img.shape[0]
        n_rows = int(np.ceil(n_images / n_columns))

        fig = plt.figure(figsize=(4., 4.))
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_columns),axes_pad=0.1)

        for i in range(n_images):
            img = transforms.ToPILImage()(tensor_img[i])
            grid[i].imshow(img)

    else:
        raise Exception('invalid dimension - shoule be 3 for single image or 4 for batch')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
