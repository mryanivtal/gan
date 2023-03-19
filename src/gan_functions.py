from pathlib import Path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from common_utils.torch_pil_utils import display_image_from_tensor
from images_dataset import ImagesDataset
from common_utils.dataloader_utils import seed_init_fn


def create_dataloader(dataset_dir: str, batch_size=50, num_workers=0) -> DataLoader:
    if not Path(dataset_dir).exists():
        raise FileNotFoundError('input data folder does not exist')

    fn_ds_transforms = Lambda(lambda x: x / 255)
    cats_ds = ImagesDataset(dataset_dir, transforms=fn_ds_transforms)
    cats_dl = DataLoader(cats_ds, batch_size=batch_size,
                         shuffle=True, worker_init_fn=seed_init_fn, num_workers=num_workers)

    return cats_dl


def train_batch(data: torch.Tensor, gen_model, gen_optimizer, dis_model, dis_optimizer, criterion, device,
                real_label=1., fake_label=0.) -> Dict:
    # ==== (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
    # a. Train discriminator with real data
    dis_model.zero_grad()
    data = data.to(device)
    batch_len = len(data)
    labels = torch.full([batch_len], real_label, dtype=torch.float, device=device)
    labels_pred = dis_model(data).view(-1)
    loss_dis_real = criterion(labels_pred, labels)
    loss_dis_real.backward()

    # b. Train discriminator with fake data
    latent_dim = gen_model.latent_dim
    noise = torch.randn([batch_len, latent_dim, 1, 1], device=device)
    fake_data = gen_model(noise)
    labels = torch.full([batch_len], fake_label, dtype=torch.float, device=device)
    labels_pred = dis_model(fake_data.detach()).view(-1)
    loss_dis_fake = criterion(labels_pred, labels)
    loss_dis_fake.backward()
    loss_dis = loss_dis_fake + loss_dis_real
    dis_optimizer.step()

    # ==== (2) Update generator network: maximize log(D(G(z)))
    gen_model.zero_grad()
    labels = torch.full([batch_len], real_label, dtype=torch.float, device=device)
    labels_pred = dis_model(fake_data).view(-1)
    loss_gen = criterion(labels_pred, labels)
    loss_gen.backward()
    gen_optimizer.step()

    batch_loss = {'loss_gen': loss_gen.item(), 'loss_dis': loss_dis.item()}
    return batch_loss


def sample_from_generator(n_samples, gen_model, device, title=None, path_to_save=None, noise=None):
    latent_dim = gen_model.latent_dim
    if noise is not None:
        assert list(noise.shape) == [n_samples, latent_dim, 1, 1]
    else:
        noise = torch.randn([n_samples, latent_dim, 1, 1], device=device)

    sample = gen_model(noise)
    display_image_from_tensor(sample, title=title, save_path=path_to_save)


# custom weights initialization called on netG and netD
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
