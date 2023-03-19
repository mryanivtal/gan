# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from common_utils.dataloader_utils import seed_init_fn
from common_utils.torch_pil_utils import display_image_from_tensor
from images_dataset import ImagesDataset
from model_parts.discriminator_model import Discriminator
from model_parts.generator_model import Generator


device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_path = Path('./output')
output_path.mkdir(exist_ok=True, parents=True)

# == Data ==
IMAGE_SIZE = [64, 64]
NUM_CHANNELS = 3
BATCH_SIZE = 50

transformer = lambda x: x/255
cats_ds = ImagesDataset('../datasets/cats', transformer=transformer)
cats_dl = DataLoader(cats_ds, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0, worker_init_fn=seed_init_fn)

# == Model ==
LATENT_DIM = 100
NUM_GEN_FEATURE_MAPS = 64
NUM_DISC_FEATURE_MAPS = 64

gen_net = Generator(num_channels=NUM_CHANNELS, num_feature_maps=NUM_GEN_FEATURE_MAPS, latent_dim=LATENT_DIM)
dis_net = Discriminator(num_channels=NUM_CHANNELS, num_feature_maps=NUM_DISC_FEATURE_MAPS)

gen_net.to(device)
dis_net.to(device)

# == Optimizer and loss ==
criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

dis_learning_rate = 1e-2
gen_learning_rate = 1e-2

optimizer_dis = torch.optim.Adam(dis_net.parameters(), lr=dis_learning_rate)
optimizer_gen = torch.optim.Adam(dis_net.parameters(), lr=gen_learning_rate)


# == Train loop ==
NUM_EPOCHS = 50

img_list = []
epoch_losses = pd.DataFrame(columns=['epoch', 'gen_loss', 'dis_loss'])

for epoch in range(NUM_EPOCHS):
    batch_losses_gen = []
    batch_losses_dis = []

    for i, data in enumerate(cats_dl):
        # ==== (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        # a. Train discriminator with real data
        dis_net.zero_grad()
        data = data.to(device)
        batch_len = len(data)
        labels = torch.full([batch_len], real_label, dtype=torch.float, device=device)

        labels_pred = dis_net(data).view(-1)
        loss_dis_real = criterion(labels_pred, labels)
        loss_dis_real.backward()

        # b. Train discriminator with fake data
        noise = torch.randn([batch_len, LATENT_DIM, 1, 1], device=device)
        fake_data = gen_net(noise)
        labels = torch.full([batch_len], fake_label, dtype=torch.float, device=device)

        labels_pred = dis_net(data).view(-1)
        loss_dis_fake = criterion(labels_pred, labels)
        loss_dis_fake.backward()

        loss_dis = loss_dis_fake + loss_dis_real
        optimizer_dis.step()

        # ==== (2) Update generator network: maximize log(D(G(z)))
        gen_net.zero_grad()
        labels = torch.full([batch_len], real_label, dtype=torch.float, device=device)
        # labels_pred = dis_net(data).view(-1)
        labels_pred = dis_net(data).view(-1)
        loss_gen = criterion(labels_pred, labels)
        loss_gen.backward()
        optimizer_gen.step()

        batch_losses_gen.append(loss_gen.item())
        batch_losses_dis.append(loss_dis.item())
        if i % 100 == 0:
            print('.', end='')

    epoch_loss = {'epoch': epoch, 'gen_loss': np.average(batch_losses_gen), 'dis_loss': np.average(batch_losses_dis)}
    epoch_losses = epoch_losses.append(epoch_loss, ignore_index=True)
    epoch_losses.to_csv(output_path / Path('train_loss.csv'))
    display_image_from_tensor(fake_data, title=f'Epoch {epoch}', save_path=output_path / Path(f'epoch_{epoch}'))

    print(epoch_losses)











