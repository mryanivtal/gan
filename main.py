# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from common_utils.torch_pil_utils import display_image_from_tensor
from gan_functions import create_dataloader
from model_parts.discriminator_model import Discriminator
from model_parts.generator_model import Generator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

OUTPUT_DIR = './output'
DATASET_DIR = '../datasets/cats'
BATCH_SIZE = 50

output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)

# == Data ==
cats_dl = create_dataloader(DATASET_DIR, batch_size=BATCH_SIZE)

# == Model ==
IMAGE_SIZE = [64, 64]
IMAGE_NUM_CHANNELS = 3
LATENT_DIM = 100
NUM_GEN_FEATURE_MAPS = 64
NUM_DISC_FEATURE_MAPS = 64

gen_net = Generator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_GEN_FEATURE_MAPS, latent_dim=LATENT_DIM)
dis_net = Discriminator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_DISC_FEATURE_MAPS)

gen_net.to(device)
dis_net.to(device)

# == Optimizer and loss ==
REAL_LABEL = 1.
FAKE_LABEL = 0.
DIS_LEARNING_RATE = 1e-2
GEN_LEARNING_RATE = 1e-2

criterion = nn.BCELoss()
optimizer_dis = torch.optim.Adam(dis_net.parameters(), lr=DIS_LEARNING_RATE)
optimizer_gen = torch.optim.Adam(dis_net.parameters(), lr=GEN_LEARNING_RATE)

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
        labels = torch.full([batch_len], REAL_LABEL, dtype=torch.float, device=device)

        labels_pred = dis_net(data).view(-1)
        loss_dis_real = criterion(labels_pred, labels)
        loss_dis_real.backward()

        # b. Train discriminator with fake data
        noise = torch.randn([batch_len, LATENT_DIM, 1, 1], device=device)
        fake_data = gen_net(noise)
        labels = torch.full([batch_len], FAKE_LABEL, dtype=torch.float, device=device)

        labels_pred = dis_net(data).view(-1)
        loss_dis_fake = criterion(labels_pred, labels)
        loss_dis_fake.backward()

        loss_dis = loss_dis_fake + loss_dis_real
        optimizer_dis.step()

        # ==== (2) Update generator network: maximize log(D(G(z)))
        gen_net.zero_grad()
        labels = torch.full([batch_len], REAL_LABEL, dtype=torch.float, device=device)
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











