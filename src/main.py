# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from common_utils.dataloader_utils import seed_init_fn
from gan_functions import create_dataloader, train_batch, sample_from_generator
from model_parts.discriminator_model import Discriminator
from model_parts.generator_model import Generator
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

argparser = argparse.ArgumentParser()
argparser.add_argument('--outdir', type=str, default='./output', help='output folder')
argparser.add_argument('--datadir', type=str, default='../../datasets/cats', help='dataset folder')
argparser.add_argument('--lrgen', type=float, default=2e-4, help='generator learning rate')
argparser.add_argument('--lrdis', type=float, default=2e-4, help='discriminator learning rate')
argparser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
argparser.add_argument('--batchsize', type=int, default=50, help='train batch size')
argparser.add_argument('--betadis', type=float, default=0.5, help='discriminator adam beta')
argparser.add_argument('--betagen', type=float, default=0.5, help='generator adam beta')

args = argparser.parse_args()

OUTPUT_DIR = args.outdir
DATASET_DIR = args.datadir
DIS_LEARNING_RATE = args.lrdis
GEN_LEARNING_RATE = args.lrgen
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batchsize
DIS_BETA = args.betadis
GEN_BETA = args.betagen

IMAGE_SIZE = [64, 64]
IMAGE_NUM_CHANNELS = 3
LATENT_DIM = 100
NUM_GEN_FEATURE_MAPS = 64
NUM_DISC_FEATURE_MAPS = 64
REAL_LABEL = 1.
FAKE_LABEL = 0.

# == prep folders ==
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True, parents=True)

# == Data ==
cats_dl = create_dataloader(DATASET_DIR, batch_size=BATCH_SIZE)

# == Model ==
gen_model = Generator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_GEN_FEATURE_MAPS, latent_dim=LATENT_DIM)
dis_model = Discriminator(num_channels=IMAGE_NUM_CHANNELS, num_feature_maps=NUM_DISC_FEATURE_MAPS)

gen_model.to(device)
dis_model.to(device)

# == Optimizer and loss ==
criterion = nn.BCELoss().to(device)
dis_optimizer = torch.optim.Adam(dis_model.parameters(), lr=DIS_LEARNING_RATE, betas=(DIS_BETA, 0.999))
gen_optimizer = torch.optim.Adam(dis_model.parameters(), lr=GEN_LEARNING_RATE, betas=(GEN_BETA, 0.999))

# == Train loop ==
epoch_losses = pd.DataFrame(columns=['epoch', 'gen_loss', 'dis_loss'])

for epoch in range(NUM_EPOCHS):
    batch_losses_gen = []
    batch_losses_dis = []

    for i, data in enumerate(cats_dl):
        batch_loss = train_batch(data, gen_model, gen_optimizer, dis_model, dis_optimizer, criterion, device, real_label=REAL_LABEL, fake_label=FAKE_LABEL)
        batch_losses_gen.append(batch_loss['loss_gen'])
        batch_losses_dis.append(batch_loss['loss_dis'])

    epoch_loss = {'epoch': epoch, 'gen_loss': np.average(batch_losses_gen), 'dis_loss': np.average(batch_losses_dis)}
    epoch_losses = epoch_losses.append(epoch_loss, ignore_index=True)
    epoch_losses.to_csv(output_path / Path('train_loss.csv'))
    sample_from_generator(16, gen_model, device, title=f'Epoch {epoch}', path_to_save=output_path / Path(f'epoch_{epoch}'))
    print(epoch_losses)











