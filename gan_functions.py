from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from images_dataset import ImagesDataset
from common_utils.dataloader_utils import seed_init_fn


def create_dataloader(dataset_dir: str, batch_size=50) -> DataLoader:
    if not Path(dataset_dir).exists():
        raise FileNotFoundError('input data folder does not exist')

    fn_ds_transforms = Lambda(lambda x: x / 255)
    cats_ds = ImagesDataset(dataset_dir, transforms=fn_ds_transforms)
    cats_dl = DataLoader(cats_ds, batch_size=batch_size,
                         shuffle=True, num_workers=0, worker_init_fn=seed_init_fn)

    return cats_dl
