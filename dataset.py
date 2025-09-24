import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import random


class DiffumojiDataset(Dataset):
    def __init__(self, image_size=64, overfit_test_size=None):
        df = pd.read_csv("data/pairs.csv")
        all_embeddings = torch.load("text_embeddings.pt")

        # group by image path to find unique images and their caption indices
        self.imgpath2idx = (
            df.groupby("image_path").apply(lambda x: x.index.tolist()).to_dict()
        )
        self.unique_imgpath = list(self.imgpath2idx.keys())
        if overfit_test_size is not None:
            print(f"RUNNING OVERFIT TEST WITH {overfit_test_size} SAMPLES")
            self.unique_imgpath = self.unique_imgpath[:overfit_test_size]
            self.imgpath2idx = {
                path: self.imgpath2idx[path] for path in self.unique_imgpath
            }
        self.embeddings = all_embeddings

        base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize to [-1, 1]
            ]
        )

        self.aug_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
            ]
        )

        assert len(df) == len(all_embeddings), "Dataset and embeddings length mismatch"  # type: ignore

        self.images = []
        for img_path in tqdm(
            self.unique_imgpath, desc="Loading and processing unique images"
        ):
            img = Image.open(img_path).convert("RGB")
            self.images.append(base_transform(img))

    def __len__(self):
        return len(self.unique_imgpath)  # type: ignore

    def __getitem__(self, idx):
        # get the image and corresponding embedding for the given index
        image = self.images[idx]
        image_path = self.unique_imgpath[idx]
        caption_indices = self.imgpath2idx[image_path]
        rand_idx = torch.randint(len(caption_indices), (1,)).item()
        choosen_caption_index = caption_indices[rand_idx]
        embedding = self.embeddings[choosen_caption_index]

        if self.aug_transform:
            image = self.aug_transform(image)
        return image, embedding
