import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class DiffumojiDataset(Dataset):
    def __init__(self, image_size=64):
        self.df = pd.read_csv("data/pairs.csv")
        self.embeddings = torch.load("text_embeddings.pt")
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
                transforms.ToTensor(),
            ]
        )

        assert len(self.df) == len(self.embeddings), (
            "Dataset and embeddings length mismatch"
        )  # type: ignore

        self.images = []
        for img_path in tqdm(
            self.df["image_path"], desc="Loading and processing images"
        ):
            img = Image.open(img_path).convert("RGB")
            self.images.append(base_transform(img))

    def __len__(self):
        return len(self.df)  # type: ignore

    def __getitem__(self, idx):
        # get the image and corresponding embedding for the given index
        embedding = self.embeddings[idx]
        image = self.images[idx]

        if self.aug_transform:
            image = self.aug_transform(image)
        return image, embedding
