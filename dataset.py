import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DiffumojiDataset(Dataset):
    def __init__(self, image_size=64):
        self.df = pd.read_csv("data/pairs.csv")
        self.embeddings = torch.load("text_embeddings.pt")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize to [-1, 1]
            ]
        )

        assert len(self.df) == len(self.embeddings), (
            "Dataset and embeddings length mismatch"
        )  # type: ignore

    def __len__(self):
        return len(self.df)  # type: ignore

    def __getitem__(self, idx):
        # get the image and corresponding embedding for the given index
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        embedding = self.embeddings[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, embedding
