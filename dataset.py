import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class DiffumojiDataset(Dataset):
    def __init__(self, image_size=64):
        self.dataset = load_dataset("arattinger/noto-emoji-captions", split="train")
        self.embeddings = torch.load("text_embeddings.pt")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # normalize to [-1, 1]
            ]
        )

        assert len(self.dataset) == len(self.embeddings), (
            "Dataset and embeddings length mismatch"
        )  # type: ignore

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        # get the iamge and corresponding embedding for the given index
        item = self.dataset[idx]
        image = item["image"].convert("RGB")  # type: ignore
        embedding = self.embeddings[idx]
        if self.transform:
            image = self.transform(image)
        return image, embedding
