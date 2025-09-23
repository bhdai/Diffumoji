import torch
import pandas as pd
from datasets import load_dataset
from torch.cuda import is_available
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def main():
    model_name = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if is_available() else "cpu")

    # load CLIPProcessor and CLIPModel
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)

    df = pd.read_csv("data/pairs.csv")
    texts = df["caption"].tolist()

    # create dataset and dataloader
    text_dataset = TextDataset(texts)
    dl = DataLoader(
        text_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # this will store embeddings
    all_embeddings = []

    for batch in tqdm(dl, desc="Processing batches"):
        inputs = processor(text=batch, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        all_embeddings.append(embeddings.cpu())  # move embs to cpu to append to list

    final_embedding = torch.cat(all_embeddings, dim=0)
    torch.save(final_embedding, "text_embeddings.pt")
    print(f"Saved {final_embedding.shape[0]} embeddings to text_embeddings.pt")


if __name__ == "__main__":
    main()
