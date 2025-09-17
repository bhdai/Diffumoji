import torch
from datasets import load_dataset
from torch.cuda import is_available
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def main():
    model_name = "openai/clip-vit-base-patch32"
    device = torch.device("cuda" if is_available() else "cpu")

    # load CLIPProcessor and CLIPModel
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)

    # load the dataset
    dataset = load_dataset("arattinger/noto-emoji-captions", split="train")

    text_dataset = dataset.select_columns(["text"])

    # wrap the dataset in a dataloader
    dl = DataLoader(
        text_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )

    # this will store embeddings
    all_embeddings = []

    for batch in tqdm(dl, desc="Processing batches"):
        # get text from batch
        texts = batch['text']
        # process text
        inputs = processor(text=texts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        all_embeddings.append(embeddings.cpu()) # move embs to cpu to append to list

    final_embedding = torch.cat(all_embeddings, dim=0)
    torch.save(final_embedding, "text_embeddings.pt")
    print(f"Saved {final_embedding.shape[0]} embeddings to text_embeddings.pt")

if __name__ == "__main__":
    main()
