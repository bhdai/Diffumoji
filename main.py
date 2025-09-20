import itertools
import os
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import DiffumojiDataset
from unet import Unet
from diffusion import GaussianDiffusion


def train():
    image_size = 64
    timesteps = 1000
    batch_size = 32
    learning_rate = 1e-4
    num_train_steps = 100000  # example value
    sample_every = 5000  # sample and save checkpoint every N steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DiffumojiDataset(image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    unet_model = Unet(dim=64, condition_dim=512, dim_mults=(1, 2, 4, 8))
    diffusion_model = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=timesteps,
        objective="pred_noise",
    )
    diffusion_model.to(device)

    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)

    os.makedirs("results", exist_ok=True)

    print("All set, good to go!")

    # training loop
    current_step = 0
    # create an infinite iterator from our dataloader
    data_iterator = itertools.cycle(dataloader)

    # use tqdm for a live progress bar
    pbar = tqdm(initial=current_step, total=num_train_steps)

    while current_step < num_train_steps:
        # get the next batch of data
        images, contexts = next(data_iterator)
        images = images.to(device)
        contexts = contexts.to(device)

        # calculate loss
        loss = diffusion_model.p_losses(images, contexts)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"loss: {loss.item():.4f}")

        # save samples and checkpoints periodically
        if current_step != 0 and current_step % sample_every == 0:
            print(f"\nStep {current_step}: Saving checkpoint and sampling...")

            diffusion_model.eval()  # switch to evaluation mode
            with torch.no_grad():
                # use a fixed context for consistent sampling
                fixed_context = contexts[:4]
                sampled_images = diffusion_model.sample(
                    batch_size=4, context=fixed_context
                )
                save_image(
                    sampled_images,
                    f"results/sample_{current_step}.png",
                    nrow=2,
                )

            diffusion_model.train()  # back to training mode

            # save checkpoint
            checkpoint = {
                "step": current_step,
                "model_state_dict": diffusion_model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }
            torch.save(checkpoint, f"results/checkpoint_{current_step}.pt")
        current_step += 1
        pbar.update(1)

    pbar.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
