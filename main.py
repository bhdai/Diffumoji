import numpy as np
import wandb
import argparse
import itertools
import os
from tqdm import tqdm
import torch
from torch.amp import GradScaler, autocast
from ema_pytorch import EMA
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset import DiffumojiDataset
from unet import Unet
from diffusion import GaussianDiffusion


def train(args):
    # initialize W&B
    wandb.init(
        project="Diffumoji",
        config=vars(args),  # automatically logs all command-line arguments
    )

    image_size = args.image_size
    timesteps = args.timesteps
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_train_steps = args.num_train_steps
    sample_every = args.sample_every  # sample and save checkpoint every N steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DiffumojiDataset(image_size=image_size)

    # handling for overfit test
    if args.overfit_test_size is not None:
        print(f"RUNNING OVERFIT TEST WITH {args.overfit_test_size} SAMPLES")
        dataset.dataset = dataset.dataset.select(range(args.overfit_test_size))
        # also need to slice the embeddings
        dataset.embeddings = dataset.embeddings[: args.overfit_test_size]

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    unet_model = Unet(dim=64, condition_dim=512, dim_mults=(1, 2, 4, 8))
    diffusion_model = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=timesteps,
        objective="pred_noise",
        beta_schedule=args.noise_schedule,
    )
    diffusion_model.to(device)

    optimizer = Adam(diffusion_model.parameters(), lr=learning_rate)

    ema = EMA(diffusion_model, beta=0.9999, update_every=10).to(device)

    scaler = GradScaler()

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

        # use autocast for forward pass
        with autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        ):
            # calculate loss
            loss = diffusion_model.p_losses(images, contexts)

        # scale the loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)  # upscale the gradients
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update()  # update ema after each optimizer step

        pbar.set_description(f"loss: {loss.item():.4f}")
        wandb.log({"loss": loss.item()})

        # save samples and checkpoints periodically
        if current_step != 0 and current_step % sample_every == 0:
            print(f"\nStep {current_step}: Saving checkpoint and sampling...")

            diffusion_model.eval()  # switch to evaluation mode
            ema.ema_model.eval()
            with torch.no_grad():
                # use a fixed context for consistent sampling
                fixed_context = contexts[:4]
                sampled_images = ema.ema_model.sample(
                    batch_size=4, context=fixed_context, cfg_scale=args.cfg_scale
                )  # type: ignore
                save_image(
                    sampled_images,
                    f"results/sample_{current_step}.png",
                    nrow=2,
                )
                # create a grid image for wandb
                grid_tensor = make_grid(
                    sampled_images, nrow=2, padding=2, normalize=False
                )
                # grid_tensor is (C, H, W), wandb_expect (H, W, C)
                grid_img = grid_tensor.permute(1, 2, 0).cpu().numpy()
                grid_img = np.clip(
                    grid_img, 0, 1
                )  # make sure image is normalized to [0, 1]
                wandb.log(
                    {
                        "samples": wandb.Image(
                            grid_img, caption=f"Samples at step {current_step}"
                        )
                    }
                )

            diffusion_model.train()  # back to training mode

            # save checkpoint
            checkpoint = {
                "step": current_step,
                "model_state_dict": diffusion_model.model.state_dict(),
                "ema_model_state_dict": ema.ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "scaler_state_dict": scaler.state_dict(),
            }
            torch.save(checkpoint, f"results/checkpoint_{current_step}.pt")
        current_step += 1
        pbar.update(1)

    pbar.close()
    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Diffusion Model")

    # add argumetns
    parser.add_argument("--image_size", type=int, default=64, help="Size of the images")
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--sample_every",
        type=int,
        default=5000,
        help="Frequency of saving samples and checkpoints",
    )
    parser.add_argument(
        "--overfit_test_size",
        type=int,
        default=None,
        help="If set, uses a tiny subset of the data to test for overfitting",
    )

    parser.add_argument(
        "--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale"
    )

    parser.add_argument(
        "--noise_schedule",
        type=str,
        default="cosine",
        help="Beta schedule (linear, cosine, or sigmoid)",
    )

    args = parser.parse_args()
    train(args)
