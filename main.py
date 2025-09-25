import numpy as np
import multiprocessing
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


def load_checkpoint(path, model, optimizer, ema, scaler, device, use_mixed_precision=False):
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if "ema_model_state_dict" in checkpoint:
        ema.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        print("EMA state dict not found in checkpoint.")

    if use_mixed_precision and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    elif use_mixed_precision:
        print("Scaler state dict not found in checkpoint.")

    start_step = checkpoint["step"]
    print(f"Resumed from step {start_step}")
    return start_step


def train(args):
    # initialize W&B
    if not args.no_wandb:
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

    dataset = DiffumojiDataset(
        image_size=image_size, overfit_test_size=args.overfit_test_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
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

    if args.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None  # type: ignore

    os.makedirs("results", exist_ok=True)

    print("All set, good to go!")

    # training loop
    current_step = 1
    if args.resume_from_checkpoint:
        current_step = load_checkpoint(
            args.resume_from_checkpoint, diffusion_model, optimizer, ema, scaler, device, args.mixed_precision
        )
    # create an infinite iterator from our dataloader
    data_iterator = itertools.cycle(dataloader)

    # use tqdm for a live progress bar
    pbar = tqdm(initial=current_step, total=num_train_steps)

    while current_step <= num_train_steps:
        # get the next batch of data
        images, contexts = next(data_iterator)
        images = images.to(device)
        contexts = contexts.to(device)

        if args.mixed_precision:
            # use autocast for forward pass
            with autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16,
            ):
                # calculate loss
                loss = diffusion_model.p_losses(images, contexts)

            # scale the loss and backward
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)  # type: ignore # upscale the gradients
            scaler.update()  # type: ignore
        else:
            # calculate loss without mixed precision
            loss = diffusion_model.p_losses(images, contexts)
            loss.backward()

        optimizer.zero_grad(set_to_none=True)
        ema.update()  # update ema after each optimizer step

        pbar.set_description(f"loss: {loss.item():.4f}")
        if not args.no_wandb:
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
                grid_tensor = make_grid(
                    sampled_images, nrow=2, padding=2, normalize=False
                )
                save_image(
                    grid_tensor,
                    f"results/sample_{current_step}.png",
                )
                if not args.no_wandb:
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
            }
            if args.mixed_precision:
                checkpoint["scaler_state_dict"] = scaler.state_dict()  # type: ignore
            torch.save(checkpoint, f"results/checkpoint_{current_step}.pt")
        current_step += 1
        pbar.update(1)

    pbar.close()
    print("Training complete!")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Diffusion Model")

    # add arguments
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
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers for the Dataloader",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--no-wandb", action="store_true", help="Disable logging to Weights & Biases"
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision training"
    )

    args = parser.parse_args()
    train(args)
