import argparse
import torch
from torchvision.utils import save_image
from transformers import CLIPModel, CLIPProcessor
from unet import Unet
from diffusion import GaussianDiffusion
import os


def load_model_from_checkpoint(checkpoint_path, device, load_ema=True):
    """load the diffusion model and Unet from a checkpoint"""
    print(f"Loading checkpoint from  {checkpoint_path}")

    unet_model = Unet(dim=64, condition_dim=512, dim_mults=(1, 2, 4, 8))
    diffusion_model = GaussianDiffusion(
        model=unet_model,
        image_size=64,
        timesteps=1000,
        objective="pred_noise",
        beta_schedule="cosine",
    )
    diffusion_model = diffusion_model.to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except RuntimeError as e:
        print(f"Error loading checkpoint: {e}")
        raise

    if load_ema and "ema_model_state_dict" in checkpoint:
        print("Loading EMA model state dict...")
        diffusion_model.load_state_dict(checkpoint["ema_model_state_dict"])
    else:
        print("Loading regular model state dict...")
        diffusion_model.model.load_state_dict(checkpoint["model_state_dict"])

    print("Model loaded successfully.")
    return diffusion_model


class DiffumojiGenerator:
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.diffusion_model = load_model_from_checkpoint(checkpoint_path, device)
        self.diffusion_model.eval()

        print("loading CLIP model...")
        model_name = "openai/clip-vit-base-patch32"
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device)
        print("CLIP model loaded.")

    def _get_text_embedding(self, prompt):
        """Generate CLIP text embedding for the given prompt"""
        print(f"Generating text embedding for prompt: '{prompt}'")
        inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        return embedding

    def generate(self, prompt, batch_size=1, cfg_scale=1.5):
        """Generate emoji images from text prompt"""
        text_embedding = self._get_text_embedding(prompt)

        print("Generating image...")
        with torch.no_grad():
            generated_images = self.diffusion_model.sample(
                batch_size=batch_size,
                context=text_embedding,
                cfg_scale=cfg_scale,
            )
        return generated_images


def main():
    parser = argparse.ArgumentParser(
        description="Generate emoji from text prompt using Diffumoji model"
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for emoji generation"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="generated_emoji.png",
        help="path to save the generated image",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=1.5, help="Classifier-free guidance scale"
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    generator = DiffumojiGenerator(args.checkpoint_path, device)

    generated_images = generator.generate(
        prompt=args.prompt,
        batch_size=args.batch_size,
        cfg_scale=args.cfg_scale,
    )
    # make sure the ouput directory exists
    os.makedirs(
        os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".",
        exist_ok=True,
    )
    save_image(generated_images, args.output_path, nrow=1)
    print(f"Image saved to {args.output_path}")


if __name__ == "__main__":
    main()
