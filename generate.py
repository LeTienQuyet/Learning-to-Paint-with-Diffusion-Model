import argparse
import os

from PIL import Image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import torch
import numpy as np

def generate_and_save_images(diffusion, batch_size, sampling_timesteps, output_folder, start_idx, device):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            imgs = diffusion.sample(batch_size=batch_size).to(device)

    for i, img_tensor in enumerate(imgs):
        img_np = (img_tensor.cpu().numpy() * 255.0).astype(np.uint8).transpose(1, 2, 0)
        save_path = os.path.join(output_folder, f"generated_image_{start_idx + i}.png")
        Image.fromarray(img_np).save(save_path)

def main(
        num_samples, ckpt_path, output_folder, dim, dim_mults, sampling_timesteps, batch_size, image_size
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Unet(
        dim=dim,
        dim_mults=dim_mults,
        flash_attn=True
    ).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()

    os.makedirs(output_folder, exist_ok=True)

    num_batchs = num_samples // batch_size
    remainder = num_samples % batch_size

    for batch in range(num_batchs):
        generate_and_save_images(diffusion, batch_size, sampling_timesteps, output_folder, batch*batch_size, device)

    if remainder > 0:
        generate_and_save_images(diffusion, remainder, sampling_timesteps, output_folder, num_batchs*batch_size, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyper-parameters for generate images using Diffusion Model")

    parser.add_argument("--num_samples", type=int, help="Number of images to generate", default=10)
    parser.add_argument("--ckpt_path", type=str, help="Checkpoint of model", default="ckpt/model-1.pt")
    parser.add_argument("--output_folder", type=str, help="Directory save generated images", default="generated_images")
    parser.add_argument("--dim", type=int, help="Dimension size of the model", default=64)
    parser.add_argument("--dim_mults", type=int, nargs='+', help="List of dimension multipliers for the model", default=(1, 2, 4, 8))
    parser.add_argument("--sampling_timesteps", type=int, help="Number of timesteps", default=1000)
    parser.add_argument("--batch_size", type=int, help="Batch size for generation", default=8)
    parser.add_argument("--image_size", type=int, help="Image size", default=256)

    args = parser.parse_args()

    if args.sampling_timesteps > 1000:
        raise ValueError(f"sampling_timesteps must be <= 1000, but get sampling_timesteps = {args.sampling_timesteps}")

    main(
        num_samples=args.num_samples,
        ckpt_path=args.ckpt_path,
        output_folder=args.output_folder,
        dim=args.dim,
        dim_mults=args.dim_mults,
        sampling_timesteps=args.sampling_timesteps,
        batch_size=args.batch_size,
        image_size=args.image_size
    )


