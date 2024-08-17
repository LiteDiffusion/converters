import os
from glob import glob

import click
import torch
from diffusers.pipelines import StableDiffusionPipeline
from safetensors.torch import load_file
from tqdm import tqdm


def convert_safetensors(path: str):
    for filename in tqdm(glob(f"{path}/*.safetensors")):
        ckpt = load_file(filename)
        torch.save(ckpt, filename.replace(".safetensors", ".bin"))
        os.remove(filename)


@click.command()
@click.argument('path', type=click.Path(exists=True))
def main(path: str):
    pipe = StableDiffusionPipeline.from_single_file(path, local_files_only=True)
    # Create the destination directory
    dest_path = path.replace(".safetensors", "")
    os.makedirs(dest_path, exist_ok=True)
    # Export and convert the text encoder
    pipe.text_encoder.save_pretrained(f"{dest_path}/text_encoder")
    convert_safetensors(f"{dest_path}/text_encoder")
    # Export and convert the UNet
    pipe.unet.save_pretrained(f"{dest_path}/unet")
    convert_safetensors(f"{dest_path}/unet")
    # Export and convert the VAE decoder
    pipe.vae.save_pretrained(f"{dest_path}/vae_decoder")
    convert_safetensors(f"{dest_path}/vae_decoder")


if __name__ == '__main__':
    main()
