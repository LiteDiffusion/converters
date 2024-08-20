import os

import click
from diffusers.pipelines import StableDiffusionPipeline


@click.command()
@click.argument('path', type=click.Path(exists=True))
def main(path: str):
    pipe = StableDiffusionPipeline.from_single_file(path, local_files_only=True)
    # Create the destination directory
    dest_path = path.replace(".safetensors", "")
    os.makedirs(dest_path, exist_ok=True)
    # Export models
    pipe.text_encoder.save_pretrained(f"{dest_path}/text_encoder", safe_serialization=False)
    pipe.unet.save_pretrained(f"{dest_path}/unet", safe_serialization=False)
    pipe.vae.save_pretrained(f"{dest_path}/vae_decoder", safe_serialization=False)


if __name__ == '__main__':
    main()
