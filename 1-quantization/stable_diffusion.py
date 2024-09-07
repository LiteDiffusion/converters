import json
import sys
from argparse import Namespace

import click
import torch
from aimet_quantsim import apply_adaround_te, calibrate_te, calibrate_unet, calibrate_vae, export_all_models
from tqdm.auto import tqdm
from transformers import CLIPTokenizer

from redefined_modules.transformers.models.clip.modeling_clip import CLIPTextModel
from redefined_modules.diffusers.models.unet_2d_condition import UNet2DConditionModel
from redefined_modules.diffusers.models.vae import AutoencoderKLDecoder
from stable_diff_pipeline import replace_mha_with_sha_blocks, run_the_pipeline, save_image
from stable_diff_pipeline import run_tokenizer, run_text_encoder, run_diffusion_steps, run_vae_decoder


sys.path.insert(0, '.')
sys.setrecursionlimit(10000)


@click.command()
@click.argument("pretrained_model_path", type=click.Path(exists=True))
@click.option("--device", "-d", default="cuda", help="Device to run the model on.")
def main(pretrained_model_path: str, device: str):

    print('-----------------------------------')
    print('Loading the pre-trained FP32 models')
    print('-----------------------------------')

    print("Loading pre-trained TextEncoder model")
    text_encoder = CLIPTextModel.from_pretrained(f'{pretrained_model_path}/text_encoder', torch_dtype=torch.float).to(device)
    text_encoder.config.return_dict = False

    print("Loading pre-trained UNET model")
    unet = UNet2DConditionModel.from_pretrained(f'{pretrained_model_path}/unet', torch_dtype=torch.float).to(device)
    unet.config.return_dict = False
    replace_mha_with_sha_blocks(unet)

    print("Loading pre-trained VAE model")
    vae = AutoencoderKLDecoder.from_pretrained(f'{pretrained_model_path}/vae_decoder', torch_dtype=torch.float).to(device)
    vae.config.return_dict = False

    print('-------------------------------')
    print("Run a floating-point evaluation")
    print('-------------------------------')
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    prompt = "decorated modern country house interior, 8 k, light reflections"
    image = run_the_pipeline(prompt, unet, text_encoder, vae, tokenizer, test_name='fp32', seed=1.36477711e+14)
    save_image(image.squeeze(0), 'generated.png')

    print('--------------------------------------------------------')
    print('Apply AIMET Adaptive Rounding (AdaRound) to the TE model')
    print('--------------------------------------------------------')
    with open('config.json', 'rt') as f:
        config = Namespace(**json.load(f))
    with open(config.calibration_prompts, "rt") as f:
        print(f'Loading prompts from {config.calibration_prompts}')
        prompts = f.readlines()
        prompts = prompts[:50]
    tokens = [run_tokenizer(tokenizer, prompt) for prompt in prompts]
    text_encoder_sim = apply_adaround_te(text_encoder, tokens, config)
    del text_encoder
    text_encoder = None

    print('----------------------')
    print('Calibrate the TE model')
    print('----------------------')
    text_encoder_sim = calibrate_te(text_encoder_sim, tokens, config)

    print('------------------------')
    print('Calibrate the UNET model')
    print('------------------------')
    embeddings = [(run_text_encoder(text_encoder_sim.model, uncond),
                run_text_encoder(text_encoder_sim.model, cond)) for cond, uncond in tokens]
    embeddings = [torch.cat([uncond, cond])for uncond, cond in embeddings]
    unet_sim = calibrate_unet(unet, embeddings, config)

    print('-----------------------')
    print('Calibrate the VAE model')
    print('-----------------------')
    latents = [run_diffusion_steps(unet_sim.model, i) for i in tqdm(embeddings)]
    vae_sim = calibrate_vae(vae, latents, config)

    print('--------------------------------------')
    print('Running Quantized off-target Inference')
    print('--------------------------------------')
    image = run_the_pipeline(prompt, unet_sim.model, text_encoder_sim.model, vae_sim.model, tokenizer, test_name="int8", seed=1.36477711e+14)
    save_image(image.squeeze(0), 'generated_after_quant.png')

    print('----------------')
    print('Export the model')
    print('----------------')
    export_all_models(text_encoder_sim, unet_sim, vae_sim, tokens, embeddings, latents)


if __name__ == "__main__":
    main()
