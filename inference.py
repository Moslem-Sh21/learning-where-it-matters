import os
from tqdm.auto import tqdm
import numpy as np
import torch
import pywt
import json

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from config import parse_args
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# -------------------------------------------------------------------
# Global concept one-hot storage (same pattern as training)
# -------------------------------------------------------------------
CURRENT_INPUT_CONDITIONS = {"tensor": None}


class BottleneckConceptModule(torch.nn.Module):
    """
    Learnable concept vectors added at the SDXL UNet mid-block (bottleneck).

    - num_concepts: dimensionality of the one-hot concept vector (e.g., 100)
    - mid_channels: number of channels in SDXL mid-block (usually 1280)
    """

    def __init__(self, num_concepts: int, mid_channels: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.mid_channels = mid_channels

        # [num_concepts, mid_channels]
        self.concept_emb = torch.nn.Parameter(torch.zeros(num_concepts, mid_channels))
        torch.nn.init.normal_(self.concept_emb, mean=0.0, std=0.02)

    def forward(self, mid_tensor: torch.Tensor, concept_onehot: torch.Tensor) -> torch.Tensor:
        if concept_onehot is None:
            return mid_tensor

        B, C, H, W = mid_tensor.shape
        concept_onehot = concept_onehot.to(mid_tensor.device).float()

        # Broadcast if needed
        if concept_onehot.shape[0] != B:
            concept_onehot = concept_onehot.expand(B, -1)

        delta = concept_onehot @ self.concept_emb  # [B, C]
        delta = delta.view(B, C, 1, 1)
        return mid_tensor + delta


def generate_and_save_seeds(num_seeds=130, seed_range=(1, 1000000), output_file="generated_seeds.json"):
    """Generate random seeds and save them to a JSON file."""
    import random

    seeds = random.sample(range(seed_range[0], seed_range[1]), num_seeds)

    with open(output_file, "w") as f:
        json.dump({"seeds": seeds}, f)

    return seeds


def load_seeds(input_file="generated_seeds.json"):
    """Load previously generated seeds from JSON file."""
    with open(input_file, "r") as f:
        data = json.load(f)
    return data["seeds"]


def calculate_sdxl_dimensions(unet_config, input_resolution):
    """Calculate the correct dimensions for SDXL mid-block based on actual architecture"""
    block_out_channels = unet_config.block_out_channels
    mid_block_channels = block_out_channels[-1]  # 1280 for SDXL

    # Input: 1024x1024 -> VAE: 128x128 -> UNet: downsample 3 times -> ~32x32 mid-block
    latent_res = input_resolution // 8
    mid_block_res = input_resolution // 32

    return mid_block_channels, mid_block_res


def extract_h_spaces(pipe, prompt, seed):
    """
    Extract mid-block (h-space) activations for a single sample (batch size = 1).
   
    """
    h_space = []

    def get_h_space(module, input, output):
        # output: [B, C, H, W], here B=1
        h_space[-1].append(output.detach().cpu())

    # Disable concept influence for h-space extraction
    prev_cond = CURRENT_INPUT_CONDITIONS["tensor"]
    CURRENT_INPUT_CONDITIONS["tensor"] = None

    with torch.no_grad():
        handle = pipe.unet.mid_block.register_forward_hook(get_h_space)
        h_space.append([])

        gen = torch.Generator(pipe.device).manual_seed(seed)

        original_size = (1024, 1024)
        target_size = (1024, 1024)
        crops_coords_top_left = (0, 0)

        out = pipe(
            prompt=prompt,
            generator=gen,
            guidance_scale=7.5,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
        )

        handle.remove()

    # Restore previous condition
    CURRENT_INPUT_CONDITIONS["tensor"] = prev_cond

    # h_space: list[ list[tensor] ], each tensor [1, C, H, W]
    # -> [1, T, C, H, W]
    h_space = torch.cat([torch.stack(x, dim=1) for x in h_space])
    return h_space.numpy(), out.images[0]


def compute_dwt_and_reconstruct_ll(data, n_components=51):
    """
    Compute DWT for each channel and reconstruct using only LL subband.

    data: numpy array [B, T, C, H, W]; here typically B=1

    Returns dict:
      - 'reconstructed': numpy array with same shape as data[:n_components]
                         (for B=1 this is [1, T, C, H, W])
    """
    batch_size = data.shape[0]
    reconstructed_data = np.zeros_like(data[:n_components])

    for i in tqdm(range(batch_size), desc="DWT over batch"):
        sample = data[i]  # [T, C, H, W]
        reconstructed_channels_per_timestep = []

        # sample[t, c, :, :] is (H, W)
        T = sample.shape[0]
        C = sample.shape[1]

        # We'll reconstruct channel-wise for each timestep
        for t in range(T):
            rec_channels = []
            for ch in range(C):
                coeffs = pywt.dwt2(sample[t, ch], 'db1')
                # Keep only LL, zero-out LH/HL/HH
                if i < n_components:
                    new_coeffs = (
                        coeffs[0],
                        (
                            np.zeros_like(coeffs[1][0]),
                            np.zeros_like(coeffs[1][1]),
                            np.zeros_like(coeffs[1][2]),
                        ),
                    )
                    reconstructed = pywt.idwt2(new_coeffs, 'db1')
                    rec_channels.append(reconstructed)
            if i < n_components:
                rec_channels = np.stack(rec_channels)  # [C, H, W]
                reconstructed_channels_per_timestep.append(rec_channels)

        if i < n_components:
            reconstructed_channels_per_timestep = np.stack(
                reconstructed_channels_per_timestep
            )  # [T, C, H, W]
            # reconstructed_data[i]: [T, C, H, W]
            reconstructed_data[i] = reconstructed_channels_per_timestep

    return {'reconstructed': reconstructed_data}


def main():
    args = parse_args()

    # Use args.image_dir if it exists, otherwise default to "images"
    image_dir = getattr(args, "image_dir", "images_out")
    base_output_dir = os.path.join(args.output_dir, image_dir)
    CONCEPT_SCALE = 2.0
    # --------------------------------------------------
    # Output dirs for the four variants
    # --------------------------------------------------
    dirs = {
        'original': os.path.join(base_output_dir, 'original'),              # 1: no concept, no wavelet
        'concept_only': os.path.join(base_output_dir, 'concept_only'),      # 2: concept only
        'wavelet_only': os.path.join(base_output_dir, 'wavelet_only'),      # 3: wavelet only
        'concept_wavelet': os.path.join(base_output_dir, 'concept_wavelet') # 4: concept + wavelet
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # --------------------------------------------------
    # Seed management
    # --------------------------------------------------
    seeds_file = "generated_seeds.json"
    if not os.path.exists(seeds_file):
        seeds = generate_and_save_seeds(output_file=seeds_file)
        print(f"Generated and saved {len(seeds)} new seeds")
    else:
        seeds = load_seeds()
        print(f"Loaded {len(seeds)} existing seeds")

    device = 'cuda'
    weight_dtype = torch.float16 if args.fp16 else torch.float32

    print("Loading SDXL components...")

    # SDXL text encoders + tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2"
    )
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    # --------------------------------------------------
    # Load clean UNet weights from training
    # --------------------------------------------------
    EXP_DIR = "female_sdxl"
    UNET_CLEAN_PATH = os.path.join(EXP_DIR, "unet_clean_epoch_20.pth")
    CONCEPT_PATH = os.path.join(EXP_DIR, "concept_epoch_20.pt")
    CONCEPT_DICT_PATH = os.path.join(EXP_DIR, "concept_dict.json")

    print(f"Loading UNet clean weights from: {UNET_CLEAN_PATH}")
    clean_state = torch.load(UNET_CLEAN_PATH, map_location="cpu")
    missing, unexpected = unet.load_state_dict(clean_state, strict=False)
    print("UNet load_state_dict missing keys:", missing)
    print("UNet load_state_dict unexpected keys:", unexpected)

    # --------------------------------------------------
    # SDXL mid-block config & concept module
    # --------------------------------------------------
    mid_block_channels, mid_block_res = calculate_sdxl_dimensions(unet.config, args.resolution)
    print(
        f"SDXL Mid-block config: channels={mid_block_channels}, "
        f"resolution={mid_block_res}x{mid_block_res}"
    )

    MAX_CONCEPT_LENGTH = 100  # same as training
    concept_module = BottleneckConceptModule(
        num_concepts=MAX_CONCEPT_LENGTH,
        mid_channels=mid_block_channels,
    )
    unet.concept_module = concept_module

    # Load concept_module weights
    print(f"Loading concept module from: {CONCEPT_PATH}")
    concept_state = torch.load(CONCEPT_PATH, map_location="cpu")
    unet.concept_module.load_state_dict(concept_state)

    # Register mid-block hook for concept injection (always active)
    def mid_block_concept_hook(module, inputs, output):
        concept_onehot = CURRENT_INPUT_CONDITIONS["tensor"]
        if concept_onehot is None:
            return output
        out = unet.concept_module(output, concept_onehot)
        return output + CONCEPT_SCALE * (out - output)

    unet.mid_block.register_forward_hook(mid_block_concept_hook)

    # Move everything to device / dtype
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        unet=unet,
        scheduler=DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        ),
    ).to(device)

    pipe.vae.to(dtype=weight_dtype)
    pipe.text_encoder.to(dtype=weight_dtype)
    pipe.text_encoder_2.to(dtype=weight_dtype)
    pipe.unet.to(device)
    pipe.safety_checker = None

    # Load concept dict and build one-hot for "female"
    with open(CONCEPT_DICT_PATH, "r") as f:
        concept_dict = json.load(f)
    female_index = concept_dict["female"]  # 0 in your dict

    # Build one-hot [1, 100]
    concept_condition = torch.zeros(1, MAX_CONCEPT_LENGTH, device=device)
    concept_condition[:, female_index] = 1.0

    prompt = "a photo of a doctor"
    guidance_scale = 7.5

    print(f"Starting SDXL inference with {len(seeds)} seeds...")

    for seed in tqdm(seeds, desc="Processing seeds"):
        # -------------------------------
        # 1) ORIGINAL (no concept, no wavelet)
        # -------------------------------
        CURRENT_INPUT_CONDITIONS["tensor"] = None

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        original_size = (1024, 1024)
        target_size = (1024, 1024)
        crops_coords_top_left = (0, 0)

        out_original = pipe(
            prompt=prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
        )
        out_original.images[0].save(os.path.join(dirs['original'], f'seed_{seed}.jpg'))

        # -------------------------------
        # 2) CONCEPT ONLY (h + concept)
        # -------------------------------
        CURRENT_INPUT_CONDITIONS["tensor"] = concept_condition

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        out_concept = pipe(
            prompt=prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
        )
        out_concept.images[0].save(os.path.join(dirs['concept_only'], f'seed_{seed}.jpg'))

        # ----------------------------------------------------
        # 3) Wavelet pipeline:

        # ----------------------------------------------------
        h_out, _ = extract_h_spaces(pipe, prompt, seed)
        results = compute_dwt_and_reconstruct_ll(h_out, n_components=1)
        filtered_h_space = results['reconstructed']  # shape [1, T, C, H, W]
        filtered_h_space_tensor = torch.tensor(filtered_h_space).float().to(device)

        # Helper: register wavelet hook for a given factor
        def register_wavelet_hook(unet, modification, factor):
            step = {'i': 0}

            def modify_h_space(module, inputs, output):
                # modification: [1, T, C, H, W]
                T = modification.shape[1]
                idx = min(step['i'], T - 1)
                delta = modification[:, idx, :, :, :].squeeze(0).to(output.device)
                step['i'] += 1
                return output + factor * delta

            handle = unet.mid_block.register_forward_hook(modify_h_space)
            return handle

        # -------------------------------
        # 3) WAVELET ONLY (no concept)
        # -------------------------------
        CURRENT_INPUT_CONDITIONS["tensor"] = None

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        wavelet_handle = register_wavelet_hook(pipe.unet, filtered_h_space_tensor, factor=1.0)

        out_wavelet = pipe(
            prompt=prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
        )
        wavelet_handle.remove()
        out_wavelet.images[0].save(os.path.join(dirs['wavelet_only'], f'seed_{seed}.jpg'))

        # -------------------------------
        # 4) CONCEPT + WAVELET
        # -------------------------------
        CURRENT_INPUT_CONDITIONS["tensor"] = concept_condition

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        wavelet_handle = register_wavelet_hook(pipe.unet, filtered_h_space_tensor, factor=1.0)

        out_concept_wavelet = pipe(
            prompt=prompt,
            generator=gen,
            guidance_scale=guidance_scale,
            original_size=original_size,
            target_size=target_size,
            crops_coords_top_left=crops_coords_top_left,
        )
        wavelet_handle.remove()
        out_concept_wavelet.images[0].save(
            os.path.join(dirs['concept_wavelet'], f'seed_{seed}.jpg')
        )

    print("✅ SDXL inference completed successfully!")
    print(f"Images saved under: {base_output_dir}")
    print("Variants per seed:")
    print("  1) original/")
    print("  2) concept_only/")
    print("  3) wavelet_only/")
    print("  4) concept_wavelet/")


if __name__ == "__main__":
    main()
