import logging
import math
import os
from typing import List

from tqdm.auto import tqdm
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from config import parse_args
from utils_data import get_dataloader

logger = get_logger(__name__)

# -------------------------------------------------------------------
# Global storage for current batch's concept one-hot (for the hook)
# -------------------------------------------------------------------
CURRENT_INPUT_CONDITIONS = {"tensor": None}


class BottleneckConceptModule(nn.Module):
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
        self.concept_emb = nn.Parameter(torch.zeros(num_concepts, mid_channels))
        nn.init.normal_(self.concept_emb, mean=0.0, std=0.02)

    def forward(self, mid_tensor: torch.Tensor, concept_onehot: torch.Tensor) -> torch.Tensor:
        """
        mid_tensor: [B, C, H, W]
        concept_onehot: [B, num_concepts]
        """
        if concept_onehot is None:
            return mid_tensor

        B, C, H, W = mid_tensor.shape
        concept_onehot = concept_onehot.to(mid_tensor.device).float()

        # [B, num_concepts] @ [num_concepts, C] -> [B, C]
        delta = concept_onehot @ self.concept_emb  # [B, C]
        delta = delta.view(B, C, 1, 1)

        return mid_tensor + delta


def unfreeze_layers_unet(unet):
    # Kept for logging only; UNet itself stays frozen except for attached modules.
    print(
        "Num trainable params in UNet (including attached modules): ",
        sum(p.numel() for p in unet.parameters() if p.requires_grad),
    )
    return unet


def calculate_sdxl_dimensions(unet_config, input_resolution):
    """Calculate the correct dimensions for SDXL mid-block based on actual architecture."""
    block_out_channels = unet_config.block_out_channels
    mid_block_channels = block_out_channels[-1]  # 1280 for SDXL

    # Input: 1024x1024 -> VAE: 128x128 -> UNet: downsample 3 times -> empirically 32x32 mid-block
    latent_res = input_resolution // 8  # 1024 -> 128
    mid_block_res = input_resolution // 32

    return mid_block_channels, mid_block_res


def save_concept_module(unet, output_dir, name):
    """
    Save only the concept-module weights (no backbone) to keep checkpoints compatible
    with standard SDXL UNet loading.
    """
    os.makedirs(output_dir, exist_ok=True)
    state = unet.concept_module.state_dict()
    ckpt_path = os.path.join(output_dir, f"{name}.pt")
    torch.save(state, ckpt_path)
    print(f"💾 Saved concept module to {ckpt_path}")


def save_unet_without_concept(unet, output_path):
    """
    Optionally save a 'clean' UNet checkpoint with concept_module.* keys stripped out,
    so it can be loaded into a vanilla UNet2DConditionModel.
    """
    full_state = unet.state_dict()
    clean_state = {k: v for k, v in full_state.items() if not k.startswith("concept_module.")}
    torch.save(clean_state, output_path)
    print(f"💾 Saved clean UNet weights (no concept_module) to {output_path}")


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.dump(vars(args), open(os.path.join(args.output_dir, "config.yaml"), "w"))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",  # Force FP32
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SDXL components...")

    # SDXL uses dual tokenizers and text encoders
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision
    )

    # SDXL has two text encoders
    text_encoder_one = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    print("✓ Loaded all SDXL components")

    # SDXL mid-block config
    mid_block_channels, mid_block_res = calculate_sdxl_dimensions(unet.config, args.resolution)
    print(
        f"SDXL Mid-block config: channels={mid_block_channels}, "
        f"resolution={mid_block_res}x{mid_block_res}"
    )
    print(f"SDXL Block out channels: {unet.config.block_out_channels}")

    # Use FP32 everywhere
    weight_dtype = torch.float32
    print(f"Using weight_dtype: {weight_dtype}")

    # Freeze backbone
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # ------------------------------------------------------------------
    # Attach bottleneck concept module and hook into mid-block
    # ------------------------------------------------------------------
    MAX_CONCEPT_LENGTH = 100  # must match utils_data.get_dataloader(..., max_concept_length=100)

    concept_module = BottleneckConceptModule(
        num_concepts=MAX_CONCEPT_LENGTH,
        mid_channels=mid_block_channels,
    )
    # Attach as submodule so parameters are tracked under UNet
    unet.concept_module = concept_module

    def mid_block_hook(module, inputs, output):
        # module is unet.mid_block, output is mid-block activations [B, C, H, W]
        concept_onehot = CURRENT_INPUT_CONDITIONS["tensor"]
        if concept_onehot is None:
            return output
        return concept_module(output, concept_onehot)

    # Register hook on the mid-block BEFORE wrapping with Accelerator
    unet.mid_block.register_forward_hook(mid_block_hook)

    # Just log params
    unet = unfreeze_layers_unet(unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing")

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Only train the concept module parameters
    trainable_params = list(concept_module.parameters())
    print("Trainable params in concept module:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ------------------------------------------------------------------
    # Tokenization wrapper for utils_data TrainingDataset
    # ------------------------------------------------------------------
    def tokenize_captions_sdxl(texts, is_train: bool = True):
        """
        texts: list[str]
        Returns: list[(ids_one, ids_two)]
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs_one = tokenizer_one(
            texts,
            max_length=tokenizer_one.model_max_length,
            padding=False,
            truncation=True,
        )
        inputs_two = tokenizer_two(
            texts,
            max_length=tokenizer_two.model_max_length,
            padding=False,
            truncation=True,
        )

        ids_one = inputs_one.input_ids  # list[list[int]]
        ids_two = inputs_two.input_ids  # list[list[int]]
        combined = list(zip(ids_one, ids_two))  # list[(ids1, ids2)]

        return combined

    # ------------------------------------------------------------------
    # Transforms & collate (NO HEATMAPS)
    # ------------------------------------------------------------------
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        # examples: list of (image, (ids_one, ids_two), concept_onehot)
        pixel_values = torch.stack([ex[0] for ex in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        # We ignore text from dataset at training time, but keep interface consistent
        input_ids_one = [ex[1][0] for ex in examples]
        input_ids_two = [ex[1][1] for ex in examples]

        padded_tokens_one = tokenizer_one.pad(
            {"input_ids": input_ids_one},
            padding=True,
            return_tensors="pt",
        )
        padded_tokens_two = tokenizer_two.pad(
            {"input_ids": input_ids_two},
            padding=True,
            return_tensors="pt",
        )

        # one-hot concept vectors
        input_conditions = torch.stack([ex[2] for ex in examples]).float()

        return {
            "pixel_values": pixel_values,
            "input_ids_one": padded_tokens_one.input_ids,
            "attention_mask_one": padded_tokens_one.attention_mask,
            "input_ids_two": padded_tokens_two.input_ids,
            "attention_mask_two": padded_tokens_two.attention_mask,
            "input_conditions": input_conditions,
        }

    train_dataloader = get_dataloader(
        args.train_data_dir,
        batch_size=args.train_batch_size,
        shuffle=True,
        transform=train_transforms,
        tokenizer=tokenize_captions_sdxl,
        collate_fn=collate_fn,
        num_workers=0,
        max_concept_length=MAX_CONCEPT_LENGTH,
        select=args.select,
    )

    # ------------------------------------------------------------------
    # Training steps & scheduler
    # ------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Move models to device
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device)

    # Prepare with accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        exp_name = f"{os.path.basename(args.output_dir)}_SDXL_lr{args.learning_rate}"
        accelerator.init_trackers(
            project_name="sdxl-bottleneck-concepts",
            config={k: v for k, v in vars(args).items() if k != "config"},
            init_kwargs={"wandb": {"name": exp_name}},
        )

    # ------------------------------------------------------------------
    # Prompt encoding for arbitrary text prompts
    # ------------------------------------------------------------------
    def encode_texts_sdxl(prompts: List[str]):
        """
        Encode a list of text prompts using both SDXL text encoders.

        Returns:
            prompt_embeds: [B, L, C1+C2]
            pooled_prompt_embeds: [B, C2]
        """
        inputs_one = tokenizer_one(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_one.model_max_length,
            return_tensors="pt",
        )
        inputs_two = tokenizer_two(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=tokenizer_two.model_max_length,
            return_tensors="pt",
        )

        input_ids_one = inputs_one.input_ids.to(accelerator.device)
        attn_one = inputs_one.attention_mask.to(accelerator.device)
        input_ids_two = inputs_two.input_ids.to(accelerator.device)
        attn_two = inputs_two.attention_mask.to(accelerator.device)

        with torch.no_grad():
            # Encoder 1
            prompt_embeds_one = text_encoder_one(
                input_ids_one,
                attention_mask=attn_one,
            )[0]

            # Encoder 2
            prompt_embeds_two_out = text_encoder_two(
                input_ids_two,
                attention_mask=attn_two,
            )

            pooled_prompt_embeds = prompt_embeds_two_out[0]
            prompt_embeds_two = prompt_embeds_two_out.last_hidden_state

            prompt_embeds_one = prompt_embeds_one.to(dtype=weight_dtype)
            prompt_embeds_two = prompt_embeds_two.to(dtype=weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)

            prompt_embeds = torch.concat([prompt_embeds_one, prompt_embeds_two], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    # ------------------------------------------------------------------
    # Training loop (two-prompt comparison loss)
    # ------------------------------------------------------------------
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running SDXL + bottleneck-concept training (two-prompt loss) *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="SDXL Training Steps",
    )

    device = accelerator.device
    print(f"Starting SDXL training on {device}")
    print(
        f"Expected mid-block tensor shape: [batch_size, {mid_block_channels}, {mid_block_res}, {mid_block_res}]"
    )

    loss_history = []
    train_loss = 0.0
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Encode image -> latents (backbone frozen; we still want grad through latents)
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(weight_dtype).to(device)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 🔑 ensure gradients flow through UNet + bottleneck concept module
                latents = latents.detach()
                latents.requires_grad_(True)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                ).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL additional time conditioning
                original_size = torch.tensor(
                    [args.resolution, args.resolution], dtype=torch.long
                )
                target_size = torch.tensor(
                    [args.resolution, args.resolution], dtype=torch.long
                )
                crops_coords_top_left = torch.tensor([0, 0], dtype=torch.long)

                add_time_ids = (
                    torch.cat([original_size, crops_coords_top_left, target_size])
                    .unsqueeze(0)
                    .repeat(bsz, 1)
                    .to(device, dtype=weight_dtype)
                )

            
                # ------------------------------------------------------
                B = batch["pixel_values"].shape[0]
                person_prompts = ["a photo of a person"] * B
                concept_prompts = ["a photo of a woman"] * B  # for female concept

                encoder_hidden_states_person, pooled_person = encode_texts_sdxl(person_prompts)
                encoder_hidden_states_concept, pooled_concept = encode_texts_sdxl(concept_prompts)

                # Debug once
                if global_step == 0:
                    print("\nDEBUG - First step tensor shapes:")
                    print(f"  noisy_latents: {noisy_latents.shape}")
                    print(f"  encoder_hidden_states_person: {encoder_hidden_states_person.shape}")
                    print(f"  encoder_hidden_states_concept: {encoder_hidden_states_concept.shape}")
                    print(f"  pooled_person: {pooled_person.shape}")
                    print(f"  pooled_concept: {pooled_concept.shape}")
                    print(f"  add_time_ids: {add_time_ids.shape}")
                    print(f"  input_conditions: {batch['input_conditions'].shape}")

                # ------------------------------------------------------
                # Person pass: neutral prompt + CONCEPT VECTORS ACTIVE
                # ------------------------------------------------------
                CURRENT_INPUT_CONDITIONS["tensor"] = batch["input_conditions"].to(device)

                model_pred_person = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states_person,
                    added_cond_kwargs={
                        "text_embeds": pooled_person,
                        "time_ids": add_time_ids,
                    },
                ).sample

                # ------------------------------------------------------
                # Concept pass: explicit prompt + FROZEN reference
                #   - no concept vectors (CURRENT_INPUT_CONDITIONS = None)
                #   - no grad (torch.no_grad)
                # ------------------------------------------------------
                CURRENT_INPUT_CONDITIONS["tensor"] = None
                with torch.no_grad():
                    model_pred_concept = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states_concept,
                        added_cond_kwargs={
                            "text_embeds": pooled_concept,
                            "time_ids": add_time_ids,
                        },
                    ).sample

                # ------------------------------------------------------
                # Concept-alignment loss: match person+concept_vecs to explicit concept prompt
                # ------------------------------------------------------
                loss = F.mse_loss(
                    model_pred_person.float(),
                    model_pred_concept.float(),
                    reduction="mean",
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Logging
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                train_loss += loss.detach().item()

                if global_step % 10 == 0:
                    avg_loss = train_loss / 10
                    accelerator.log(
                        {
                            "train_loss": avg_loss,
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                    loss_history.append(avg_loss)
                    train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

        # Checkpointing by epoch: save concept module + optional clean UNet
        if (epoch + 1) % 10 == 0 and accelerator.is_main_process:
            raw_unet = accelerator.unwrap_model(unet)
            save_concept_module(raw_unet, args.output_dir, f"concept_epoch_{epoch+1}")
            save_unet_without_concept(
                raw_unet, os.path.join(args.output_dir, f"unet_clean_epoch_{epoch+1}.pth")
            )

    # Final save + loss plot
    if accelerator.is_main_process:
        raw_unet = accelerator.unwrap_model(unet)
        save_concept_module(raw_unet, args.output_dir, "concept_final")
        save_unet_without_concept(raw_unet, os.path.join(args.output_dir, "unet_clean_final.pth"))

        plt.figure(figsize=(10, 6))
        plt.plot(loss_history)
        plt.title("SDXL Bottleneck Concept Training Loss History (Two-Prompt)")
        plt.xlabel("Steps (x10)")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(
            os.path.join(args.output_dir, "loss_history.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        with open(os.path.join(args.output_dir, "loss_history.json"), "w") as f:
            json.dump(loss_history, f)

        print("✓ SDXL training completed successfully!")
        print(
            f"Final concept module saved to: "
            f"{os.path.join(args.output_dir, 'concept_final.pt')}"
        )
        print(
            f"Clean UNet weights (no concept_module) saved to: "
            f"{os.path.join(args.output_dir, 'unet_clean_final.pth')}"
        )


if __name__ == "__main__":
    main()
