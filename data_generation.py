# Simple SDXL image generator (no DAAM, no heatmaps)

import os
import argparse
import json
import zipfile
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from utils import set_seed, auto_device


class SDXLImageDataCreator:
    def __init__(self,
                 prompt: str,
                 concepts: List[str],
                 output_dir: str,
                 num_samples: int = 10,
                 guidance_scale: float = 7.5,
                 num_inference_steps: int = 50,
                 height: int = 1024,  # SDXL native resolution
                 width: int = 1024,   # SDXL native resolution
                 low_memory: bool = False,
                 base_seed: Optional[int] = 42,
                 model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 use_refiner: bool = False,
                 refiner_model_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
                 high_noise_frac: float = 0.8,
                 create_zip: bool = True,
                 zip_filename: Optional[str] = None):
        """
        SDXL data creator that only generates images (no heatmaps, no DAAM).

        Args:
            prompt: The text prompt to generate images from.
            concepts: Kept for metadata / labels compatibility (not used for heatmaps).
            output_dir: Directory to save the outputs.
            num_samples: Number of samples to generate.
            guidance_scale: Guidance scale for the diffusion model.
            num_inference_steps: Number of denoising steps.
            height: Height of the generated images (1024 for SDXL).
            width: Width of the generated images (1024 for SDXL).
            low_memory: Whether to use low memory tricks.
            base_seed: Starting seed for reproducibility.
            model_path: Path to the pretrained SDXL base model.
            use_refiner: Whether to use SDXL refiner.
            refiner_model_path: Path to the SDXL refiner model.
            high_noise_frac: Fraction of noise steps for base model when using refiner.
            create_zip: Whether to create a zip file of the generated content.
            zip_filename: Custom name for the zip file (optional).
        """
        self.prompt = prompt
        self.concepts = concepts
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.height = height
        self.width = width
        self.low_memory = low_memory
        self.base_seed = base_seed
        self.model_path = model_path
        self.use_refiner = use_refiner
        self.refiner_model_path = refiner_model_path
        self.high_noise_frac = high_noise_frac
        self.create_zip = create_zip
        self.zip_filename = zip_filename

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir
        self.images_dir.mkdir(exist_ok=True)

        # Set up metadata
        self.metadata = {
            "prompt": prompt,
            "concepts": concepts,
            "samples": [],
        }

        # Set up the SDXL model
        print(f"Loading SDXL base model from {model_path}...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None
        )
        self.pipe = auto_device(self.pipe)

        # Optional low-memory tweaks
        if self.low_memory:
            print("Enabling attention/vae slicing for low-memory mode.")
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        # Disable safety checker for research purposes
        self.pipe.safety_checker = None

        # Load refiner if requested
        if self.use_refiner:
            print(f"Loading SDXL refiner model from {refiner_model_path}...")
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_model_path,
                text_encoder_2=self.pipe.text_encoder_2,
                vae=self.pipe.vae,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
            self.refiner = auto_device(self.refiner)

            if self.low_memory:
                self.refiner.enable_attention_slicing()
                self.refiner.enable_vae_slicing()
        else:
            self.refiner = None

    def generate_sample(self, sample_idx: int):
        """
        Generate a single sample (image only) using SDXL.

        Args:
            sample_idx: Index of the sample to generate.

        Returns:
            Dict containing metadata for the generated sample.
        """
        # Use deterministic seed based on sample index
        seed = self.base_seed + sample_idx if self.base_seed is not None else None
        generator = set_seed(seed) if seed is not None else None

        print(f"\nGenerating sample {sample_idx + 1}/{self.num_samples}")
        print(f"Prompt: '{self.prompt}', Seed: {seed}")

        # Generate the image with SDXL
        if self.use_refiner and self.refiner is not None:
            # Base model: generate latent
            base_out = self.pipe(
                prompt=self.prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                denoising_end=self.high_noise_frac,
                output_type="latent",
            )
            latent = base_out.images[0]

            # Refiner: denoise latent to image
            image = self.refiner(
                prompt=self.prompt,
                image=latent,
                num_inference_steps=self.num_inference_steps,
                denoising_start=self.high_noise_frac,
                generator=generator,
            ).images[0]
        else:
            # Use base model only, directly to image
            image = self.pipe(
                prompt=self.prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
            ).images[0]

        # Save the generated image
        image_path = self.images_dir / f"image_{sample_idx:04d}.jpg"
        image.save(image_path)
        print(f"Saved image to {image_path}")

        # Record metadata for this sample
        sample_metadata = {
            "index": sample_idx,
            "seed": seed,
            "image_path": str(image_path.relative_to(self.output_dir)),
        }

        return sample_metadata

    def create_zip_archive(self):
        """
        Create a zip archive of all generated content.
        """
        if not self.create_zip:
            return

        # Determine zip filename
        if self.zip_filename:
            zip_path = Path(self.zip_filename)
            if not zip_path.suffix:
                zip_path = zip_path.with_suffix('.zip')
        else:
            # Create a descriptive filename based on prompt and timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in self.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]  # Truncate if too long
            zip_path = self.output_dir.parent / f"generated_images_{safe_prompt}_{timestamp}.zip"

        print(f"\nCreating zip archive: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                # Add all files in the output directory
                for file_path in self.output_dir.rglob('*'):
                    if file_path.is_file():
                        # Get the relative path for the archive
                        arcname = file_path.relative_to(self.output_dir.parent)
                        zipf.write(file_path, arcname)

            print(f"✓ Successfully created zip archive: {zip_path}")
            print(f"Archive size: {zip_path.stat().st_size / (1024*1024):.1f} MB")

        except Exception as e:
            print(f"✗ Error creating zip archive: {e}")

    def run(self):
        """
        Generate all samples and save metadata.
        """
        print(f"Starting generation of {self.num_samples} samples for prompt: '{self.prompt}'")
        print(f"Output directory: {self.output_dir}")
        print(f"Using SDXL with resolution: {self.width}x{self.height}")
        if self.use_refiner:
            print("Using SDXL refiner for enhanced quality")
        if self.create_zip:
            print("Will create zip archive after generation")

        # Generate all samples with progress bar
        for i in tqdm(range(self.num_samples), desc="Generating samples"):
            sample_metadata = self.generate_sample(i)
            self.metadata["samples"].append(sample_metadata)

            # Save metadata after each sample (in case of interruptions)
            with open(self.output_dir / "metadata.json", "w") as f:
                json.dump(self.metadata, f, indent=2)

        print(f"\nGeneration completed successfully.")
        print(f"Generated {self.num_samples} samples.")
        print(f"Images saved to: {self.images_dir}")
        print(f"Metadata saved to: {self.output_dir / 'metadata.json'}")

        # Create zip archive if requested
        self.create_zip_archive()


def repeat_ntimes(x, n):
    return [item for item in x for i in range(n)]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with SDXL (no heatmaps, no DAAM)")
    parser.add_argument("--prompt", type=str, default="a photo of a man", help="Text prompt for image generation")
    parser.add_argument("--concepts", nargs="+", default=["man"], help="(Optional) concepts, kept for metadata only")
    parser.add_argument("--output_dir", type=str, default="/project/def-ilminkim/Moslem/SD_ViT/datasets_SDXL_male", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--low_memory", action="store_true", help="Use low memory mode (attention/vae slicing)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (SDXL native: 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width (SDXL native: 1024)")
    parser.add_argument("--model_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Path to pretrained SDXL base model")
    parser.add_argument("--use_refiner", action="store_true", help="Use SDXL refiner model")
    parser.add_argument("--refiner_model_path", type=str, default="stabilityai/stable-diffusion-xl-refiner-1.0",
                        help="Path to SDXL refiner model")
    parser.add_argument("--high_noise_frac", type=float, default=0.8,
                        help="Fraction of noise steps for base model when using refiner")
    parser.add_argument("--create_zip", action="store_true", default=True,
                        help="Create zip archive of generated content")
    parser.add_argument("--no_zip", action="store_false", dest="create_zip", help="Skip creating zip archive")
    parser.add_argument("--zip_filename", type=str, default=None, help="Custom filename for zip archive")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Keep label/metadata JSONs similar to your original script so your downstream
    # pipeline does not break, but they are not used for heatmaps anymore.
    concept_dict = ["female", "male", "young", "old", "white-race", "black-race", "anti-sexual"]
    concept_dict = {c: i for i, c in enumerate(concept_dict)}
    image_prompt = [
        "a photo of a woman",
    ]

    input_prompt_and_target_concept = [
        [
            ["a photo of a person", ["female"]],
        ],
    ]

    image_prompt = repeat_ntimes(image_prompt, args.num_samples)
    input_prompt_and_target_concept = repeat_ntimes(input_prompt_and_target_concept, args.num_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(input_prompt_and_target_concept, open(os.path.join(args.output_dir, "labels.json"), "w"))
    json.dump(concept_dict, open(os.path.join(args.output_dir, "concept_dict.json"), "w"))

    creator = SDXLImageDataCreator(
        prompt=args.prompt,
        concepts=args.concepts,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        low_memory=args.low_memory,
        base_seed=args.seed,
        model_path=args.model_path,
        use_refiner=args.use_refiner,
        refiner_model_path=args.refiner_model_path,
        high_noise_frac=args.high_noise_frac,
        create_zip=args.create_zip,
        zip_filename=args.zip_filename
    )

    creator.run()
