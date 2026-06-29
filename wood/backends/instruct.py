"""Frozen InstructPix2Pix white-box internals for WOOD."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class InstructSettings:
    model_id: str = "timbrooks/instruct-pix2pix"
    torch_dtype: str = "float16"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    objective_timestep_index: int = 6
    seed: int = 1234


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return aliases[name.lower()]


class InstructBackend:
    name = "instruct"

    def __init__(self, device: torch.device, settings: InstructSettings | None = None) -> None:
        self.device = device
        self.settings = settings or InstructSettings()
        self.pipe = self._load()

    def _load(self):
        from diffusers import StableDiffusionInstructPix2PixPipeline

        if self.device.type != "cuda":
            raise RuntimeError("InstructPix2Pix white-box WOOD runs require CUDA.")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.settings.model_id,
            torch_dtype=_dtype(self.settings.torch_dtype),
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        pipe.set_progress_bar_config(disable=True)
        for module_name in ("vae", "text_encoder", "unet"):
            module = getattr(pipe, module_name, None)
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        return pipe

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        if hasattr(self.pipe, "_encode_prompt"):
            return self.pipe._encode_prompt(prompt, self.device, 1, False)
        encoded = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        if isinstance(encoded, tuple):
            return encoded[0]
        return encoded

    def encode_image_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        image = (image_tensor * 2.0 - 1.0).to(device=self.device, dtype=self.pipe.vae.dtype)
        latent = self.pipe.vae.encode(image).latent_dist.mode()
        return latent.to(dtype=self.pipe.unet.dtype)

    def _unet_prediction(self, image_latent: torch.Tensor, embedding: torch.Tensor, reference: dict[str, Any]) -> torch.Tensor:
        noisy = self.pipe.scheduler.scale_model_input(reference["fixed_noise"], reference["timestep"])
        sample = torch.cat([noisy.to(dtype=self.pipe.unet.dtype), image_latent.to(dtype=self.pipe.unet.dtype)], dim=1)
        return self.pipe.unet(
            sample,
            reference["timestep"],
            encoder_hidden_states=embedding,
            return_dict=False,
        )[0]

    def prepare_blank_reference(
        self,
        original_tensor: torch.Tensor,
        blank_tensor: torch.Tensor,
        prompt: str,
        objective: str,
    ) -> dict[str, Any]:
        """Compute blank objective-space references once for a stable Z."""

        with torch.no_grad():
            prompt_embedding = self._encode_prompt(prompt).detach()
            blank_latent = self.encode_image_latent(blank_tensor).detach()
            original_latent = self.encode_image_latent(original_tensor).detach()

            self.pipe.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
            steps = self.pipe.scheduler.timesteps
            timestep = steps[min(max(0, self.settings.objective_timestep_index), len(steps) - 1)]
            generator = torch.Generator(device=self.device).manual_seed(self.settings.seed)
            fixed_noise = torch.randn(
                blank_latent.shape,
                generator=generator,
                device=self.device,
                dtype=self.pipe.unet.dtype,
            ) * self.pipe.scheduler.init_noise_sigma

            payload = {
                "prompt": prompt,
                "objective": objective,
                "prompt_embedding": prompt_embedding,
                "blank_latent": blank_latent,
                "original_latent": original_latent,
                "fixed_noise": fixed_noise,
                "timestep": timestep,
            }
            payload["blank_prediction"] = self._unet_prediction(blank_latent, prompt_embedding, payload).detach()
            payload["original_prediction"] = self._unet_prediction(original_latent, prompt_embedding, payload).detach()
        return payload

    def compute_Z(
        self,
        perturbed: torch.Tensor,
        reference: dict[str, Any],
        objective: str,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        perturbed_latent = self.encode_image_latent(perturbed)
        prediction = self._unet_prediction(perturbed_latent, reference["prompt_embedding"], reference)

        vae_to_blank = F.mse_loss(perturbed_latent.float(), reference["blank_latent"].float())
        unet_to_blank = F.mse_loss(prediction.float(), reference["blank_prediction"].float())
        vae_to_original = F.mse_loss(perturbed_latent.float(), reference["original_latent"].float())
        unet_to_original = F.mse_loss(prediction.float(), reference["original_prediction"].float())

        if objective == "vae_conditioning":
            Z = vae_to_blank
            original_diag = vae_to_original
        elif objective == "unet_prediction":
            Z = unet_to_blank
            original_diag = unet_to_original
        else:
            raise ValueError(f"Unsupported InstructPix2Pix objective: {objective}")

        return Z, {
            "Z_to_blank_objective": Z,
            "Z_to_original_objective_diagnostic": original_diag,
            "vae_conditioning_Z_to_blank": vae_to_blank,
            "unet_prediction_Z_to_blank": unet_to_blank,
            "vae_conditioning_Z_to_original_diagnostic": vae_to_original,
            "unet_prediction_Z_to_original_diagnostic": unet_to_original,
        }

    @torch.inference_mode()
    def generate_edit(self, image: Image.Image, prompt: str, seed: int) -> Image.Image:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=self.settings.num_inference_steps,
            guidance_scale=self.settings.guidance_scale,
            image_guidance_scale=self.settings.image_guidance_scale,
            generator=generator,
        )
        return result.images[0].convert("RGB")
