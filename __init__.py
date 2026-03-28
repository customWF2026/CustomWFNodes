import torch
import numpy as np
import os
import folder_paths
import comfy.utils


def _get_diffusers_dirs():
    """List model directories across all registered diffusers paths."""
    dirs = []
    for search_path in folder_paths.get_folder_paths("diffusers"):
        if os.path.isdir(search_path):
            for name in sorted(os.listdir(search_path)):
                full = os.path.join(search_path, name)
                if os.path.isdir(full) and not name.startswith("."):
                    if name not in dirs:
                        dirs.append(name)
    return dirs if dirs else ["AiArtLab--sdxs-1b"]


def _resolve_diffusers_path(model_name):
    """Find the full path for a diffusers model directory."""
    for search_path in folder_paths.get_folder_paths("diffusers"):
        full = os.path.join(search_path, model_name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(
        f"Model directory '{model_name}' not found. "
        f"Searched: {folder_paths.get_folder_paths('diffusers')}"
    )


# Cache loaded pipeline to avoid reloading across nodes
_pipeline_cache = {}


def _load_pipeline(model_name, dtype_str):
    cache_key = f"{model_name}_{dtype_str}"
    if cache_key not in _pipeline_cache:
        from diffusers import DiffusionPipeline

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[dtype_str]

        model_path = _resolve_diffusers_path(model_name)

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if device == "cpu":
            torch_dtype = torch.float32

        print(f"[SDXS-1B] Loading pipeline from {model_path} ({dtype_str})...")
        pipe = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
        print(f"[SDXS-1B] Pipeline loaded on {device}")

        _pipeline_cache[cache_key] = {
            "pipe": pipe,
            "device": device,
            "dtype": torch_dtype,
        }

    return _pipeline_cache[cache_key]


class SDXS1BQwenLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_diffusers_dirs(),),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("SDXS_CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "load"
    CATEGORY = "SDXS-1B"

    def load(self, model_name, dtype):
        cached = _load_pipeline(model_name, dtype)
        pipe = cached["pipe"]
        return (
            {
                "text_encoder": pipe.text_encoder,
                "tokenizer": pipe.tokenizer,
                "device": cached["device"],
            },
        )


class SDXS1BUnetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_diffusers_dirs(),),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("SDXS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "SDXS-1B"

    def load(self, model_name, dtype):
        cached = _load_pipeline(model_name, dtype)
        pipe = cached["pipe"]
        vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
        return (
            {
                "unet": pipe.unet,
                "scheduler": pipe.scheduler,
                "device": cached["device"],
                "dtype": cached["dtype"],
                "vae_scale_factor": vae_scale_factor,
            },
        )


class SDXS1BVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_get_diffusers_dirs(),),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float16"}),
            }
        }

    RETURN_TYPES = ("SDXS_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load"
    CATEGORY = "SDXS-1B"

    def load(self, model_name, dtype):
        cached = _load_pipeline(model_name, dtype)
        pipe = cached["pipe"]
        # Per-channel latent normalization (from pipeline_sdxs.py)
        mean = getattr(pipe.vae.config, "latents_mean", None)
        std = getattr(pipe.vae.config, "latents_std", None)
        vae_data = {"vae": pipe.vae}
        if mean is not None and std is not None:
            vae_data["latents_mean"] = torch.tensor(mean).view(1, len(mean), 1, 1)
            vae_data["latents_std"] = torch.tensor(std).view(1, len(std), 1, 1)
        return (vae_data,)


class SDXS1BClipTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("SDXS_CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("SDXS_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "SDXS-1B"

    def encode(self, clip, text):
        text_encoder = clip["text_encoder"]
        tokenizer = clip["tokenizer"]
        device = clip["device"]

        # Chat template formatting (from pipeline_sdxs.py)
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        formatted = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        toks = tokenizer(
            [formatted],
            padding="max_length",
            max_length=248,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=toks.input_ids,
                attention_mask=toks.attention_mask,
                output_hidden_states=True,
            )

        # Layer -2 hidden states
        hidden = outputs.hidden_states[-2]

        # Pool from last non-padding token
        seq_len = toks.attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(len(hidden)), seq_len.clamp(min=0)]

        # Prepend pooled to sequence
        encoder_hidden_states = torch.cat([pooled.unsqueeze(1), hidden], dim=1)
        attention_mask = torch.cat(
            [torch.ones((hidden.shape[0], 1), device=device), toks.attention_mask],
            dim=1,
        )

        return (
            {
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": attention_mask,
            },
        )


class SDXS1BSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SDXS_MODEL",),
                "positive": ("SDXS_COND",),
                "negative": ("SDXS_COND",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 150}),
                "cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.5},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
            }
        }

    RETURN_TYPES = ("SDXS_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "SDXS-1B"

    def sample(self, model, positive, negative, seed, steps, cfg, width, height):
        unet = model["unet"]
        scheduler = model["scheduler"]
        device = model["device"]
        dtype = model["dtype"]

        generator = torch.Generator(device="cpu").manual_seed(seed)

        vae_scale_factor = model["vae_scale_factor"]
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor
        in_channels = unet.config.in_channels

        latents = torch.randn(
            (1, in_channels, latent_h, latent_w),
            generator=generator,
            device="cpu",
            dtype=dtype,
        ).to(device)

        scheduler.set_timesteps(steps, device=device)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        pbar = comfy.utils.ProgressBar(steps)

        for t in scheduler.timesteps:
            if cfg > 1.0:
                latent_input = torch.cat([latents] * 2)
                enc_hidden = torch.cat(
                    [
                        negative["encoder_hidden_states"],
                        positive["encoder_hidden_states"],
                    ]
                )
                enc_mask = torch.cat(
                    [
                        negative["encoder_attention_mask"],
                        positive["encoder_attention_mask"],
                    ]
                )
            else:
                latent_input = latents
                enc_hidden = positive["encoder_hidden_states"]
                enc_mask = positive["encoder_attention_mask"]

            with torch.no_grad():
                noise_pred = unet(
                    latent_input,
                    t,
                    encoder_hidden_states=enc_hidden,
                    encoder_attention_mask=enc_mask,
                    return_dict=False,
                )[0]

            if cfg > 1.0:
                pred_uncond, pred_cond = noise_pred.chunk(2)
                noise_pred = pred_uncond + cfg * (pred_cond - pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            pbar.update(1)

        return ({"samples": latents},)


class SDXS1BVAEDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("SDXS_LATENT",),
                "vae": ("SDXS_VAE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "SDXS-1B"

    def decode(self, latent, vae):
        vae_model = vae["vae"]
        samples = latent["samples"]

        with torch.no_grad():
            # Denormalize latents using per-channel mean/std (from pipeline_sdxs.py)
            if "latents_mean" in vae and "latents_std" in vae:
                lat_std = vae["latents_std"].to(samples.device, samples.dtype)
                lat_mean = vae["latents_mean"].to(samples.device, samples.dtype)
                samples = samples * lat_std + lat_mean
            elif hasattr(vae_model.config, "scaling_factor"):
                samples = samples / vae_model.config.scaling_factor

            image = vae_model.decode(samples.to(vae_model.dtype), return_dict=False)[0]

        image = (image.clamp(-1, 1) + 1) / 2
        # ComfyUI IMAGE format: [B, H, W, C] float32 0-1
        image = image.permute(0, 2, 3, 1).cpu().float()

        return (image,)


NODE_CLASS_MAPPINGS = {
    "SDXS1BQwenLoader": SDXS1BQwenLoader,
    "SDXS1BUnetLoader": SDXS1BUnetLoader,
    "SDXS1BVAELoader": SDXS1BVAELoader,
    "SDXS1BClipTextEncode": SDXS1BClipTextEncode,
    "SDXS1BSampler": SDXS1BSampler,
    "SDXS1BVAEDecode": SDXS1BVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SDXS1BQwenLoader": "Load SDXS-1B Qwen Text Encoder",
    "SDXS1BUnetLoader": "Load SDXS-1B UNet",
    "SDXS1BVAELoader": "Load SDXS-1B VAE",
    "SDXS1BClipTextEncode": "SDXS-1B Qwen Text Encode",
    "SDXS1BSampler": "SDXS-1B Sampler",
    "SDXS1BVAEDecode": "SDXS-1B VAE Decode",
}

print(
    "\033[92m[SDXS-1B] Loaded 6 nodes: "
    "SDXS1BQwenLoader, SDXS1BUnetLoader, SDXS1BVAELoader, "
    "SDXS1BClipTextEncode, SDXS1BSampler, SDXS1BVAEDecode\033[0m"
)
