# ComfyUI-SDXS1B

Custom ComfyUI nodes for [AiArtLab/sdxs-1b](https://huggingface.co/AiArtLab/sdxs-1b) - a fast text-to-image model using a Qwen 3.5 text encoder, modified SD 1.5 UNet (1.5B params), and asymmetric FLUX 2-based VAE.

## Nodes

| Node | Description |
|------|-------------|
| **Load SDXS-1B Qwen Text Encoder** | Loads the Qwen 3.5-2B text encoder and tokenizer |
| **Load SDXS-1B UNet** | Loads the 1.5B param UNet and flow-matching scheduler |
| **Load SDXS-1B VAE** | Loads the asymmetric VAE (8x encode / 16x decode) |
| **SDXS-1B Qwen Text Encode** | Encodes text with Qwen 3.5 (chat template, layer -2 pooling) |
| **SDXS-1B Sampler** | Denoising sampler with progress bar |
| **SDXS-1B VAE Decode** | Decodes latents to images |

All three loader nodes share a pipeline cache, so the model is only loaded into VRAM once.

## Workflow

```
Qwen Loader --CLIP--> Text Encode (+) -->
                       Text Encode (-) --> Sampler --> VAE Decode --> Preview
UNet Loader --MODEL------------------->
VAE Loader  --VAE--------------------------------------------->
```

### Default settings (from model card)

- Resolution: 1024x1024
- Steps: 40
- CFG: 4.0
- Sampler: Flow Match Euler Discrete

## Requirements

- ComfyUI (any recent version)
- Python packages (install in your ComfyUI venv):

```bash
pip install diffusers transformers>=4.52 accelerate
```

`transformers>=4.52` is required for Qwen 3.5 model class support.

## Installation

### 1. Install the custom nodes

Copy this folder into your ComfyUI `custom_nodes` directory:

```
ComfyUI/
  custom_nodes/
    ComfyUI-SDXS1B/
      __init__.py
      sdxs-1b-workflow.json
      README.md
```

### 2. Download the model

The model must be cloned as a full diffusers directory (a single `.safetensors` file will not work):

```bash
cd <your ComfyUI models/diffusers directory>
git lfs install
git clone https://huggingface.co/AiArtLab/sdxs-1b AiArtLab--sdxs-1b
```

The model directory should contain `unet/`, `text_encoder/`, `vae/`, `tokenizer/`, `scheduler/`, and `pipeline_sdxs.py`.

**StabilityMatrix users:** The diffusers directory is typically at `Data/Models/Diffusers/`. If the model doesn't appear in the node dropdown, try placing it in `Data/Packages/<your ComfyUI package>/models/diffusers/` instead.

### 3. Load the workflow

Drag `sdxs-1b-workflow.json` into ComfyUI. Select `AiArtLab--sdxs-1b` in each loader node's `model_name` dropdown.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: cannot import name 'Qwen3_5ForConditionalGeneration'` | Upgrade transformers: `pip install transformers>=4.52` |
| `ModuleNotFoundError: No module named 'diffusers'` | Install: `pip install diffusers accelerate` |
| Model not appearing in dropdown | Ensure the model is a cloned directory (not a single file) in your diffusers models path |
| VAE decode very slow / hangs | Make sure you're using the latest `__init__.py` (latent size should use `// 16`, not `// 8`) |

## License

Apache 2.0 (same as the model)
