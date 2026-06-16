# ComfyUI-trellis2-apple

TRELLIS.2 image-to-3D for ComfyUI using the MLX backend — runs natively on Apple Silicon (M1–M4).

## Install

```bash
cd custom_nodes
git clone --recursive https://github.com/LeonardMeagher2/ComfyUI-trellis2-apple.git
cd ComfyUI-trellis2-apple
source ../venv/bin/activate    # or your ComfyUI venv
bash setup.sh
```

ComfyUI creates its venv at `venv/` or `.venv/` inside its install directory.

Restart ComfyUI. Model weights download on first run.

## Patch gated models (optional)

Some model repos require HF authentication. Run this script to swap them for public equivalents:

```bash
source ../venv/bin/activate
python patch_models.py
```

A backup is saved as `pipeline.json.bak`. To restore, copy it back.
