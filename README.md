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

Edit `weights/microsoft--TRELLIS.2-4B/pipeline.json` after first run. Replace the two gated model repos:

- `model_name` → `kryveil/dinov3-vitl16-pretrain-lvd1689m`
- rembg `model_name` → `ZhengPeng7/BiRefNet`

ComfyUI creates its venv at `venv/` or `.venv/` inside its install directory. Activate it before running `setup.sh` so packages install to the right place.

Restart ComfyUI.
