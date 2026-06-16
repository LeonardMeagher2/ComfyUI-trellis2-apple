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

ComfyUI creates its venv at `venv/` or `.venv/` inside its install directory. Activate it before running `setup.sh` so packages install to the right place.

Restart ComfyUI. Model weights download on first use.
