import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import folder_paths

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMODULE_DIR = os.path.join(PACKAGE_DIR, "trellis2-apple")
sys.path.insert(0, SUBMODULE_DIR)

from mlx_backend.pipeline import create_mlx_pipeline, to_glb

WEIGHTS_DIR = os.path.join(PACKAGE_DIR, "weights")
MODEL_REPO = "microsoft/TRELLIS.2-4B"
MODEL_DIR = os.path.join(WEIGHTS_DIR, MODEL_REPO.replace("/", "--"))

filename_prefix = "trellis2"

PIPELINE_TYPE_MAP = {
    "Fast (512px)": "512",
    "High Quality (1024px)": "1024",
    "Refined (1024px)": "1024_cascade",
}


def _get_output_path() -> str:
    return folder_paths.get_output_directory()


def _next_output_path(prefix: str, extension: str = ".glb") -> Path:
    base_dir = Path(_get_output_path())
    stem = f"{prefix}"
    counter = 0
    while True:
        p = base_dir / f"{stem}_{counter}{extension}" if counter else base_dir / f"{stem}{extension}"
        if not p.exists():
            return p
        counter += 1


def _download_weights():
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    from huggingface_hub import snapshot_download
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    print(f"Downloading {MODEL_REPO} to {MODEL_DIR}...")
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    return MODEL_DIR


def _pick_device():
    try:
        import mlx.core as mx
        if mx.metal.is_available():
            return "mlx"
    except Exception:
        pass
    return "cpu"


class Trellis2ShapeNode:
    CATEGORY = "TRELLIS.2"
    FUNCTION = "generate_shape"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pipeline_type": (
                    ["Fast (512px)", "High Quality (1024px)", "Refined (1024px)"],
                    {"default": "Fast (512px)"},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999999}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "texture_size": (
                    "INT",
                    {"default": 1024, "min": 1, "max": 2048, "step": 1},
                ),
                "use_rembg": ("BOOLEAN", {"default": False}),
                "cpu_voxelize": ("BOOLEAN", {"default": False}),
                "resolution": (
                    "INT",
                    {"default": 200000, "min": 1000, "max": 1000000, "step": 1000},
                ),
                "remesh": ("BOOLEAN", {"default": True}),
            },
        }

    def generate_shape(
        self,
        image,
        pipeline_type,
        seed,
        steps,
        texture_size,
        use_rembg,
        cpu_voxelize,
        resolution,
        remesh,
    ):
        import gc
        if image.ndim == 4:
            image = image[0]
        img_array = image.cpu().numpy()
        if img_array.dtype != np.uint8:
            img_array = (img_array.clip(0.0, 1.0) * 255.0).round().astype("uint8")
        pil_image = Image.fromarray(img_array, mode="RGBA" if img_array.shape[-1] == 4 else "RGB")

        # Ensure weights are downloaded
        weights_path = _download_weights()

        # Build pipeline (always fresh to free memory after run)
        if not use_rembg:
            import trellis2.pipelines.rembg as _rembg_pkg
            class _NoopRembg:
                def __init__(self, *args, **kwargs): pass
            _orig_bi = _rembg_pkg.BiRefNet
            _rembg_pkg.BiRefNet = _NoopRembg
            try:
                pipeline = create_mlx_pipeline(weights_path)
            finally:
                _rembg_pkg.BiRefNet = _orig_bi
            pipeline.rembg_model = None
        else:
            pipeline = create_mlx_pipeline(weights_path)
            pipeline.rembg_model.model = pipeline.rembg_model.model.float()

        # Generate
        torch.manual_seed(seed)
        out_mesh = pipeline.run(
            image=pil_image,
            seed=seed,
            sparse_structure_sampler_params={"steps": steps},
            shape_slat_sampler_params={"steps": steps},
            tex_slat_sampler_params={"steps": steps},
            pipeline_type=PIPELINE_TYPE_MAP[pipeline_type],
            preprocess_image=use_rembg,
        )

        mesh = out_mesh[0] if isinstance(out_mesh, list) else out_mesh
        verts = mesh.vertices
        faces = mesh.faces
        print(f"Mesh: {len(verts):,} verts, {len(faces):,} faces")

        # Free MLX Metal cache before GPU‑hungry o‑voxel baking
        import mlx.core as mx
        mx.metal.clear_cache()
        mx.metal.set_cache_limit(256 * 1024 ** 2)  # 256 MB

        glb_path = _next_output_path(filename_prefix, extension=".glb")
        glb_path.parent.mkdir(parents=True, exist_ok=True)

        if cpu_voxelize:
            import o_voxel.postprocess_cpu as _ov_cpu
            _orig_get_device = _ov_cpu._get_device
            _ov_cpu._get_device = lambda: torch.device('cpu')
            print("o-voxel forced to CPU")

        try:
            to_glb(mesh, str(glb_path), texture_size=texture_size,
                   decimation_target=resolution, remesh=remesh)
        finally:
            if cpu_voxelize:
                _ov_cpu._get_device = _orig_get_device
            # Free all MLX / pipeline memory
            del pipeline
            del mesh
            mx.metal.clear_cache()
            gc.collect()
            print("Pipeline memory freed")

        return (str(glb_path),)


NODE_CLASS_MAPPINGS = {
    "Trellis2Shape": Trellis2ShapeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Shape": "TRELLIS.2 Image to 3D (MLX)",
}
