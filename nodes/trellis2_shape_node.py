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


def _preprocess_image(image):
    if image.ndim == 4:
        image = image[0]
    img_array = image.cpu().numpy()
    if img_array.dtype != np.uint8:
        img_array = (img_array.clip(0.0, 1.0) * 255.0).round().astype("uint8")
    return Image.fromarray(img_array, mode="RGBA" if img_array.shape[-1] == 4 else "RGB")


def _create_pipeline(use_rembg, weights_path):
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
    return pipeline


def _run_pipeline(pipeline, pil_image, seed, steps, pipeline_type, use_rembg):
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
    print(f"Mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
    return mesh


def _fix_mesh(mesh):
    import trimesh
    v = mesh.vertices.detach().cpu().numpy()
    f = mesh.faces.detach().cpu().numpy()
    t = trimesh.Trimesh(vertices=v, faces=f)
    t.remove_unreferenced_vertices()
    t.merge_vertices()
    trimesh.repair.fill_holes(t)
    trimesh.repair.fix_winding(t)
    trimesh.repair.fill_holes(t, use_fan=True)
    trimesh.repair.fill_holes(t)
    trimesh.repair.fix_normals(t, multibody=True)
    device = mesh.vertices.device
    mesh.vertices = torch.from_numpy(t.vertices).float().to(device)
    mesh.faces = torch.from_numpy(t.faces).int().to(device)
    print(f"After fix: {len(t.vertices):,} verts, {len(t.faces):,} faces")
    return mesh


def _inpaint_query_attrs(mesh, vertices):
    """Sample vertex attrs with distance-transform inpainting of empty voxels."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    import torch.nn.functional as F

    C = mesh.attrs.shape[-1]
    D, H, W = mesh.voxel_shape[2:]

    dense = torch.zeros(1, C, D, H, W, dtype=mesh.attrs.dtype, device=mesh.attrs.device)
    mask = torch.zeros(1, 1, D, H, W, dtype=torch.bool, device=mesh.attrs.device)
    cx, cy, cz = mesh.coords[:, 0].long(), mesh.coords[:, 1].long(), mesh.coords[:, 2].long()
    dense[0, :, cx, cy, cz] = mesh.attrs.T
    mask[0, 0, cx, cy, cz] = True

    if not mask.all():
        m = mask[0, 0].cpu().numpy().astype(bool)
        vol_np = dense[0].cpu().numpy()
        _, indices = distance_transform_edt(~m, return_indices=True)
        nz, ny, nx = indices
        empty = ~m
        for c in range(C):
            vol_np[c][empty] = vol_np[c][nz[empty], ny[empty], nx[empty]]
        dense[0] = torch.from_numpy(vol_np).to(mesh.attrs.device, dtype=mesh.attrs.dtype)

    grid_pts = ((vertices - mesh.origin) / mesh.voxel_size)
    grid_pts_norm = torch.stack([
        grid_pts[:, 2] / (W - 1) * 2 - 1,
        grid_pts[:, 1] / (H - 1) * 2 - 1,
        grid_pts[:, 0] / (D - 1) * 2 - 1,
    ], dim=-1).reshape(1, 1, 1, -1, 3)
    sampled = F.grid_sample(dense, grid_pts_norm, mode='bilinear', align_corners=True, padding_mode='border')
    return sampled.reshape(C, -1).T


def _cleanup():
    import gc
    import mlx.core as mx
    mx.metal.clear_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    print("Pipeline memory freed")


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
                "fix_mesh": ("BOOLEAN", {"default": True}),
                "resolution": (
                    "INT",
                    {"default": 200000, "min": 1000, "max": 1000000, "step": 1000},
                ),
                "remesh": ("BOOLEAN", {"default": True}),
            },
        }

    def generate_shape(self, image, pipeline_type, seed, steps, texture_size, use_rembg, cpu_voxelize, fix_mesh, resolution, remesh):
        pipeline = None
        mesh = None
        try:
            weights_path = _download_weights()
            pil_image = _preprocess_image(image)
            pipeline = _create_pipeline(use_rembg, weights_path)
            mesh = _run_pipeline(pipeline, pil_image, seed, steps, pipeline_type, use_rembg)

            if fix_mesh:
                mesh = _fix_mesh(mesh)

            import mlx.core as mx
            mx.metal.clear_cache()
            mx.metal.set_cache_limit(256 * 1024 ** 2)

            glb_path = _next_output_path("trellis2", extension=".glb")
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
        finally:
            del pipeline, mesh
            _cleanup()
        return (str(glb_path),)


class Trellis2ShapeFastNode:
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
                "use_rembg": ("BOOLEAN", {"default": False}),
                "cpu_voxelize": ("BOOLEAN", {"default": False}),
                "fix_mesh": ("BOOLEAN", {"default": True}),
            },
        }

    def generate_shape(self, image, pipeline_type, seed, steps, use_rembg, cpu_voxelize, fix_mesh):
        pipeline = None
        mesh = None
        try:
            weights_path = _download_weights()
            pil_image = _preprocess_image(image)
            pipeline = _create_pipeline(use_rembg, weights_path)
            mesh = _run_pipeline(pipeline, pil_image, seed, steps, pipeline_type, use_rembg)

            if fix_mesh:
                mesh = _fix_mesh(mesh)

            import mlx.core as mx
            mx.metal.clear_cache()

            glb_path = _next_output_path("trellis2_fast", extension=".glb")
            glb_path.parent.mkdir(parents=True, exist_ok=True)

            attrs = _inpaint_query_attrs(mesh, mesh.vertices)
            colors = attrs[:, :3].clamp(0, 1).cpu().numpy()
            colors = (colors * 255).astype(np.uint8)

            v = mesh.vertices.detach().cpu().numpy()
            # Z-up (TRELLIS) → Y-up (GLB standard): rotate -90° around X
            rotated = np.empty_like(v)
            rotated[:, 0] = v[:, 0]
            rotated[:, 1] = v[:, 2]
            rotated[:, 2] = -v[:, 1]

            import trimesh
            t = trimesh.Trimesh(
                vertices=rotated,
                faces=mesh.faces.detach().cpu().numpy(),
                vertex_colors=colors,
            )
            trimesh.repair.fix_normals(t, multibody=True)
            trimesh.repair.fix_inversion(t, multibody=True)
            t.export(str(glb_path))
            print(f"Exported vertex-color GLB: {glb_path}")
            del attrs, colors, v, rotated, t
        finally:
            del pipeline, mesh
            _cleanup()
        return (str(glb_path),)


NODE_CLASS_MAPPINGS = {
    "Trellis2Shape": Trellis2ShapeNode,
    "Trellis2ShapeFast": Trellis2ShapeFastNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Trellis2Shape": "TRELLIS.2 Image to 3D (MLX)",
    "Trellis2ShapeFast": "TRELLIS.2 Fast (Vertex Color) (MLX)",
}
