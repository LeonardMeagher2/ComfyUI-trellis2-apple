#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "=== ComfyUI-trellis2-apple Setup ==="
echo

PIP="python3 -m pip install"

# Core MLX (Apple Silicon ML framework)
echo "Installing MLX..."
$PIP mlx

# o-voxel (mesh extraction & texture baking)
echo "Installing o-voxel..."
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
$PIP setuptools wheel pybind11
$PIP --no-build-isolation "trellis2-apple/o-voxel"

# Python deps
$PIP fast-simplification trimesh

echo
echo "Done. Restart ComfyUI."
