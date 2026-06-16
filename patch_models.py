#!/usr/bin/env python3
"""Patch pipeline.json to swap gated model repos for public equivalents.

Creates pipeline.json.bak before modifying. Run once after weights download.

Usage:
    cd ComfyUI-trellis2-apple
    source ../venv/bin/activate
    python patch_models.py
"""

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
PIPELINE = REPO_ROOT / "weights" / "microsoft--TRELLIS.2-4B" / "pipeline.json"

SWAPS = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": "kryveil/dinov3-vitl16-pretrain-lvd1689m",
    "briaai/RMBG-2.0": "ZhengPeng7/BiRefNet",
}


def main():
    if not PIPELINE.exists():
        print(f"Not found: {PIPELINE}")
        print("Run ComfyUI once to download weights first.")
        sys.exit(1)

    # Backup
    backup = PIPELINE.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(PIPELINE, backup)
        print(f"Backup saved: {backup}")

    with open(PIPELINE) as f:
        data = json.load(f)

    cfg = data.get("args", data)
    patched = 0

    for section in ("image_cond_model", "rembg_model"):
        entry = cfg.get(section, {})
        name = entry.get("args", {}).get("model_name", "")
        if name in SWAPS:
            entry["args"]["model_name"] = SWAPS[name]
            patched += 1
            print(f"  {name}  ->  {SWAPS[name]}")

    if patched:
        with open(PIPELINE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Patched {patched} model(s). Backups at: {backup}")
        print("To restore: cp {0} {1}".format(backup, PIPELINE))
    else:
        print("No gated models found — nothing to patch.")


if __name__ == "__main__":
    main()
