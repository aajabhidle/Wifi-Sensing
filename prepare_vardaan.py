#!/usr/bin/env python3
"""
prepare_vardaan.py - Runs Step 1-4 (Torchvision Mask R-CNN + Keypoint R-CNN) on Vardaan's
9 standardized recordings to produce siwis_data_5p/rec/{bhu,lhu,rhu}_Vardaan_S{1,2,3}.
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ROOT_DIR / "siwis", ROOT_DIR / "Codebase" / "percom", ROOT_DIR / "siwis_percom"]:
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import percom_common as pc
import prepare_percom

def main():
    ap = argparse.ArgumentParser(description="Generate SiWiS labels and features for Vardaan (P5)")
    ap.add_argument("--src", default="~/percom", help="Folder containing BHU/, LHU/, RHU/")
    ap.add_argument("--out", default="~/percom/siwis_data_5p", help="Output dataset root")
    ap.add_argument("--npz_dir", default="~/percom/work/npz_5p", help="Folder containing converted .npz files")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    npz_dir = Path(args.npz_dir).expanduser().resolve()

    print("=== Preparing Pose & CSI Labels for Vardaan (P5) ===")
    print(f"Source recordings : {src}")
    print(f"NPZ directory     : {npz_dir}")
    print(f"Output directory  : {out}")
    print(f"Device            : {args.device}")

    # Prepare command args for prepare_percom.main()
    sys.argv = [
        "prepare_percom.py",
        "--src", str(src),
        "--out", str(out),
        "--npz_dir", str(npz_dir),
        "--persons", "Vardaan",  # Strictly Vardaan only
        "--label_fps", "2.0",
        "--backend", "torchvision",
        "--device", args.device,
    ]
    prepare_percom.main()

if __name__ == "__main__":
    main()
