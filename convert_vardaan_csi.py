#!/usr/bin/env python3
"""
convert_vardaan_csi.py - Converts Vardaan's 9 standardized recordings into .npz files.
Can be executed in any environment where `picoscenes` is installed (laptop Ubuntu or server).
"""

import sys
from pathlib import Path
import subprocess

def main():
    try:
        import picoscenes
        print("[OK] PicoScenes library is available.")
    except ImportError:
        print("[ERROR] 'picoscenes' module not found.")
        print("Please activate your PicoScenes virtual environment before running:")
        print("  source ~/PicoScenes-Python-Toolbox/.venv/bin/activate (or your picoscenes venv)")
        sys.exit(1)

    repo_dir = Path(__file__).resolve().parent
    csi_script = repo_dir / "csi_to_npz.py"
    if not csi_script.exists():
        csi_script = Path.home() / "percom" / "csi_to_npz.py"
    assert csi_script.exists(), f"Cannot find csi_to_npz.py at {csi_script}"

    out_dir = repo_dir / "work" / "npz_vardaan"
    out_dir.mkdir(parents=True, exist_ok=True)

    vardaan_folders = []
    for act in ["BHU", "LHU", "RHU"]:
        for sess in ["S1", "S2", "S3"]:
            folder = repo_dir / act / f"{act}_{sess}_Vardaan"
            if not folder.exists():
                folder = Path.home() / "percom" / act / f"{act}_{sess}_Vardaan"
            if folder.exists():
                vardaan_folders.append(folder)
            else:
                print(f"Warning: folder not found: {folder}")

    assert len(vardaan_folders) == 9, f"Expected 9 Vardaan folders, found {len(vardaan_folders)}"
    print(f"Found 9 Vardaan folders. Converting CSI -> NPZ into {out_dir} ...")

    cmd = [
        sys.executable,
        str(csi_script),
        *[str(f) for f in vardaan_folders],
        "--out", str(out_dir),
        "--rate", "150.0"
    ]
    print("Running command:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode == 0:
        npz_files = list(out_dir.glob("*.npz"))
        print(f"\n[SUCCESS] Successfully generated {len(npz_files)} NPZ files in {out_dir}:")
        for f in sorted(npz_files):
            print(f"  {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"\n[FAILED] Conversion failed with return code {res.returncode}")

if __name__ == "__main__":
    main()
