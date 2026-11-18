#!/usr/bin/env python3
"""
setup_vardaan_folders.py - Standardizes Vardaan's 9 recordings into BHU/, LHU/, RHU/
matching the exact structure of Priyanshu, Sanchit, Sir, and Vansh.
Adds activity.txt to each folder.
"""

from pathlib import Path
import shutil
import sys

BASE_DIR = Path("E:/percom")
SOURCE_DIR = BASE_DIR / "Vardaan"

MAPPING = {
    "vardaan_bhu_1": ("BHU", "BHU_S1_Vardaan", "bhu"),
    "Vardaan_bhu_2": ("BHU", "BHU_S2_Vardaan", "bhu"),
    "Vardaan_bhu_3": ("BHU", "BHU_S3_Vardaan", "bhu"),
    "Vardaan_lhu_1": ("LHU", "LHU_S1_Vardaan", "lhu"),
    "Vardaan_lhu_2": ("LHU", "LHU_S2_Vardaan", "lhu"),
    "Vardaan_lhu_3": ("LHU", "LHU_S3_Vardaan", "lhu"),
    "Vardaan_rhu_1": ("RHU", "RHU_S1_Vardaan", "rhu"),
    "Vardaan_rhu_2": ("RHU", "RHU_S2_Vardaan", "rhu"),
    "Vardaan_rhu_3": ("RHU", "RHU_S3_Vardaan", "rhu"),
}

def main():
    print(f"Setting up Vardaan standardized recordings under {BASE_DIR}...")
    assert SOURCE_DIR.exists(), f"Source directory {SOURCE_DIR} missing!"

    for src_name, (act_dir, target_name, gesture) in MAPPING.items():
        src_folder = SOURCE_DIR / src_name
        target_folder = BASE_DIR / act_dir / target_name
        assert src_folder.exists(), f"Source folder {src_folder} missing!"
        
        target_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {src_name} -> {act_dir}/{target_name}")

        for item in src_folder.iterdir():
            target_file = target_folder / item.name
            if not target_file.exists():
                print(f"  Copying {item.name} ({item.stat().st_size / (1024*1024):.1f} MB)...", flush=True)
                shutil.copy2(item, target_file)
            else:
                print(f"  Already exists: {item.name}", flush=True)

        # Write activity.txt
        act_file = target_folder / "activity.txt"
        act_file.write_text(f"{gesture}\n", encoding="utf-8")
        print(f"  Wrote {act_file.name} = '{gesture}'")

    print("\nAll 9 Vardaan recordings successfully standardized into BHU/, LHU/, RHU/!")

if __name__ == "__main__":
    main()
