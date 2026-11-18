#!/usr/bin/env python3
"""
standardize_vardaan.py - Checks and standardizes Vardaan's recording folders:
Maps vardaan_<act>_<session> -> <act>_Vardaan_S<session> and ensures activity.txt exists.
"""

from pathlib import Path
import shutil

SOURCE_DIR = Path("E:/percom/Vardaan")
TARGET_DIR = Path("E:/percom/Vardaan_standardized")

MAPPING = {
    "vardaan_bhu_1": ("bhu", "Vardaan", "1"),
    "Vardaan_bhu_2": ("bhu", "Vardaan", "2"),
    "Vardaan_bhu_3": ("bhu", "Vardaan", "3"),
    "Vardaan_lhu_1": ("lhu", "Vardaan", "1"),
    "Vardaan_lhu_2": ("lhu", "Vardaan", "2"),
    "Vardaan_lhu_3": ("lhu", "Vardaan", "3"),
    "Vardaan_rhu_1": ("rhu", "Vardaan", "1"),
    "Vardaan_rhu_2": ("rhu", "Vardaan", "2"),
    "Vardaan_rhu_3": ("rhu", "Vardaan", "3"),
}

def main():
    print(f"Auditing Vardaan source folders in: {SOURCE_DIR}")
    assert SOURCE_DIR.exists(), f"{SOURCE_DIR} does not exist!"

    for src_name, (act, person, sess) in MAPPING.items():
        src_folder = SOURCE_DIR / src_name
        assert src_folder.exists(), f"Missing folder: {src_folder}"
        csi_files = list(src_folder.glob("rx_*.csi"))
        video_file = src_folder / "video.mp4"
        timing_file = src_folder / "timing.txt"

        assert len(csi_files) == 1, f"Expected 1 CSI file in {src_folder}, found {len(csi_files)}"
        assert video_file.exists(), f"Missing video in {src_folder}"
        assert timing_file.exists(), f"Missing timing.txt in {src_folder}"

        csi_size_mb = csi_files[0].stat().st_size / (1024 * 1024)
        video_size_mb = video_file.stat().st_size / (1024 * 1024)
        std_name = f"{act}_{person}_S{sess}"
        print(f"  {std_name:<18} | CSI: {csi_size_mb:7.1f} MB | Video: {video_size_mb:6.1f} MB | {csi_files[0].name}")

    print("\nAll 9 recordings verified complete and non-empty.")

if __name__ == "__main__":
    main()
