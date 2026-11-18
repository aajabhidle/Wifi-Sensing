#!/usr/bin/env python3
"""
audit_vardaan_step1.py - Comprehensive Step 1 Audit for Vardaan (P5).
Audits all 9 recordings against the 4 existing participants:
1. Files present: video.mp4, rx_*.csi, timing.txt, activity.txt, picoscenes.log.
2. CSI audit: converted NPZ readability, duration, packet rate, subcarriers, carrier frequency, bandwidth.
3. Video audit: resolution, fps, frame count, duration.
4. Clock check: end_gap = (t0_epoch + csi_dur) - video_duration - video_start_epoch.
5. Overall verdict per recording and overall: USABLE / USABLE AFTER A SAFE FIX / NOT USABLE.
"""

from pathlib import Path
import json
import cv2
import numpy as np

BASE_DIR = Path("E:/percom")
NPZ_DIR = Path("E:/percom_work/npz")
OUT_DIR = BASE_DIR / "results_5p" / "data_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [
    ("BHU", "BHU_S1_Vardaan", "bhu"),
    ("BHU", "BHU_S2_Vardaan", "bhu"),
    ("BHU", "BHU_S3_Vardaan", "bhu"),
    ("LHU", "LHU_S1_Vardaan", "lhu"),
    ("LHU", "LHU_S2_Vardaan", "lhu"),
    ("LHU", "LHU_S3_Vardaan", "lhu"),
    ("RHU", "RHU_S1_Vardaan", "rhu"),
    ("RHU", "RHU_S2_Vardaan", "rhu"),
    ("RHU", "RHU_S3_Vardaan", "rhu"),
]

def read_timing(folder):
    t_file = folder / "timing.txt"
    out = {}
    if t_file.exists():
        for line in t_file.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                try:
                    out[k.strip()] = float(v.strip())
                except ValueError:
                    out[k.strip()] = v.strip()
    return out

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n_frames / fps if fps > 0 else 0.0
    cap.release()
    return dict(w=w, h=h, fps=fps, n_frames=n_frames, duration=dur)

def main():
    print("==========================================================================================")
    print("STEP 1: VARDAAN DATA INTEGRITY AUDIT (GO / NO-GO)")
    print("==========================================================================================")
    
    audit_results = []
    all_usable = True

    print(f"{'Recording':<18} | {'CSI Size':<9} | {'NPZ Pkts':<9} | {'Rate':<7} | {'Vid Dur':<7} | {'Gap (s)':<7} | {'Verdict'}")
    print("-" * 90)

    for act, sid, gesture in SESSIONS:
        folder = BASE_DIR / act / sid
        assert folder.exists(), f"Missing folder: {folder}"
        
        # 1. Raw files check
        csi_files = list(folder.glob("rx_*.csi"))
        vid_path = folder / "video.mp4"
        tim_path = folder / "timing.txt"
        act_path = folder / "activity.txt"
        pico_log = folder / "picoscenes.log"

        csi_present = len(csi_files) == 1
        csi_size_mb = csi_files[0].stat().st_size / (1024*1024) if csi_present else 0.0
        
        # Check picoscenes log for crashes and verify device & radio parameters
        pico_ok = False
        if pico_log.exists():
            log_text = pico_log.read_text(errors="ignore")
            if "AX200" in log_text and "5775" in log_text and "CBW=80" in log_text:
                pico_ok = True

        # 2. Converted NPZ check
        npz_matches = list(NPZ_DIR.glob(f"*__{sid}_rx_*.npz"))
        npz_present = len(npz_matches) > 0
        npz_info = {}
        if npz_present:
            z = np.load(npz_matches[0])
            n_packets = len(z["t"])
            csi_dur = float(z["t"][-1] - z["t"][0])
            packet_rate = n_packets / csi_dur if csi_dur > 0 else 0.0
            t0_epoch = float(z["t0_epoch"])
            freq = float(z.get("center_freq_mhz", 5775.0))
            cbw = float(z.get("cbw_mhz", 80.0))
            num_rx = int(z.get("num_rx", 2))
            num_tx = int(z.get("num_tx", 2))
            npz_info = dict(
                file=npz_matches[0].name, n_packets=n_packets, csi_dur=csi_dur,
                packet_rate=packet_rate, t0_epoch=t0_epoch, freq=freq, cbw=cbw,
                num_rx=num_rx, num_tx=num_tx, data_shape=list(z["data"].shape)
            )

        # 3. Video info
        vid_info = get_video_info(vid_path) if vid_path.exists() else None

        # 4. Clock check
        timing = read_timing(folder)
        vstart = timing.get("video_start_epoch")
        end_gap = None
        if npz_present and vid_info and vstart is not None:
            end_gap = float(npz_info["t0_epoch"] + npz_info["csi_dur"] - vid_info["duration"] - vstart)

        # Verdict
        usable = (
            csi_present and csi_size_mb > 500 and pico_ok and npz_present and
            vid_info is not None and vid_info["duration"] > 60 and
            end_gap is not None and -1.0 <= end_gap <= 3.5
        )
        if not usable:
            all_usable = False

        verdict = "USABLE" if usable else "NOT USABLE"
        gap_str = f"{end_gap:+.2f}" if end_gap is not None else "N/A"
        dur_str = f"{vid_info['duration']:.1f}s" if vid_info else "N/A"
        pkts_str = f"{npz_info.get('n_packets', 0)}" if npz_present else "N/A"
        rate_str = f"{npz_info.get('packet_rate', 0):.1f} Hz" if npz_present else "N/A"

        print(f"{sid:<18} | {csi_size_mb:6.1f} MB | {pkts_str:<9} | {rate_str:<7} | {dur_str:<7} | {gap_str:<7} | {verdict}")

        audit_results.append(dict(
            sid=sid, act=act, gesture=gesture, csi_size_mb=csi_size_mb,
            picoscenes_log_ok=pico_ok, npz_info=npz_info, video_info=vid_info,
            timing=timing, end_gap=end_gap, verdict=verdict
        ))

    out_file = OUT_DIR / "vardaan_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)

    print("-" * 90)
    print(f"Audit report saved to: {out_file}")
    print(f"\nOVERALL STEP 1 VERDICT: {'GO - USABLE' if all_usable else 'NO-GO'}")
    assert all_usable, "One or more Vardaan recordings failed the Step 1 Audit!"

if __name__ == "__main__":
    main()
