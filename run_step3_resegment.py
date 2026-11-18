#!/usr/bin/env python3
"""
run_step3_resegment.py - Executes Step 3 of the BTP Protocol:
1. Sanchit Reproducibility Check against baseline segments_mp.json.
2. Uniform MediaPipe segmentation across all 5 participants (P1-P5).
3. Produces diagnostics table and unified segments_5p.json.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add local paths
ROOT_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ROOT_DIR / "siwis", ROOT_DIR / "Codebase" / "percom"]:
    if str(p) not in sys.path and p.exists():
        sys.path.insert(0, str(p))

import percom_common as pc
import video_segments as vs

EXPECTED_PERSONS = ["Priyanshu", "Sanchit", "Sir", "Vansh", "Vardaan"]

def main():
    ap = argparse.ArgumentParser(description="Step 3: Re-segment all 5 participants with MediaPipe pipeline")
    ap.add_argument("--video_dir", default="~/percom/video_5p", help="Directory holding BHU/, LHU/, RHU/ for all 5 participants")
    ap.add_argument("--data", default="~/percom/siwis_data_5p", help="siwis_data_5p directory containing rec/")
    ap.add_argument("--old_segments", default="~/percom/siwis_data/segments_mp.json", help="Existing Sanchit segments for check")
    ap.add_argument("--out", default="~/percom/siwis_data_5p/segments_5p.json", help="Output unified segments file")
    ap.add_argument("--plots", default="~/percom/results_5p/segments/plots", help="Directory to save segment signal plots")
    ap.add_argument("--model", default="~/percom/hand_landmarker.task", help="Path to hand_landmarker.task")
    args = ap.parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    data = Path(args.data).expanduser().resolve()
    old_seg_path = Path(args.old_segments).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    plot_dir = Path(args.plots).expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("==========================================================================================")
    print("STEP 3: RE-SEGMENT ALL PARTICIPANTS (MEDIAPIPE PIPELINE)")
    print("==========================================================================================")
    print(f"Videos Directory: {video_dir}")
    print(f"Data Directory  : {data}")
    print(f"Target Output   : {out_path}")
    print(f"Plot Directory  : {plot_dir}")

    # Import mp_cut_and_run functionality
    import mp_cut_and_run as mp_run

    recs = pc.find_recordings(video_dir, acts=("BHU", "LHU", "RHU"), persons=EXPECTED_PERSONS)
    print(f"\nFound {len(recs)} total recordings across {sorted({r['person'] for r in recs})}")
    assert len(recs) >= 44, f"Expected at least 44 recordings, found {len(recs)}"

    # 1. Reproducibility Check on Sanchit
    sanchit_recs = [r for r in recs if r["person"] == "Sanchit"]
    print(f"\n--- 1. SANCHIT REPRODUCIBILITY CHECK ({len(sanchit_recs)} recordings) ---")
    
    old_segs = {}
    if old_seg_path.exists():
        old_segs = json.loads(old_seg_path.read_text())
        print(f"Loaded reference segments from {old_seg_path}")
    else:
        alt_path = Path("~/percom/siwis_data/segments.json").expanduser()
        if alt_path.exists():
            old_segs = json.loads(alt_path.read_text())
            print(f"Loaded fallback reference segments from {alt_path}")

    # Initialize Hands detector
    det = mp_run.Hands("tasks", args.model, 0.5, 0.5, 1, ROOT_DIR)
    print(f"MediaPipe HandLandmarker initialized successfully.")

    sig_dir = data / "mp_signal"
    sig_dir.mkdir(parents=True, exist_ok=True)

    unified_segments = {}
    diagnostics = []

    for r in recs:
        sid = r["sid"]
        vid = r["folder"] / "video.mp4"
        meta_p = data / "rec" / sid / "meta.json"
        ts_p = data / "rec" / sid / "frame_timestamps.npy"
        
        if not vid.exists() or not meta_p.exists() or not ts_p.exists():
            print(f"  [SKIP] {sid}: video or prepared pose labels missing")
            continue

        meta = json.loads(meta_p.read_text())
        ts = np.load(ts_p)
        label_fps = float(meta.get("label_fps", 2.0))
        timing = pc.read_timing(r["folder"])
        vstart = timing.get("video_start_epoch")

        cache = sig_dir / f"{sid}.npz"
        if cache.exists():
            z = np.load(cache)
            sig, lm, fps = z["sig"], z["lm"], float(z["fps"])
        else:
            sig, lm, fps = mp_run.run_video(vid, det, 10.0)
            np.savez_compressed(cache, sig=sig, lm=lm, fps=fps)

        found_frac = float(np.mean(~np.isnan(sig).all(1)))
        curve = mp_run.fill_curve(sig, fps, vs)
        if curve is None:
            print(f"  [FAIL] {sid}: Hand detection failed (< 5s detected)")
            continue

        res = vs.segment_signal(curve, fps, smooth=0.2, min_period=1.5, prom=0.35, rest_frac=0.30)
        t_off = float(vstart) + float(meta.get("sync_shift_s", 0.0)) if vstart is not None else None
        frames = vs.to_frames(res["segments"], ts - t_off) if t_off is not None else []

        unified_segments[sid] = dict(
            person=meta["person"], act=meta["activity"], n_frames=int(len(ts)),
            fps=label_fps, method="mediapipe", segments=frames,
            up_runs=[[a, b] for a, b in frames]
        )

        durations = np.array([(b - a) / label_fps for a, b in frames]) if frames else np.zeros(1)
        med_dur = float(np.median(durations))
        iqr_dur = float(np.percentile(durations, 75) - np.percentile(durations, 25)) if len(durations) > 1 else 0.0
        vid_dur = float(len(ts)) / label_fps
        segs_per_min = (len(frames) / vid_dur) * 60.0 if vid_dur > 0 else 0.0

        diag_entry = dict(
            sid=sid, person=meta["person"], act=meta["activity"],
            found_pct=round(found_frac * 100, 1),
            n_segs=len(frames),
            med_dur=round(med_dur, 2),
            iqr_dur=round(iqr_dur, 2),
            segs_per_min=round(segs_per_min, 1),
            vid_dur=round(vid_dur, 1)
        )
        diagnostics.append(diag_entry)

        # Plot for visual confirmation
        vs.plot(plot_dir / f"{sid}.png", sid, curve, fps, res, f"[{len(frames)} segs, med {med_dur:.1f}s]")

        # If Sanchit, check reproducibility
        if meta["person"] == "Sanchit" and sid in old_segs:
            ref_n = len(old_segs[sid]["segments"])
            diff = abs(len(frames) - ref_n)
            print(f"  Sanchit check {sid:<16}: New = {len(frames):2d} segs | Ref = {ref_n:2d} segs | Diff = {diff}")
            assert diff <= 2, f"Sanchit reproducibility check failed on {sid}: diff {diff} exceeds tolerance!"

    # Save unified segments
    out_path.write_text(json.dumps(unified_segments, indent=1))
    print(f"\n[SUCCESS] Successfully saved unified segments for {len(unified_segments)} recordings to:")
    print(f"  {out_path}")

    # Diagnostics Summary Table
    print("\n==========================================================================================")
    print("STEP 3: DIAGNOSTICS TABLE (ALL 5 PARTICIPANTS)")
    print("==========================================================================================")
    print(f"{'Recording':<18} | {'Person':<10} | {'Act':<4} | {'Hand%':<6} | {'Segs':<5} | {'MedDur(s)':<9} | {'Segs/min':<8} | {'VidDur(s)'}")
    print("-" * 88)
    for d in diagnostics:
        flag = " (!)" if d["n_segs"] < 35 or d["n_segs"] > 60 else ""
        print(f"{d['sid']:<18} | {d['person']:<10} | {d['act']:<4} | {d['found_pct']:5.1f}% | {d['n_segs']:4d}{flag} | {d['med_dur']:7.2f}s  | {d['segs_per_min']:7.1f}  | {d['vid_dur']:6.1f}s")
    print("-" * 88)

if __name__ == "__main__":
    main()
