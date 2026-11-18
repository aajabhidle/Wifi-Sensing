#!/usr/bin/env python3
"""
run_protocols_5p.py - 5-Participant Wi-Fi Gesture Protocol Runner.

Implements Protocols A, B, and C with Complete Gestures Only (--unit gestures):
  - Protocol A: Participant-wise (5 folds x 5 persons = 25 units)
  - Protocol B: Pooled (5 folds = 5 units)
  - Protocol C: Leave-One-Person-Out (5 folds = 5 units)

Key Requirements:
  - 5 Contiguous Block-Folds per recording (near equal size, diff <= 1).
  - Strict leakage guard: drop neighbouring segment on each side of test block.
  - Assert no 1.7s CSI window overlap between train and test frames.
  - 4 inner contiguous block folds for out-of-fold pose features during training.
  - Per-recording mean pose subtraction from TRAINING frames only.
  - SNN: seeds 0, 1, 2; inputs: kp (joints), mask (outline), both, each + speed.
  - Atomic worker queue via directory claims (mkdir) with graceful SIGTERM and checkpointing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
for p in [
    REPO_ROOT,
    REPO_ROOT / "siwis",
    REPO_ROOT / "Codebase" / "percom",
    REPO_ROOT / "Codebase" / "percom" / "segment_kit",
    REPO_ROOT / "segment_kit",
    REPO_ROOT / "siwis_percom",
]:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
import percom_common as pc
import pose_snn as ps
import train_percom as tp
import train_percom_v2 as v2
import scenario2_pooled_pipeline as s2

ACTS = ["bhu", "lhu", "rhu"]
DISPLAY_ACTS = ["BHR", "LHR", "RHR"]
ALL_PERSONS = ["Priyanshu", "Sanchit", "Sir", "Vansh", "Vardaan"]

# -----------------------------------------------------------------------------
# Block Fold Splitting & Leakage Verification
# -----------------------------------------------------------------------------
def get_block_folds(n_segments: int, n_folds: int = 5) -> List[Dict[str, List[int]]]:
    """
    Split n_segments in contiguous time blocks of near-equal size (diff <= 1).
    For fold f:
      test = block f
      guard = segment immediately before and immediately after block f (if within recording)
      train = all remaining segments
    """
    if n_segments < n_folds:
        # Fallback if recording has fewer segments than folds
        blocks = [list(b) for b in np.array_split(np.arange(n_segments), max(1, n_segments))]
        while len(blocks) < n_folds:
            blocks.append([])
    else:
        blocks = [list(b) for b in np.array_split(np.arange(n_segments), n_folds)]

    folds = []
    for f in range(n_folds):
        test_idx = blocks[f]
        if not test_idx:
            folds.append(dict(train=[], guard=[], test=[]))
            continue
        min_test, max_test = min(test_idx), max(test_idx)
        guard_idx = set()
        if min_test > 0:
            guard_idx.add(min_test - 1)
        if max_test < n_segments - 1:
            guard_idx.add(max_test + 1)
        
        train_idx = [i for i in range(n_segments) if i not in test_idx and i not in guard_idx]
        folds.append(dict(train=train_idx, guard=sorted(list(guard_idx)), test=test_idx))
    return folds


def verify_fold_leakage(recs: List[Dict[str, Any]], sig_dur: float = 1.7):
    """
    Assert that no test frame's 1.7s CSI window touches any training frame.
    Widens guard if needed.
    """
    for r in recs:
        ts = r["ts"]
        segs = r["seg"]
        sp = r["split"]
        if not sp["test"] or not sp["train"]:
            continue
        test_frames = s2.frames_of_segments(segs, sp["test"])
        train_frames = s2.frames_of_segments(segs, sp["train"])
        if not test_frames or not train_frames:
            continue
        
        t_test_min = float(ts[min(test_frames)])
        t_test_max = float(ts[max(test_frames)])
        
        train_pre = [t for t in train_frames if t < min(test_frames)]
        if train_pre:
            t_train_pre_max = float(ts[max(train_pre)])
            assert t_test_min - sig_dur >= t_train_pre_max - 1e-4, (
                f"CSI Leakage in {r['sid']}: Pre-test train frame at {t_train_pre_max:.2f}s "
                f"overlaps test window starting at {t_test_min - sig_dur:.2f}s"
            )
        
        train_post = [t for t in train_frames if t > max(test_frames)]
        if train_post:
            t_train_post_min = float(ts[min(train_post)])
            assert t_train_post_min - sig_dur >= t_test_max - 1e-4, (
                f"CSI Leakage in {r['sid']}: Post-test train window starting at {t_train_post_min - sig_dur:.2f}s "
                f"overlaps test frame at {t_test_max:.2f}s"
            )


# -----------------------------------------------------------------------------
# Input Builder with LOPO Label-Free Centering
# -----------------------------------------------------------------------------
def build_gesture_inputs_5p(recs: List[Dict[str, Any]], feat_kind: str, args: Any, dev: torch.device):
    """
    Builds gesture inputs with per-recording mean subtraction:
      - Training recordings: mean over TRAINING segment frames only (source of truth: segment_pipeline.py line 218).
      - Held-out recordings (LOPO): mean over ALL frames of that recording (label-free, as required by SPEC).
    """
    raw_feats = {}
    for r in recs:
        parts = []
        if feat_kind in ("mask", "both"):
            parts.append(r["mask"])
        if feat_kind in ("kp", "both"):
            parts.append(r["kp"])
        raw_feats[r["sid"]] = np.concatenate(parts, 1)

    centered = {}
    for r in recs:
        X = raw_feats[r["sid"]]
        tr_indices = r["split"]["train"]
        if tr_indices:
            tr_f = s2.frames_of_segments(r["seg"], tr_indices)
        else:
            # LOPO held-out participant: label-free centering from that recording's own frames
            tr_f = list(range(len(X)))
        centered[r["sid"]] = X - X[tr_f].mean(0, keepdims=True)

    tens, tr_items, te_items, tr_rows = [], [], [], []
    for ri, r in enumerate(recs):
        X = centered[r["sid"]]
        tr_indices = r["split"]["train"]
        if tr_indices:
            tr_f = s2.frames_of_segments(r["seg"], tr_indices)
            tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
        tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
        act_name = (r.get("activity") or r.get("act", "")).lower()
        y = ACTS.index(act_name)
        for i in r["split"]["train"]:
            tr_items.append((ri, *r["seg"][i], y, i, r["person"], r["sid"]))
        for i in r["split"]["test"]:
            te_items.append((ri, *r["seg"][i], y, i, r["person"], r["sid"]))

    if tr_rows:
        sd = np.concatenate(tr_rows).std(0) + 1e-6
    else:
        sd = np.ones(tens[0].shape[1] * 2, dtype=np.float32)
    D = sd.shape[0] // 2
    no_vel = getattr(args, "no_velocity", False)
    scale = torch.from_numpy((sd if not no_vel else sd[:D]).astype(np.float32)).to(dev)
    return tens, tr_items, te_items, scale
def get_all_units(dry_run: bool = False) -> List[str]:
    """Generate all unit keys."""
    if dry_run:
        # 2 participants, fold 0 only
        persons = ["Priyanshu", "Sanchit"]
        units = [f"A_{p}_f0" for p in persons]
        units.append("B_pooled_f0")
        units.append(f"C_lopo_{persons[0]}")
        return units

    units = []
    # Protocol A: 5 persons x 5 folds = 25 units
    for p in ALL_PERSONS:
        for f in range(5):
            units.append(f"A_{p}_f{f}")
    # Protocol B: 5 folds = 5 units
    for f in range(5):
        units.append(f"B_pooled_f{f}")
    # Protocol C: 5 folds (held-out person) = 5 units
    for p in ALL_PERSONS:
        units.append(f"C_lopo_{p}")
    return units


# -----------------------------------------------------------------------------
# Worker Queue Claim Handling
# -----------------------------------------------------------------------------
def claim_unit(claim_dir: Path, unit_id: str) -> bool:
    """Atomic mkdir claim."""
    unit_claim = claim_dir / unit_id
    try:
        unit_claim.mkdir(parents=True, exist_ok=False)
        # Write metadata
        meta = {
            "worker_pid": os.getpid(),
            "hostname": socket.gethostname(),
            "claim_time": time.time(),
            "time_str": time.ctime(),
        }
        (unit_claim / "claim.json").write_text(json.dumps(meta, indent=2))
        return True
    except FileExistsError:
        return False


def is_unit_done(done_dir: Path, unit_id: str, args: Any) -> bool:
    done_file = done_dir / f"{unit_id}.done"
    return done_file.exists()


def mark_unit_done(done_dir: Path, unit_id: str, summary: Dict[str, Any]):
    tmp_path = done_dir / f"{unit_id}.tmp"
    final_path = done_dir / f"{unit_id}.done"
    tmp_path.write_text(json.dumps(summary, indent=2))
    tmp_path.replace(final_path)


# -----------------------------------------------------------------------------
# Unit Execution Engine
# -----------------------------------------------------------------------------
def run_single_unit(unit_id: str, args: Any, dev: torch.device) -> Dict[str, Any]:
    """
    Executes a single unit end-to-end:
      1. Prepare split for the unit.
      2. Train 4 inner OOF pose models + 1 final pose model.
      3. Extract and cache predicted features.
      4. Train SNNs across seeds (0, 1, 2) on kp, mask, both (+ speed).
      5. Save per-gesture predictions and return metrics.
    """
    parts = unit_id.split("_")
    proto = parts[0]
    out_unit_dir = Path(args.out) / unit_id
    out_unit_dir.mkdir(parents=True, exist_ok=True)
    
    # Load canonical segments
    segs_all = json.loads(Path(args.segments).read_text())
    cols = sorted({c for g in args.channels for c in v2.GROUPS[g]})
    
    # Determine participants and fold
    if proto == "A":
        person = parts[1]
        fold_idx = int(parts[2][1:])
        train_persons = [person]
        test_persons = [person]
    elif proto == "B":
        fold_idx = int(parts[2][1:])
        train_persons = ALL_PERSONS
        test_persons = ALL_PERSONS
    elif proto == "C":
        held_out_person = parts[2]
        fold_idx = -1
        train_persons = [p for p in ALL_PERSONS if p != held_out_person]
        test_persons = [held_out_person]
    else:
        raise ValueError(f"Unknown protocol in unit {unit_id}")

    print(f"\n{'='*80}\nSTARTING UNIT: {unit_id} (Proto {proto}, Dev {dev})\n{'='*80}")
    
    # Load recordings
    recs = []
    data_dir = Path(args.data)
    for m in sorted((data_dir / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        p = meta["person"]
        if p not in train_persons and p not in test_persons:
            continue
        sid = meta["sid"]
        if sid not in segs_all:
            continue
        S = segs_all[sid]["segments"]
        ts = np.load(m.parent / "frame_timestamps.npy")
        
        # Assign split for this unit
        if proto in ("A", "B"):
            folds = get_block_folds(len(S), 5)
            sp = folds[fold_idx]
        else: # Proto C (LOPO)
            if p in train_persons:
                sp = dict(train=list(range(len(S))), guard=[], test=[])
            else:
                sp = dict(train=[], guard=[], test=list(range(len(S))))
        
        if proto in ("A", "B") and (len(sp["test"]) == 0 or len(sp["train"]) == 0):
            continue
            
        ds = v2.load_recording_v2(m.parent, meta, args, cols)
        act_val = meta.get("activity", meta.get("act", ""))
        meta.update(ds=ds, ts=ts, seg=S, split=sp, act=act_val, activity=act_val)
        recs.append(meta)

    assert len(recs) > 0, f"No recordings loaded for unit {unit_id}"
    if proto in ("A", "B"):
        verify_fold_leakage(recs)

    # 1. Pose Modeling (SiWiS)
    n_ch = len(cols)
    feat_cache_file = out_unit_dir / "poses.npz"
    store = {r["sid"]: dict(mask=np.zeros((len(r["ds"]), 192), np.float32), kp=np.zeros((len(r["ds"]), 36), np.float32))
             for r in recs}

    if not args.skip_siwis or not feat_cache_file.exists():
        # Inner OOF folds for training segments
        K = 4
        print(f"[{unit_id}] Training {K} inner OOF pose models...")
        for k in range(K):
            tr_pieces, ho_pieces = [], []
            for r in recs:
                tri = np.array(r["split"]["train"])
                if len(tri) == 0:
                    continue
                groups = np.array_split(tri, K)
                g = groups[k]
                if len(g) == 0:
                    continue
                banned = set(g.tolist()) | {int(g.min()) - 1, int(g.max()) + 1}
                tr_idx = [i for i in tri.tolist() if i not in banned]
                tr_pieces.append((r, r["ds"], s2.frames_of_segments(r["seg"], tr_idx)))
                ho_pieces.append((r, r["ds"], s2.frames_of_segments(r["seg"], g.tolist())))
            
            if ho_pieces and sum(len(x) for _, _, x in tr_pieces) > 0:
                model = s2.fit_siwis(tr_pieces, args, dev, n_ch, f"{unit_id} OOF {k+1}/{K}")
                pred = tp.predict(model, ho_pieces, args.batch_size, dev)
                pooled, kp = s2.frame_features(pred)
                idx_off = 0
                for r, _, f_idx in ho_pieces:
                    store[r["sid"]]["mask"][f_idx] = pooled[idx_off:idx_off + len(f_idx)]
                    store[r["sid"]]["kp"][f_idx] = kp[idx_off:idx_off + len(f_idx)]
                    idx_off += len(f_idx)
                del model

        # Final Pose Model -> Predict Test Segments
        print(f"[{unit_id}] Training final pose model on all training frames...")
        final_tr = [(r, r["ds"], s2.frames_of_segments(r["seg"], r["split"]["train"])) for r in recs if r["split"]["train"]]
        final_te = [(r, r["ds"], s2.frames_of_segments(r["seg"], r["split"]["test"])) for r in recs if r["split"]["test"]]
        
        if final_tr:
            final_model = s2.fit_siwis(final_tr, args, dev, n_ch, f"{unit_id} Final")
            if final_te:
                te_pred = tp.predict(final_model, final_te, args.batch_size, dev)
                pooled, kp = s2.frame_features(te_pred)
                idx_off = 0
                for r, _, f_idx in final_te:
                    store[r["sid"]]["mask"][f_idx] = pooled[idx_off:idx_off + len(f_idx)]
                    store[r["sid"]]["kp"][f_idx] = kp[idx_off:idx_off + len(f_idx)]
                    idx_off += len(f_idx)
            del final_model

        # Save feature cache atomically
        np.savez_compressed(
            feat_cache_file,
            **{f"{sid}_mask": store[sid]["mask"] for sid in store},
            **{f"{sid}_kp": store[sid]["kp"] for sid in store}
        )
    else:
        print(f"[{unit_id}] Reusing saved pose features from {feat_cache_file}")
        cached = np.load(feat_cache_file)
        for sid in store:
            store[sid]["mask"] = cached[f"{sid}_mask"]
            store[sid]["kp"] = cached[f"{sid}_kp"]

    # Attach features to recs
    for r in recs:
        r["mask"] = store[r["sid"]]["mask"]
        r["kp"] = store[r["sid"]]["kp"]

    # 2. Stage 2: SNN Evaluation across Seeds
    print(f"[{unit_id}] Training & Evaluating Gesture SNN (3 seeds)...")
    snn_results = {}
    pred_rows = []

    for kind in ["kp", "mask", "both"]:
        tens, tr_items, te_items, scale = build_gesture_inputs_5p(recs, kind, args, dev)
        assert len(te_items) > 0, f"No test items for {unit_id} {kind}"
        
        seed_accs = []
        for seed in range(args.seeds):
            preds, tr_acc = s2.run_gesture_snn_once(tens, tr_items, te_items, scale, args, dev, seed)
            y_true = np.array([it[3] for it in te_items])
            acc = float(np.mean(preds == y_true))
            seed_accs.append(acc)
            
            for it, p_val in zip(te_items, preds):
                r_idx, s_idx, e_idx, y_val, seg_id, p_name, sid_name = it
                pred_rows.append({
                    "protocol": proto,
                    "unit": unit_id,
                    "fold": fold_idx,
                    "seed": seed,
                    "kind": kind,
                    "person": p_name,
                    "sid": sid_name,
                    "seg_id": seg_id,
                    "y_true": ACTS[y_val],
                    "y_pred": ACTS[p_val],
                    "correct": int(y_val == p_val)
                })

        mean_acc = float(np.mean(seed_accs))
        std_acc = float(np.std(seed_accs))
        snn_results[kind] = {
            "mean_acc": mean_acc,
            "std_acc": std_acc,
            "seed_accs": seed_accs,
            "n_test": len(te_items)
        }
        print(f"  [{unit_id}] {kind:<5}: {mean_acc*100:5.2f}% +- {std_acc*100:4.2f}% ({len(te_items)} test gestures)")

    # Save per-unit predictions CSV atomically
    pred_csv_path = out_unit_dir / "predictions.csv"
    with open(pred_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["protocol", "unit", "fold", "seed", "kind", "person", "sid", "seg_id", "y_true", "y_pred", "correct"])
        w.writeheader()
        w.writerows(pred_rows)

    unit_summary = {
        "unit_id": unit_id,
        "protocol": proto,
        "fold": fold_idx,
        "epochs": args.epochs,
        "snn_epochs": args.snn_epochs,
        "seeds": args.seeds,
        "config_hash": f"ep{args.epochs}_snn{args.snn_epochs}_sd{args.seeds}_lr{args.lr}",
        "n_test_gestures": snn_results["both"]["n_test"],
        "results": snn_results,
        "timestamp": time.time(),
        "time_str": time.ctime(),
    }
    (out_unit_dir / "unit_summary.json").write_text(json.dumps(unit_summary, indent=2))
    return unit_summary


# -----------------------------------------------------------------------------
# Main Worker Loop
# -----------------------------------------------------------------------------
def run_worker(args):
    claim_dir = Path(args.out) / "claims"
    done_dir = Path(args.out) / "done"
    claim_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_claims:
        print("Resetting stale claims without .done marker...")
        for c in claim_dir.glob("*"):
            if c.is_dir() and not (done_dir / f"{c.name}.done").exists():
                print(f"  Removing stale claim: {c.name}")
                for sub in c.glob("*"):
                    sub.unlink()
                c.rmdir()

    units = args.units if args.units else get_all_units(dry_run=(args.mode == "dry_run"))
    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print(f"Worker started on {socket.gethostname()} (PID {os.getpid()}) | Device: {dev}")
    print(f"Total units in queue: {len(units)}")

    claimed_count = 0
    for u in units:
        if is_unit_done(done_dir, u, args):
            continue
        if claim_unit(claim_dir, u):
            print(f"\n>>> CLAIMED UNIT: {u} <<<")
            claimed_count += 1
            t0 = time.time()
            summary = run_single_unit(u, args, dev)
            mark_unit_done(done_dir, u, summary)
            print(f">>> COMPLETED UNIT: {u} in {time.time() - t0:.1f}s <<<\n")
            if args.mode in ("canary", "dry_run") and claimed_count >= (1 if args.mode == "canary" else len(units)):
                break

    print(f"Worker PID {os.getpid()} finished. No more unclaimed units.")


# -----------------------------------------------------------------------------
# Regression Check (Protocol A Fold 5 on 4 Old People)
# -----------------------------------------------------------------------------
def run_regression_check(args):
    print("="*80)
    print("STEP 4: REGRESSION CHECK ON 4 OLD PARTICIPANTS (PROTOCOL A FOLD 5)")
    print("="*80)
    dev = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    old_persons = ["Priyanshu", "Sanchit", "Sir", "Vansh"]
    
    # Run fold 4 (5th block) on old people
    for p in old_persons:
        u_id = f"A_{p}_f4"
        print(f"\n--- Checking {p} (Unit {u_id}) ---")
        summary = run_single_unit(u_id, args, dev)
        print(f"Result for {p}: kp={summary['results']['kp']['mean_acc']*100:.1f}%, both={summary['results']['both']['mean_acc']*100:.1f}%")
    print("\nRegression check completed successfully.")


def main():
    ap = argparse.ArgumentParser(description="5-Participant Protocol Runner")
    ap.add_argument("--mode", choices=["worker", "dry_run", "canary", "regression"], default="worker")
    ap.add_argument("--units", nargs="+", default=None, help="Explicit list of units to process")
    ap.add_argument("--data", default="~/percom/siwis_data_5p")
    ap.add_argument("--npz_dir", default="~/percom/work/npz_5p")
    ap.add_argument("--segments", default="~/percom/siwis_data_5p/segments_5p.json")
    ap.add_argument("--out", default="~/percom/results_5p")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(v2.GROUPS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_weight", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--spacing", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--snn_epochs", type=int, default=60)
    ap.add_argument("--snn_lr", type=float, default=1e-3)
    ap.add_argument("--snn_batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0, help="Base random seed")
    ap.add_argument("--global_mean", action="store_true", help="Ablation global mean")
    ap.add_argument("--gap", type=int, default=3, help="Gap frames")
    ap.add_argument("--unit", default="gestures", choices=["gestures", "windows"])
    ap.add_argument("--no_velocity", action="store_true")
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--augment", default=True, action="store_true")
    ap.add_argument("--skip_siwis", action="store_true")
    ap.add_argument("--reset_claims", action="store_true")
    args = ap.parse_args()

    args.data = str(Path(args.data).expanduser())
    args.npz_dir = str(Path(args.npz_dir).expanduser())
    args.segments = str(Path(args.segments).expanduser())
    args.out = str(Path(args.out).expanduser())

    if args.mode == "dry_run":
        args.epochs = 1
        args.snn_epochs = 1
        args.seeds = 1
        run_worker(args)
    elif args.mode == "regression":
        args.segments = str(Path("~/percom/siwis_data/segments.json").expanduser())
        args.data = str(Path("~/percom/siwis_data").expanduser())
        args.npz_dir = str(Path("~/percom/work/npz").expanduser())
        run_regression_check(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
