#!/usr/bin/env python3
"""
scenario2_pooled_pipeline.py - SCENARIO 2 (POOLED) Wi-Fi Gesture Pipeline.
Pools all 4 people's data (Priyanshu P1, Sanchit P2, Sir P3, Vansh P4) together.

Trains ONE SiWiS pose model and ONE SNN on all 4 people's data together,
with an 80/20 time split per recording, and evaluates:
  1. Complete-gesture unit (--unit gestures, default): evaluated on whole segments from segments.json.
  2. Window unit (--unit windows): evaluated on 10-frame (5 s) sliding windows.

Key features:
  - 4 OOF fold models + 1 final pose model (5 pose models in total, not 16 or 20).
  - Complete leak-free splits with 3-frame gaps between train/test and OOF blocks.
  - Primary normalization: per-recording centering — gestures: subtract training-segment-frame mean
    (matches segment_pipeline.py line 218); windows: subtract mean of train+test frames
    (matches within_pipeline.py line 182). Source of truth: the Scenario 1 code, not the SPEC.
  - Ablation (--global_mean): subtract one global mean from the pooled training set.
  - Feature scaling and class weights derived exclusively from pooled training data.
  - Classifiers evaluated with 3 seeds (0, 1, 2) on joints (36), outline (192), and both (+ speed).
  - Generates results.json, results.csv, predictions.csv, publication confusion matrices,
    pose quality reports, and Scenario 1 vs 2 comparison table.
  - Handles SIGTERM gracefully and checkpoints after each fold/stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

# Ensure relevant folders are in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
for p in [
    REPO_ROOT,
    REPO_ROOT / "siwis",
    REPO_ROOT / "Codebase" / "percom",
    REPO_ROOT / "Codebase" / "percom" / "segment_kit",
    REPO_ROOT / "Codebase" / "percom" / "project" / "siwis_percom",
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
from losses.losses import mask_loss, pose_loss
from models.signal_encoder import SignalEncoder
from models.siwis_net import SiWiSNet

ACTS: List[str] = ["bhu", "lhu", "rhu"]
DISPLAY_ACTS: List[str] = ["bhr", "lhr", "rhr"]
PERSON_NAMES = ["Priyanshu", "Sanchit", "Sir", "Vansh"]
PERSON_IDS = {"Priyanshu": "P1", "Sanchit": "P2", "Sir": "P3", "Vansh": "P4"}

# -----------------------------------------------------------------------------
# Graceful Shutdown / Checkpointing on SIGTERM
# -----------------------------------------------------------------------------
GLOBAL_CHECKPOINT_STATE = {}


def sigterm_handler(sig, frame):
    print("\n[WARNING] Received SIGTERM! Saving checkpoint state before termination...", flush=True)
    if "save_fn" in GLOBAL_CHECKPOINT_STATE:
        try:
            GLOBAL_CHECKPOINT_STATE["save_fn"]()
            print("[INFO] Checkpoint saved successfully.", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)


# -----------------------------------------------------------------------------
# Stage 1: SiWiS Pose Model Fitting and Feature Extraction
# -----------------------------------------------------------------------------
def fit_siwis(pieces, args, dev, n_ch, tag):
    """Train SiWiSNet (same recipe as train_percom_v2.py) on pieces = [(meta, dataset, indices)]."""
    n = sum(len(i) for _, _, i in pieces)
    if n == 0:
        raise ValueError(f"No frames in training pieces for {tag}!")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SiWiSNet()
    model.signal_encoder = SignalEncoder(in_channels=n_ch)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(
        ConcatDataset([Subset(d, idx) for _, d, idx in pieces if len(idx)]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        drop_last=(n >= args.batch_size),
    )
    npix = config.HEATMAP_H * config.HEATMAP_W
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        lm_s = lp_s = 0.0
        nb = 0
        for b in loader:
            o = model(b["signal"].to(dev))
            B = o["mask"].shape[0]
            lm = mask_loss(torch.sigmoid(o["mask"]), b["mask_label"].to(dev)) / (B * npix)
            lp = pose_loss(
                o["pose_heatmaps"],
                b["keypoint_label"].to(dev),
                o["pose_tags"],
                b["keypoint_coords"].to(dev),
                b["keypoint_visible"].to(dev),
            )
            loss = lm + args.pose_weight * lp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lm_s += float(lm.detach())
            lp_s += float(lp.detach())
            nb += 1
        sched.step()
        if ep == 0 or (ep + 1) % max(args.epochs // 4, 1) == 0 or ep == args.epochs - 1:
            print(
                f"    [{tag}] epoch {ep + 1:3d}/{args.epochs}  mask {lm_s / max(nb, 1):.4f}  "
                f"pose {lp_s / max(nb, 1):.5f}  ({time.time() - t0:.0f} s, {n} frames)",
                flush=True,
            )
    return model


def frame_features(pred: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract per-frame pooled 12x16 mask (192) and joint features (36)."""
    pm = pred["pm"].astype(np.float32)
    n, h, w = pm.shape
    pooled = pm[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4)).reshape(n, -1)
    kp = ps.kp_features(pred["pp"])
    return pooled, kp


# -----------------------------------------------------------------------------
# Split Definitions & Assertions
# -----------------------------------------------------------------------------
def split_regions_windows(n: int, args) -> Tuple[int, int]:
    """Train cut and test start index for window unit (80/20 chronological split)."""
    cut = int(n * config.TRAIN_FRACTION)
    return cut, min(cut + args.gap, n)


def split_segments_gestures(n: int, test_frac: float = 0.2) -> Optional[Dict[str, List[int]]]:
    """n segments in time order -> dict(train=[...], guard=[i], test=[...]); None if too few."""
    if n < 3:
        return None
    n_test = max(1, int(round(test_frac * n)))
    n_train = n - n_test - 1
    if n_train < 1:
        n_test, n_train = 1, n - 2
    return dict(train=list(range(n_train)), guard=[n_train], test=list(range(n_train + 1, n)))


def frames_of_segments(segs: List[List[int]], idx: List[int]) -> List[int]:
    out = []
    for i in idx:
        out += list(range(segs[i][0], segs[i][1]))
    return out


def verify_no_csi_overlap(recs, unit: str, args):
    """Assert that no test frame's 1.7s CSI window touches any training frame."""
    print("\n--- Verifying No CSI Window Overlap (Leakage Check) ---")
    sig_dur = config.SIGNAL_DURATION_SEC  # 1.7 seconds
    for r in recs:
        ts = r["ts"]
        sid = r["sid"]
        if unit == "windows":
            cut, start_test = r["split_info"]
            if start_test < len(ts) and cut > 0:
                t_last_train = float(ts[cut - 1])
                t_first_test = float(ts[start_test])
                window_start_test = t_first_test - sig_dur
                assert window_start_test >= t_last_train - 1e-5, (
                    f"CSI Overlap Violation in {sid}: test window starts at {window_start_test:.3f}s "
                    f"but last train frame is at {t_last_train:.3f}s! (gap frames: {args.gap})"
                )
        else:  # gestures
            sp = r["split"]
            segs = r["seg"]
            if sp["test"] and sp["train"]:
                last_train_frame = segs[sp["train"][-1]][1] - 1
                first_test_frame = segs[sp["test"][0]][0]
                t_last_train = float(ts[last_train_frame])
                t_first_test = float(ts[first_test_frame])
                window_start_test = t_first_test - sig_dur
                assert window_start_test >= t_last_train - 1e-5, (
                    f"CSI Overlap Violation in {sid}: test segment starts at {window_start_test:.3f}s "
                    f"but last train segment ends at {t_last_train:.3f}s! Guard segment violated."
                )
    print("PASS: Verified that no test frame's 1.7s CSI window overlaps any training window.")


# -----------------------------------------------------------------------------
# Data Loading: Windows and Gestures
# -----------------------------------------------------------------------------
def load_all_windows(args, cols) -> List[Dict[str, Any]]:
    """Load recordings for window-unit evaluation."""
    data_dir = Path(args.data).expanduser()
    recs = []
    for m in sorted((data_dir / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        person = meta["person"]
        if person.lower() in {e.lower() for e in args.exclude}:
            continue
        if args.persons and person.lower() not in {p.lower() for p in args.persons}:
            continue
        ts = np.load(m.parent / "frame_timestamps.npy")
        cut, start_test = split_regions_windows(len(ts), args)
        first_test = start_test if start_test < len(ts) else len(ts) - 1
        ref_until = float(ts[first_test]) - config.SIGNAL_DURATION_SEC
        ds = v2.load_recording_v2(m.parent, meta, args, cols)
        meta.update(
            ds=ds,
            ts=ts,
            cut=cut,
            start_test=start_test,
            split_info=(cut, start_test),
        )
        recs.append(meta)
    if not recs:
        raise SystemExit("No usable recordings found for windows unit!")
    return recs


def load_all_gestures(args, cols, segs_all) -> List[Dict[str, Any]]:
    """Load recordings for complete-gesture unit."""
    data_dir = Path(args.data).expanduser()
    recs = []
    for m in sorted((data_dir / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        person = meta["person"]
        if person.lower() in {e.lower() for e in args.exclude}:
            continue
        if args.persons and person.lower() not in {p.lower() for p in args.persons}:
            continue
        sid = meta["sid"]
        if sid not in segs_all:
            print(f"  skip {sid}: not in segments.json")
            continue
        S = segs_all[sid]["segments"]
        sp = split_segments_gestures(len(S), args.test_frac)
        if sp is None:
            print(f"  skip {sid}: only {len(S)} segments (need >= 3)")
            continue
        ts = np.load(m.parent / "frame_timestamps.npy")
        if len(ts) != segs_all[sid]["n_frames"]:
            raise SystemExit(f"{sid}: frame timestamp count mismatch with segments.json!")
        first_test = S[sp["test"][0]][0]
        ref_until = float(ts[first_test]) - config.SIGNAL_DURATION_SEC
        ds = v2.load_recording_v2(m.parent, meta, args, cols)
        meta.update(ds=ds, ts=ts, seg=S, split=sp)
        recs.append(meta)
    if not recs:
        raise SystemExit("No usable recordings found for gestures unit!")
    return recs


# -----------------------------------------------------------------------------
# Pose Quality Scoring
# -----------------------------------------------------------------------------
def evaluate_pose_quality(pred_all, tr_pieces, args, dev):
    """Compute outline score, body points, wrists found vs mean-pose baseline."""
    pts = pc.argmax_points(pred_all["pp"])
    model_score = tp.score(pred_all["pm"], pred_all["gm"], pts, pred_all["gc"], pred_all["gv"])
    base_score = tp.baseline(tr_pieces, pred_all)
    return model_score, base_score


# -----------------------------------------------------------------------------
# Pooled Pose Pipeline (Stage 1)
# -----------------------------------------------------------------------------
def run_siwis_pooled(recs, unit: str, args, dev, n_ch, out_dir: Path):
    """Train 4 OOF fold models and 1 final model on POOLED data from all 4 persons."""
    K = args.oof_folds
    store = {
        r["sid"]: dict(
            mask=np.zeros((len(r["ds"]), 192), np.float32),
            kp=np.zeros((len(r["ds"]), 36), np.float32),
        )
        for r in recs
    }

    def put_predictions(pieces, pred):
        pooled, kp = frame_features(pred)
        i = 0
        for r, _, idx in pieces:
            store[r["sid"]]["mask"][idx] = pooled[i : i + len(idx)]
            store[r["sid"]]["kp"][idx] = kp[i : i + len(idx)]
            i += len(idx)

    print("\n" + "=" * 90)
    print(f"STAGE 1: POOLED SiWiS POSE MODEL ({unit.upper()} UNIT, {K} OOF FOLDS + 1 FINAL MODEL)")
    print("=" * 90)

    # 1. Out-of-fold models (4 models in total across ALL people)
    for k in range(K):
        tr_pieces, ho_pieces = [], []
        for r in recs:
            if unit == "windows":
                cut = r["cut"]
                e = [round(j * cut / K) for j in range(K + 1)]
                idx_tr = list(range(0, max(0, e[k] - args.gap))) + list(range(min(cut, e[k + 1] + args.gap), cut))
                idx_ho = list(range(e[k], e[k + 1]))
                tr_pieces.append((r, r["ds"], idx_tr))
                ho_pieces.append((r, r["ds"], idx_ho))
            else:  # gestures
                tri = np.array(r["split"]["train"])
                g = np.array_split(tri, K)[k]
                if len(g) == 0:
                    continue
                banned = set(g.tolist()) | {int(g.min()) - 1, int(g.max()) + 1}
                tr_segs = [i for i in tri.tolist() if i not in banned]
                tr_pieces.append((r, r["ds"], frames_of_segments(r["seg"], tr_segs)))
                ho_pieces.append((r, r["ds"], frames_of_segments(r["seg"], g.tolist())))

        n_tr = sum(len(x) for _, _, x in tr_pieces)
        n_ho = sum(len(x) for _, _, x in ho_pieces)
        print(f"\n--- OOF Fold {k + 1}/{K}: Training pooled on {n_tr} frames, predicting {n_ho} frames ---")
        model = fit_siwis(tr_pieces, args, dev, n_ch, f"Pooled OOF {k + 1}/{K}")
        pred = tp.predict(model, ho_pieces, args.batch_size, dev)
        put_predictions(ho_pieces, pred)
        del model

    # 2. Final model: trained on whole training part across ALL people -> predicts pooled test part
    if unit == "windows":
        tr_pieces = [(r, r["ds"], list(range(r["cut"]))) for r in recs]
        te_pieces = [(r, r["ds"], list(range(r["start_test"], len(r["ds"])))) for r in recs]
    else:
        tr_pieces = [(r, r["ds"], frames_of_segments(r["seg"], r["split"]["train"])) for r in recs]
        te_pieces = [(r, r["ds"], frames_of_segments(r["seg"], r["split"]["test"])) for r in recs]

    n_tr = sum(len(x) for _, _, x in tr_pieces)
    n_te = sum(len(x) for _, _, x in te_pieces)
    print(f"\n--- Final Pooled Model: Training on {n_tr} frames, predicting {n_te} test frames ---")
    final_model = fit_siwis(tr_pieces, args, dev, n_ch, "Pooled Final Model")
    test_pred = tp.predict(final_model, te_pieces, args.batch_size, dev)
    put_predictions(te_pieces, test_pred)
    del final_model

    # 3. Pose Quality Evaluation on pooled test part
    model_score, base_score = evaluate_pose_quality(test_pred, tr_pieces, args, dev)
    print("\n=== Pooled Test Part Pose Quality ===")
    print("  MODEL     :", tp.fmt(model_score))
    print("  MEAN-POSE :", tp.fmt(base_score))

    # Per-person breakdown of pose quality
    per_person_quality = {}
    for p in sorted({r["person"] for r in recs}):
        p_te = [it for it in te_pieces if it[0]["person"] == p]
        if not p_te:
            continue
        p_idx = []
        cur = 0
        for r, _, idx in te_pieces:
            if r["person"] == p:
                p_idx.extend(range(cur, cur + len(idx)))
            cur += len(idx)
        p_pred = {
            "pm": test_pred["pm"][p_idx],
            "gm": test_pred["gm"][p_idx],
            "pp": test_pred["pp"][p_idx],
            "gc": test_pred["gc"][p_idx],
            "gv": test_pred["gv"][p_idx],
        }
        p_tr = [it for it in tr_pieces if it[0]["person"] == p]
        p_pts = pc.argmax_points(p_pred["pp"])
        p_m_score = tp.score(p_pred["pm"], p_pred["gm"], p_pts, p_pred["gc"], p_pred["gv"])
        p_b_score = tp.baseline(p_tr, p_pred)
        per_person_quality[p] = {"model": p_m_score, "baseline": p_b_score}
        print(f"  {p:10s} MODEL    : {tp.fmt(p_m_score)}")
        print(f"  {p:10s} BASELINE : {tp.fmt(p_b_score)}")

    pose_quality_out = {
        "overall": {"model": model_score, "baseline": base_score},
        "per_person": per_person_quality,
    }
    (out_dir / f"pose_quality_{unit}.json").write_text(json.dumps(pose_quality_out, indent=2))

    # Save extracted features
    feat_file = out_dir / f"pooled_feats_{unit}.npz"
    save_dict = {
        "names": np.array([r["sid"] for r in recs]),
        "persons": np.array([r["person"] for r in recs]),
        "acts": np.array([r["activity"] for r in recs]),
    }
    for r in recs:
        save_dict[f"mask_{r['sid']}"] = store[r["sid"]]["mask"]
        save_dict[f"kp_{r['sid']}"] = store[r["sid"]]["kp"]
        if unit == "windows":
            save_dict[f"cut_{r['sid']}"] = r["cut"]
            save_dict[f"start_test_{r['sid']}"] = r["start_test"]
        else:
            save_dict[f"segments_{r['sid']}"] = np.array(r["seg"])
            save_dict[f"split_train_{r['sid']}"] = np.array(r["split"]["train"])
            save_dict[f"split_test_{r['sid']}"] = np.array(r["split"]["test"])
    np.savez_compressed(feat_file, **save_dict)
    print(f"\nSaved Stage 1 features to {feat_file}")
    return store


# -----------------------------------------------------------------------------
# SNN Stage for Window Unit
# -----------------------------------------------------------------------------
def make_windows_of(a, T, hop, velocity):
    if velocity:
        v = np.diff(a, axis=0, prepend=a[:1])
        a = np.concatenate([a, v], 1)
    return [a[s : s + T] for s in range(0, len(a) - T + 1, hop)]


def build_window_sets(recs, feat_kind: str, args):
    """Build train and test sets for window unit."""
    # 1. Extract raw features per recording
    raw_feats = {}
    for r in recs:
        parts = []
        if feat_kind in ("mask", "both"):
            parts.append(r["mask"])
        if feat_kind in ("kp", "both"):
            parts.append(r["kp"])
        raw_feats[r["sid"]] = np.concatenate(parts, 1)

    # 2. Centering (Subtract mean pose) — PER-RECORDING, same as within_pipeline.py line 182:
    #    a = a - a[used].mean(0, keepdims=True)   where used = train frames + test frames.
    #    For window unit, "used" = all frames that belong to train OR test (not the gap in between).
    centered = {}
    if args.global_mean:
        # Ablation: single global mean from pooled training set
        tr_frames = [raw_feats[r["sid"]][: r["cut"]] for r in recs]
        mu_global = np.concatenate(tr_frames, axis=0).mean(0, keepdims=True)
        for r in recs:
            centered[r["sid"]] = raw_feats[r["sid"]] - mu_global
    else:
        # Primary: per-recording mean using all used frames (train + test, not gap),
        # matching within_pipeline.py behaviour exactly.
        for r in recs:
            X = raw_feats[r["sid"]]
            used = np.r_[0 : r["cut"], r["start_test"] : len(X)]
            centered[r["sid"]] = X - X[used].mean(0, keepdims=True)

    # 3. Create windows
    Xtr, ytr, ptr = [], [], []
    Xte, yte, pte, rte, wte_id = [], [], [], [], []
    for r in recs:
        a = centered[r["sid"]]
        y = ACTS.index(r["activity"])
        p = r["person"]
        sid = r["sid"]
        for w in make_windows_of(a[: r["cut"]], args.window, args.train_hop, True):
            Xtr.append(w)
            ytr.append(y)
            ptr.append(p)
        w_idx = 0
        for w in make_windows_of(a[r["start_test"] :], args.window, args.test_hop, True):
            Xte.append(w)
            yte.append(y)
            pte.append(p)
            rte.append(sid)
            wte_id.append(f"{sid}_w{w_idx}")
            w_idx += 1

    return (
        np.array(Xtr, np.float32),
        np.array(ytr),
        np.array(ptr),
        np.array(Xte, np.float32),
        np.array(yte),
        np.array(pte),
        np.array(rte),
        np.array(wte_id),
    )


def train_eval_window_snn(Xtr, ytr, Xte, args, dev):
    """Train window SNN on pooled training data, evaluate on pooled test data."""
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(args.seed)
    net = ps.SNN(Xtr.shape[2]).to(dev)
    lr = getattr(args, "snn_lr", getattr(args, "lr", 2e-3))
    epochs = getattr(args, "snn_epochs", getattr(args, "epochs", 40))
    batch = getattr(args, "snn_batch", getattr(args, "batch", 64))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Class weights from pooled training set only
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), batch):
            b = perm[i : i + batch]
            loss = lossf(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        tr_acc = np.mean(
            np.concatenate([net(Xt[i : i + 256]).argmax(1).cpu().numpy() for i in range(0, len(Xt), 256)]) == ytr
        )
        pred = np.concatenate(
            [net(torch.tensor(Xte[i : i + 256], device=dev)).argmax(1).cpu().numpy() for i in range(0, len(Xte), 256)]
        )
    return pred, tr_acc, net.rate2


# -----------------------------------------------------------------------------
# SNN Stage for Complete-Gesture Unit (Segment Unit)
# -----------------------------------------------------------------------------
class SegSNN(nn.Module):
    """Linear -> LIF 256 -> LIF 128 -> leaky read-out averaged over segment."""

    def __init__(self, d_in, h1=256, h2=128, beta=0.9, thr=1.0, drop=0.3):
        super().__init__()
        self.beta, self.thr = beta, thr
        self.drop = nn.Dropout(drop)
        self.f1, self.f2, self.f3 = nn.Linear(d_in, h1), nn.Linear(h1, h2), nn.Linear(h2, 3)

    def forward(self, x, m):
        B, T, _ = x.shape
        cur = self.f1(self.drop(x))
        v1 = cur.new_zeros(B, self.f1.out_features)
        v2 = cur.new_zeros(B, self.f2.out_features)
        out = cur.new_zeros(B, 3)
        acc = cur.new_zeros(B, 3)
        for t in range(T):
            mt = m[:, t : t + 1].float()
            n1 = self.beta * v1 + cur[:, t]
            s1 = ps.Spike.apply(n1 - self.thr)
            n1 = n1 - s1 * self.thr
            n2 = self.beta * v2 + self.f2(s1)
            s2 = ps.Spike.apply(n2 - self.thr)
            n2 = n2 - s2 * self.thr
            no = self.beta * out + self.f3(s2)
            v1 = mt * n1 + (1 - mt) * v1
            v2 = mt * n2 + (1 - mt) * v2
            out = mt * no + (1 - mt) * out
            acc = acc + mt * out
        return acc / m.sum(1, keepdim=True).clamp(min=1)


def build_gesture_inputs(recs, feat_kind: str, args, dev):
    """Build inputs for complete gestures from pooled data."""
    # 1. Feature selection
    raw_feats = {}
    for r in recs:
        parts = []
        if feat_kind in ("mask", "both"):
            parts.append(r["mask"])
        if feat_kind in ("kp", "both"):
            parts.append(r["kp"])
        raw_feats[r["sid"]] = np.concatenate(parts, 1)

    # 2. Centering (Subtract mean pose) — PER-RECORDING, same as segment_pipeline.py line 218:
    #    X = X - X[tr_f].mean(0, keepdims=True)   where tr_f = frames of training segments.
    #    For the gesture unit, the mean is computed only from TRAINING segment frames.
    centered = {}
    if args.global_mean:
        tr_frames = [raw_feats[r["sid"]][frames_of_segments(r["segments"], r["split_train"])] for r in recs]
        mu_global = np.concatenate(tr_frames, axis=0).mean(0, keepdims=True)
        for r in recs:
            centered[r["sid"]] = raw_feats[r["sid"]] - mu_global
    else:
        # Primary: per-recording mean from training segments only (source of truth: segment_pipeline.py).
        for r in recs:
            X = raw_feats[r["sid"]]
            tr_f = frames_of_segments(r["segments"], r["split_train"])
            centered[r["sid"]] = X - X[tr_f].mean(0, keepdims=True)

    # 3. Assemble items and training rows for scale
    tens, tr_items, te_items, tr_rows = [], [], [], []
    for ri, r in enumerate(recs):
        X = centered[r["sid"]]
        tr_f = frames_of_segments(r["segments"], r["split_train"])
        tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
        tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
        y = ACTS.index(r["act"])
        for i in r["split_train"]:
            tr_items.append((ri, *r["segments"][i], y, i, r["person"], r["sid"]))
        for i in r["split_test"]:
            te_items.append((ri, *r["segments"][i], y, i, r["person"], r["sid"]))

    # Scaling statistics from POOLED training segments only
    sd = np.concatenate(tr_rows).std(0) + 1e-6
    scale = torch.from_numpy(sd.astype(np.float32)).to(dev)
    return tens, tr_items, te_items, scale


def batch_of_gestures(items, tens, args, dev, train, rng, scale):
    seqs = []
    for ri, s, e, y, _, _, _ in items:
        if train and args.augment:
            a, b = rng.integers(0, 3), rng.integers(0, 3)
            if e - s - a - b >= 6:
                s, e = s + a, e - b
        x = tens[ri][s:e]
        x = torch.cat([x, x - torch.cat([x[:1], x[:-1]], 0)], 1)
        x = x / scale
        if train and args.augment:
            x = x + 0.05 * torch.randn_like(x)
        seqs.append(x)
    T = max(len(x) for x in seqs)
    X = torch.zeros(len(seqs), T, seqs[0].shape[1], device=dev)
    M = torch.zeros(len(seqs), T, dtype=torch.bool, device=dev)
    for i, x in enumerate(seqs):
        X[i, : len(x)] = x
        M[i, : len(x)] = True
    return X, M, torch.tensor([it[3] for it in items], device=dev)


def run_gesture_snn_once(tens, tr_items, te_items, scale, args, dev, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SegSNN(int(scale.numel())).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.snn_lr, weight_decay=1e-4)
    ytr = np.array([it[3] for it in tr_items])
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    for ep in range(args.snn_epochs):
        net.train()
        perm = rng.permutation(len(tr_items))
        for i in range(0, len(perm), args.snn_batch):
            b_items = [tr_items[j] for j in perm[i : i + args.snn_batch]]
            X, M, y = batch_of_gestures(b_items, tens, args, dev, True, rng, scale)
            loss = lossf(net(X, M), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    pred, tr_pred = [], []
    with torch.no_grad():
        for items, store in ((te_items, pred), (tr_items, tr_pred)):
            for i in range(0, len(items), 16):
                X, M, y = batch_of_gestures(items[i : i + 16], tens, args, dev, False, rng, scale)
                store += net(X, M).argmax(1).cpu().tolist()
    return np.array(pred), float(np.mean(np.array(tr_pred) == ytr))


# -----------------------------------------------------------------------------
# Confusion Matrix & Reporting Helpers
# -----------------------------------------------------------------------------
def plot_confusion_matrix(cm, title, out_path, normalize=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, np.where(row_sums == 0, 1, row_sums)) * 100.0

    im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(3),
        yticks=np.arange(3),
        xticklabels=DISPLAY_ACTS,
        yticklabels=DISPLAY_ACTS,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    fmt = ".1f%" if normalize else "d"
    thresh = data.max() / 2.0
    for i in range(3):
        for j in range(3):
            val = data[i, j]
            txt = f"{val:.1f}%" if normalize else f"{int(val)}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def run_pipeline(args):
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for g in args.channels for c in v2.GROUPS[g]})

    print("\n" + "=" * 90)
    print("SCENARIO 2 (POOLED) WI-FI GESTURE PREDICTION")
    print(f"Device: {dev} | Rate: {args.rate}Hz | Channels: {[v2.CHANNELS[c] for c in cols]}")
    print(f"Unit: {args.unit} | SiWiS Epochs: {args.epochs} | SNN Epochs: {args.snn_epochs} | Seeds: {args.seeds}")
    print("=" * 90)

    # 1. Load Data & Perform Split
    if args.unit == "gestures":
        sp_path = Path(args.segments or (Path(args.data) / "segments.json")).expanduser()
        if not sp_path.exists():
            raise SystemExit(f"segments.json not found at {sp_path}!")
        segs_all = json.loads(sp_path.read_text())
        recs = load_all_gestures(args, cols, segs_all)
    else:
        recs = load_all_windows(args, cols)

    # Print Train/Test Counts
    print("\n--- Train / Test Counts Breakdown ---")
    persons = sorted({r["person"] for r in recs})
    for p in persons:
        p_recs = [r for r in recs if r["person"] == p]
        for a in ACTS:
            ap_recs = [r for r in p_recs if r["activity"] == a]
            if args.unit == "gestures":
                tr_count = sum(len(r["split"]["train"]) for r in ap_recs)
                te_count = sum(len(r["split"]["test"]) for r in ap_recs)
                print(f"  {p:10s} | {a:4s} | Train segments: {tr_count:3d} | Test segments: {te_count:3d}")
            else:
                tr_count = sum(len(make_windows_of(np.zeros((r["cut"], 1)), args.window, args.train_hop, False)) for r in ap_recs)
                te_count = sum(len(make_windows_of(np.zeros((len(r["ds"]) - r["start_test"], 1)), args.window, args.test_hop, False)) for r in ap_recs)
                print(f"  {p:10s} | {a:4s} | Train windows: {tr_count:4d} | Test windows: {te_count:4d}")

    # Assertion on Test Set Size for Gestures
    if args.unit == "gestures" and len(persons) == 4 and not args.smoke:
        tot_test_gestures = sum(len(r["split"]["test"]) for r in recs)
        print(f"\nTotal Test Gestures across 4 people: {tot_test_gestures}")
        if not args.persons:
            assert tot_test_gestures == 269, (
                f"SPEC ASSERTION FAILED: Expected exactly 269 test gestures across 4 persons, but found {tot_test_gestures}!"
            )
            print("PASS: Exact match with Scenario 1's test set (269 test gestures).")

    # Verify No CSI Window Overlap
    verify_no_csi_overlap(recs, args.unit, args)

    # 2. Stage 1: SiWiS Pose Model
    if not args.skip_siwis:
        run_siwis_pooled(recs, args.unit, args, dev, len(cols), out_dir)

    # 3. Stage 2: SNN Evaluation
    if not args.skip_snn:
        print("\n" + "=" * 90)
        print(f"STAGE 2: SNN ON CSI-PREDICTED POSES (POOLED 80/20, {args.unit.upper()} UNIT)")
        print("=" * 90)

        # Load saved poses
        feat_file = out_dir / f"pooled_feats_{args.unit}.npz"
        if not feat_file.exists():
            raise SystemExit(f"{feat_file} not found! Run SiWiS stage first.")
        z = np.load(feat_file, allow_pickle=False)

        # Reconstruct records
        recs_snn = []
        for j, name in enumerate(z["names"]):
            name = str(name)
            p = str(z["persons"][j])
            act = str(z["acts"][j])
            entry = dict(
                sid=name,
                person=p,
                act=act,
                activity=act,
                mask=z[f"mask_{name}"],
                kp=z[f"kp_{name}"],
            )
            if args.unit == "windows":
                entry["cut"] = int(z[f"cut_{name}"])
                entry["start_test"] = int(z[f"start_test_{name}"])
            else:
                entry["segments"] = z[f"segments_{name}"].tolist()
                entry["split_train"] = z[f"split_train_{name}"].tolist()
                entry["split_test"] = z[f"split_test_{name}"].tolist()
            recs_snn.append(entry)

        results_dict = {}
        predictions_rows = []

        for feat in args.feats:
            print(f"\n>>> Input Feature Set: {feat} + speed (velocity)")
            cm_counts = np.zeros((3, 3), dtype=int)
            accs_overall = []
            rec_accs_overall = []
            per_person_accs = {p: [] for p in persons}
            per_class_accs = {a: [] for a in ACTS}

            if args.unit == "windows":
                Xtr, ytr, ptr, Xte, yte, pte, rte, wte_id = build_window_sets(recs_snn, feat, args)
                print(f"  Windows: {len(Xtr)} train windows, {len(Xte)} test windows from {len(set(rte))} recordings")
                for seed in range(args.seeds):
                    snn_args = SimpleNamespace(
                        seed=seed,
                        epochs=args.snn_epochs,
                        lr=args.snn_lr,
                        batch=args.snn_batch,
                    )
                    pred, tr_acc, _ = train_eval_window_snn(Xtr, ytr, Xte, snn_args, dev)
                    w_acc = float(np.mean(pred == yte))
                    accs_overall.append(w_acc)

                    # Recording accuracy via majority vote
                    ok = tot = 0
                    for r_sid in sorted(set(rte)):
                        s = rte == r_sid
                        pred_vote = np.bincount(pred[s], minlength=3).argmax()
                        ok += int(pred_vote == yte[s][0])
                        tot += 1
                    r_acc = ok / tot
                    rec_accs_overall.append(r_acc)

                    # Per person accuracy
                    for p in persons:
                        mask_p = pte == p
                        per_person_accs[p].append(float(np.mean(pred[mask_p] == yte[mask_p])))

                    # Per class accuracy
                    for c_idx, a in enumerate(ACTS):
                        mask_c = yte == c_idx
                        per_class_accs[a].append(float(np.mean(pred[mask_c] == yte[mask_c])))

                    # Confusion matrix
                    for i in range(3):
                        for j in range(3):
                            cm_counts[i, j] += int(np.sum((yte == i) & (pred == j)))

                    # Collect predictions
                    for it_id, p, r_sid, y_true, y_pred in zip(wte_id, pte, rte, yte, pred):
                        predictions_rows.append([
                            p,
                            r_sid,
                            it_id,
                            ACTS[y_true],
                            ACTS[y_pred],
                            seed,
                            feat,
                            "windows",
                        ])

                mean_w = float(np.mean(accs_overall))
                std_w = float(np.std(accs_overall))
                mean_r = float(np.mean(rec_accs_overall))
                std_r = float(np.std(rec_accs_overall))
                print(f"  Window Accuracy:    {mean_w * 100:5.1f}% +- {std_w * 100:.1f}%")
                print(f"  Recording Accuracy: {mean_r * 100:5.1f}% +- {std_r * 100:.1f}%")

                results_dict[f"windows_{feat}"] = {
                    "unit": "windows",
                    "feat": feat,
                    "window_acc_mean": mean_w,
                    "window_acc_std": std_w,
                    "recording_acc_mean": mean_r,
                    "recording_acc_std": std_r,
                    "per_person": {p: float(np.mean(per_person_accs[p])) for p in persons},
                    "per_class": {a: float(np.mean(per_class_accs[a])) for a in ACTS},
                }

            else:  # gestures
                tens, tr_items, te_items, scale = build_gesture_inputs(recs_snn, feat, args, dev)
                print(f"  Gestures: {len(tr_items)} train gestures, {len(te_items)} test gestures")
                for seed in range(args.seeds):
                    pred, tr_acc = run_gesture_snn_once(tens, tr_items, te_items, scale, args, dev, seed)
                    y = np.array([it[3] for it in te_items])
                    # te_items tuple: (ri, s, e, y, seg_i, person, sid)
                    s_idx = np.array([it[4] for it in te_items])
                    p_arr = np.array([it[5] for it in te_items])
                    r_arr = np.array([it[6] for it in te_items])

                    g_acc = float(np.mean(pred == y))
                    accs_overall.append(g_acc)

                    for p in persons:
                        mask_p = p_arr == p
                        per_person_accs[p].append(float(np.mean(pred[mask_p] == y[mask_p])))

                    for c_idx, a in enumerate(ACTS):
                        mask_c = y == c_idx
                        per_class_accs[a].append(float(np.mean(pred[mask_c] == y[mask_c])))

                    for i in range(3):
                        for j in range(3):
                            cm_counts[i, j] += int(np.sum((y == i) & (pred == j)))

                    for it, p, r_sid, seg_i, y_true, y_pred in zip(range(len(pred)), p_arr, r_arr, s_idx, y, pred):
                        predictions_rows.append([
                            p,
                            r_sid,
                            f"seg_{seg_i}",
                            ACTS[y_true],
                            ACTS[y_pred],
                            seed,
                            feat,
                            "gestures",
                        ])

                mean_g = float(np.mean(accs_overall))
                std_g = float(np.std(accs_overall))
                lo, hi = wilson_ci(mean_g * len(te_items), len(te_items))
                print(f"  Gesture Accuracy: {mean_g * 100:5.1f}% +- {std_g * 100:.1f}% (95% CI: {lo*100:.1f}-{hi*100:.1f}%)")

                results_dict[f"gestures_{feat}"] = {
                    "unit": "gestures",
                    "feat": feat,
                    "gesture_acc_mean": mean_g,
                    "gesture_acc_std": std_g,
                    "ci95": [lo, hi],
                    "per_person": {p: float(np.mean(per_person_accs[p])) for p in persons},
                    "per_class": {a: float(np.mean(per_class_accs[a])) for a in ACTS},
                }

            # Plot Confusion Matrices
            cm_title = f"Scenario 2 (Pooled) - {args.unit.title()} - {feat.upper()} + speed"
            cm_png = out_dir / f"confusion_matrix_{args.unit}_{feat}.png"
            cm_norm_png = out_dir / f"confusion_matrix_{args.unit}_{feat}_norm.png"
            plot_confusion_matrix(cm_counts, cm_title, cm_png, normalize=False)
            plot_confusion_matrix(cm_counts, f"{cm_title} (Row %)", cm_norm_png, normalize=True)

            if args.unit == "gestures" and feat == "kp":
                # Publication-quality main figure
                main_cm_png = out_dir / "confusion_matrix_main_gestures_joints.png"
                plot_confusion_matrix(cm_counts, "Scenario 2 Pooled: Gestures (Joints + Speed)", main_cm_png, normalize=True)

        # Save results and predictions
        res_json_file = out_dir / f"results_{args.unit}.json"
        res_json_file.write_text(json.dumps(results_dict, indent=2))
        print(f"\nSaved SNN results to {res_json_file}")

        pred_csv_file = out_dir / f"predictions_{args.unit}.csv"
        with open(pred_csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["person", "recording", "item_id", "true_label", "predicted_label", "seed", "input_type", "unit"])
            w.writerows(predictions_rows)
        print(f"Saved predictions to {pred_csv_file}")

    print("\n##### Done with Scenario 2 Pooled Pipeline Execution #####")


# -----------------------------------------------------------------------------
# Command-line Interface
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Scenario 2 (Pooled) Wi-Fi Gesture Pipeline")
    ap.add_argument("--data", default="~/percom/siwis_data", help="Path to siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz", help="Path to converted CSI npz files")
    ap.add_argument("--segments", default="", help="Path to segments.json (default: <data>/segments.json)")
    ap.add_argument("--out", default="scenario2_pooled", help="Output directory")
    ap.add_argument("--unit", default="gestures", choices=["gestures", "windows", "both"], help="Evaluation unit")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(v2.GROUPS))
    ap.add_argument("--persons", nargs="*", default=[], help="Filter specific persons (default: all 4)")
    ap.add_argument("--exclude", nargs="*", default=["Divij"], help="Excluded persons")
    ap.add_argument("--epochs", type=int, default=60, help="SiWiS pose model epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="SiWiS batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="SiWiS learning rate")
    ap.add_argument("--pose_weight", type=float, default=500.0, help="SiWiS pose loss weight")
    ap.add_argument("--rate", type=float, default=150.0, help="CSI sampling rate Hz")
    ap.add_argument("--spacing", type=float, default=0.5, help="Antenna spacing fraction")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--oof_folds", type=int, default=4, help="Number of out-of-fold models")
    ap.add_argument("--gap", type=int, default=3, help="Frames dropped between train and test blocks")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Test fraction")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed")
    ap.add_argument("--skip_siwis", action="store_true", help="Skip SiWiS pose training and use saved features")
    ap.add_argument("--skip_snn", action="store_true", help="Skip SNN training")
    ap.add_argument("--feats", nargs="+", default=["kp", "mask", "both"], choices=["kp", "mask", "both"])
    ap.add_argument("--window", type=int, default=10, help="Window frames (10 = 5 s)")
    ap.add_argument("--train_hop", type=int, default=2, help="Train window hop")
    ap.add_argument("--test_hop", type=int, default=5, help="Test window hop")
    ap.add_argument("--snn_epochs", type=int, default=60, help="SNN training epochs")
    ap.add_argument("--snn_lr", type=float, default=1e-3, help="SNN learning rate")
    ap.add_argument("--snn_batch", type=int, default=16, help="SNN batch size")
    ap.add_argument("--seeds", type=int, default=3, help="SNN seeds (0, 1, 2)")
    ap.add_argument("--global_mean", action="store_true", help="Ablation: subtract single global mean instead of per-person")
    ap.add_argument("--augment", action="store_true", default=True, help="Gesture augmentation")
    ap.add_argument("--smoke", action="store_true", help="Smoke test (1 epoch, 1 seed, small subset)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.smoke:
        print("[SMOKE TEST MODE ENABLED]")
        args.epochs = 1
        args.snn_epochs = 1
        args.seeds = 1
        args.oof_folds = 2
        args.workers = 0

    if args.unit == "both":
        print("\n=== Running BOTH Complete-Gesture and Window Units ===")
        args.unit = "gestures"
        run_pipeline(args)
        args.unit = "windows"
        args.snn_epochs = 40
        args.snn_lr = 2e-3
        args.snn_batch = 64
        run_pipeline(args)
    else:
        run_pipeline(args)


if __name__ == "__main__":
    main()
