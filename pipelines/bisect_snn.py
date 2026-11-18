#!/usr/bin/env python3
"""
bisect_snn.py - Diagnostic & Bisect script for SNN on gesture segments.
Compares SNN accuracies under:
1. Scenario 1 saved P1 pose features (~/percom/siwis_results_seg/poses_Priyanshu)
2. Ground-truth video features (CONTROL, should be 99-100%)
3. Scenario 2 P1 pose features (~/percom/scenario2_pooled/check_p1/pooled_feats_gestures.npz)
4. Centering ablation: Per-Recording vs Per-Person vs Global vs No-Center
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Setup paths
for p in [Path.cwd(), Path.cwd() / "siwis", Path.cwd() / "Codebase" / "percom"]:
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
import percom_common as pc
import pose_snn as ps
import segment_pipeline as sp

ACTS = ["bhu", "lhu", "rhu"]


def gt_feats(rec_dir: Path):
    """CONTROL: the same two feature sets computed from VIDEO labels."""
    mask = np.load(rec_dir / "mask_labels.npy").astype(np.float32)
    lz = pc.LazyHeatmaps(np.load(rec_dir / "kp_xy.npy"), np.load(rec_dir / "kp_vis.npy"))
    heat = np.stack([lz[i] for i in range(len(mask))]).astype(np.float32)
    n, h, w = mask.shape
    pooled = mask[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4)).reshape(n, -1)
    return pooled, ps.kp_features(heat)


def load_from_s1_poses(s1_dir: Path, person: str, source: str, data_dir: Path):
    """Load from Scenario 1 poses_<Person> folder."""
    d = s1_dir / f"poses_{person}"
    if not d.exists():
        d = s1_dir
    recs = []
    split_info = json.loads((d / "split.json").read_text())
    for e in split_info:
        sid = e["sid"]
        z = np.load(d / f"{sid}_feat.npz")
        e["ts"] = z["ts"]
        if source == "gt":
            e["mask"], e["kp"] = gt_feats(data_dir / "rec" / sid)
        else:
            e["mask"], e["kp"] = z["mask"], z["kp"]
        recs.append(e)
    return recs


def load_from_s2_npz(npz_path: Path, source: str, data_dir: Path):
    """Load from Scenario 2 pooled_feats_gestures.npz."""
    z = np.load(npz_path, allow_pickle=False)
    recs = []
    for j, name in enumerate(z["names"]):
        sid = str(name)
        p = str(z["persons"][j])
        act = str(z["acts"][j])
        e = {
            "sid": sid,
            "person": p,
            "act": act,
            "activity": act,
            "segments": z[f"segments_{sid}"].tolist(),
            "train": z[f"split_train_{sid}"].tolist(),
            "test": z[f"split_test_{sid}"].tolist(),
        }
        if source == "gt":
            e["mask"], e["kp"] = gt_feats(data_dir / "rec" / sid)
        else:
            e["mask"] = z[f"mask_{sid}"]
            e["kp"] = z[f"kp_{sid}"]
        recs.append(e)
    return recs


def build_inputs(recs, kind: str, centering: str, dev: torch.device):
    """
    centering:
      - 'recording': per-recording mean from training segments (Scenario 1 code)
      - 'person': per-person mean from training segments
      - 'global': global mean across all recordings' training segments
      - 'none': no mean subtracted
    """
    raw_feats = {}
    for r in recs:
        parts = []
        if kind in ("mask", "both"):
            parts.append(r["mask"])
        if kind in ("kp", "both"):
            parts.append(r["kp"])
        raw_feats[r["sid"]] = np.concatenate(parts, 1)

    tens, tr_items, te_items, tr_rows = [], [], [], []

    if centering == "global":
        tr_frames = [raw_feats[r["sid"]][sp.frames_of(r["segments"], r["train"])] for r in recs]
        mu = np.concatenate(tr_frames, axis=0).mean(0, keepdims=True)
        for ri, r in enumerate(recs):
            X = raw_feats[r["sid"]] - mu
            tr_f = sp.frames_of(r["segments"], r["train"])
            tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
            tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
            y = ACTS.index(r["act"])
            for i in r["train"]:
                tr_items.append((ri, *r["segments"][i], y, i))
            for i in r["test"]:
                te_items.append((ri, *r["segments"][i], y, i))

    elif centering == "person":
        for p in sorted({r.get("person", "P1") for r in recs}):
            p_tr_frames = [
                raw_feats[r["sid"]][sp.frames_of(r["segments"], r["train"])]
                for r in recs
                if r.get("person", "P1") == p
            ]
            mu_p = np.concatenate(p_tr_frames, axis=0).mean(0, keepdims=True)
            for r in recs:
                if r.get("person", "P1") == p:
                    raw_feats[r["sid"]] = raw_feats[r["sid"]] - mu_p
        for ri, r in enumerate(recs):
            X = raw_feats[r["sid"]]
            tr_f = sp.frames_of(r["segments"], r["train"])
            tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
            tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
            y = ACTS.index(r["act"])
            for i in r["train"]:
                tr_items.append((ri, *r["segments"][i], y, i))
            for i in r["test"]:
                te_items.append((ri, *r["segments"][i], y, i))

    elif centering == "recording":  # Scenario 1 segment_pipeline.py
        for ri, r in enumerate(recs):
            X = raw_feats[r["sid"]].copy()
            tr_f = sp.frames_of(r["segments"], r["train"])
            X = X - X[tr_f].mean(0, keepdims=True)
            tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
            tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
            y = ACTS.index(r["act"])
            for i in r["train"]:
                tr_items.append((ri, *r["segments"][i], y, i))
            for i in r["test"]:
                te_items.append((ri, *r["segments"][i], y, i))

    else:  # 'none'
        for ri, r in enumerate(recs):
            X = raw_feats[r["sid"]].copy()
            tr_f = sp.frames_of(r["segments"], r["train"])
            tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
            tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
            y = ACTS.index(r["act"])
            for i in r["train"]:
                tr_items.append((ri, *r["segments"][i], y, i))
            for i in r["test"]:
                te_items.append((ri, *r["segments"][i], y, i))

    sd = np.concatenate(tr_rows).std(0) + 1e-6
    scale = torch.from_numpy(sd.astype(np.float32)).to(dev)
    return tens, tr_items, te_items, scale


def evaluate_snn(tens, tr_items, te_items, scale, dev: torch.device, seeds: int = 3):
    accs = []
    snn_args = type(
        "Args",
        (),
        {
            "augment": True,
            "no_velocity": False,
            "snn_lr": 1e-3,
            "snn_epochs": 60,
            "snn_batch": 16,
        },
    )()
    for seed in range(seeds):
        pred, tr_acc = sp.run_snn_once(tens, tr_items, te_items, scale, snn_args, dev, seed)
        y = np.array([it[3] for it in te_items])
        accs.append(float(np.mean(pred == y)))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1_poses", default="~/percom/siwis_results_seg/poses_Priyanshu")
    ap.add_argument("--s2_npz", default="~/percom/scenario2_pooled/check_p1/pooled_feats_gestures.npz")
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    dev = torch.device(args.device)
    data_dir = Path(args.data).expanduser()
    s1_path = Path(args.s1_poses).expanduser()
    s2_path = Path(args.s2_npz).expanduser()

    print("=" * 80)
    print("BISECTING SNN ON P1 (PRIYANSHU) GESTURE FEATURES")
    print(f"Device: {dev} | Seeds: {args.seeds}")
    print("=" * 80)

    # Test 1: S2 P1 pose features (from check_p1 run)
    if s2_path.exists():
        print(f"\n--- [Test 1] S2 P1 Poses ({s2_path}) ---")
        recs_s2 = load_from_s2_npz(s2_path, "pred", data_dir)
        for centering in ["recording", "person", "global", "none"]:
            tens, tr, te, sc = build_inputs(recs_s2, "kp", centering, dev)
            m, s = evaluate_snn(tens, tr, te, sc, dev, args.seeds)
            print(f"  kp + speed | centering={centering:<10s} -> accuracy: {m*100:5.1f}% +- {s*100:.1f}%")

        for feat in ["mask", "both"]:
            tens, tr, te, sc = build_inputs(recs_s2, feat, "recording", dev)
            m, s = evaluate_snn(tens, tr, te, sc, dev, args.seeds)
            print(f"  {feat:4s} + speed | centering=recording  -> accuracy: {m*100:5.1f}% +- {s*100:.1f}%")

    # Test 2: S1 P1 pose features (if available)
    if s1_path.exists():
        print(f"\n--- [Test 2] S1 Saved Poses ({s1_path}) ---")
        recs_s1 = load_from_s1_poses(s1_path, "Priyanshu", "pred", data_dir)
        for feat in ["kp", "mask", "both"]:
            tens, tr, te, sc = build_inputs(recs_s1, feat, "recording", dev)
            m, s = evaluate_snn(tens, tr, te, sc, dev, args.seeds)
            print(f"  {feat:4s} + speed | centering=recording  -> accuracy: {m*100:5.1f}% +- {s*100:.1f}%")

    # Test 3: Ground Truth Video Poses (CONTROL)
    print(f"\n--- [Test 3] Video-Derived Control (Ground Truth Poses) ---")
    if s2_path.exists():
        recs_gt = load_from_s2_npz(s2_path, "gt", data_dir)
    elif s1_path.exists():
        recs_gt = load_from_s1_poses(s1_path, "Priyanshu", "gt", data_dir)
    else:
        recs_gt = []

    if recs_gt:
        for feat in ["kp", "mask", "both"]:
            tens, tr, te, sc = build_inputs(recs_gt, feat, "recording", dev)
            m, s = evaluate_snn(tens, tr, te, sc, dev, args.seeds)
            print(f"  CONTROL: {feat:4s} + speed | centering=recording -> accuracy: {m*100:5.1f}% +- {s*100:.1f}%")


if __name__ == "__main__":
    main()
