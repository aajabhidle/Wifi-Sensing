#!/usr/bin/env python3
"""
scenario3_lopo_pipeline.py - Scenario 3: Leave-One-Person-Out (LOPO) Wi-Fi Gesture Pipeline.
"Overall Performance Without Calibration"

Protocol:
  - 4 folds: each person (Priyanshu, Sanchit, Sir, Vansh) is the held-out test participant H in exactly 1 fold.
  - In fold H: Train on ALL recordings of the other 3 participants; Test on ALL recordings of H.
  - Pose model: 4 OOF fold models + 1 final model per fold.
  - Centering: per-recording mean pose (training recordings use own frames; H uses own label-free frames).
  - Scaler & class weights: from training participants only.
  - SNN: seeds 0, 1, 2.
  - Generates per-fold results and the AVERAGE confusion matrix.
  - Reuses modules from scenario2_pooled_pipeline.py (no copy-pasting).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Ensure local imports work
ROOT_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ROOT_DIR / "siwis", ROOT_DIR / "Codebase" / "percom"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import config
import percom_common as pc
import pose_snn as ps
import train_percom as tp
import train_percom_v2 as v2

# Import shared modules from Scenario 2 pipeline
import scenario2_pooled_pipeline as s2

ACTS = ["bhu", "lhu", "rhu"]
DISPLAY_ACTS = ["BHR", "LHR", "RHR"]
PERSON_NAMES = ["Priyanshu", "Sanchit", "Sir", "Vansh"]
PERSON_IDS = {"Priyanshu": "P1", "Sanchit": "P2", "Sir": "P3", "Vansh": "P4"}


def plot_confusion_matrix_pct(cm_pct, title, out_path):
    """Plot an already-normalized (percentage) confusion matrix correctly with % labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Accuracy (%)", fontsize=13, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(DISPLAY_ACTS, fontsize=13, fontweight="bold")
    ax.set_yticklabels(DISPLAY_ACTS, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Gesture", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Gesture", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    thresh = 50.0
    for i in range(3):
        for j in range(3):
            val = cm_pct[i, j]
            ax.text(j, i, f"{val:.1f}%",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_path.name}")


def plot_confusion_matrix(cm, title, out_path, normalize=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=200)
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, np.where(row_sums == 0, 1, row_sums)) * 100.0

    im = ax.imshow(data, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    if normalize:
        cbar.set_label("Accuracy (%)", fontsize=13, fontweight="bold", labelpad=10)
    else:
        cbar.set_label("Total Counts", fontsize=13, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(DISPLAY_ACTS, fontsize=13, fontweight="bold")
    ax.set_yticklabels(DISPLAY_ACTS, fontsize=13, fontweight="bold")
    ax.set_ylabel("True Gesture", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_xlabel("Predicted Gesture", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

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
                fontsize=14,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Configuration Banner
# -----------------------------------------------------------------------------
def print_config_table(args, dev, cols):
    print("\n" + "=" * 90)
    print("SCENARIO 3 (LOPO): LEAVE-ONE-PERSON-OUT CROSS-VALIDATION")
    print(f"Device: {dev} | Rate: {args.rate}Hz | Channels: {[v2.CHANNELS[c] for c in cols]}")
    print(f"Unit: {args.unit} | SiWiS Epochs: {args.epochs} | SNN Epochs: {args.snn_epochs} | Seeds: {args.seeds}")
    print(f"Held-out Persons (4 Folds): {PERSON_NAMES}")
    print("=" * 90)


# -----------------------------------------------------------------------------
# Gesture LOPO Setup
# -----------------------------------------------------------------------------
def setup_fold_gestures(recs_all: List[Dict[str, Any]], held_out: str):
    """Partition recordings for LOPO gestures."""
    train_recs, test_recs = [], []
    for r in recs_all:
        r_copy = dict(r)
        n_segs = len(r_copy["seg"])
        if r_copy["person"] == held_out:
            r_copy["split"] = {"train": [], "guard": [], "test": list(range(n_segs))}
            test_recs.append(r_copy)
        else:
            r_copy["split"] = {"train": list(range(n_segs)), "guard": [], "test": []}
            train_recs.append(r_copy)

    # Assert strict participant isolation
    train_persons = {r["person"] for r in train_recs}
    test_persons = {r["person"] for r in test_recs}
    assert held_out not in train_persons, f"LEAKAGE ASSERTION FAILED: {held_out} in train set!"
    assert test_persons == {held_out}, f"LEAKAGE ASSERTION FAILED: Test set has {test_persons} != {held_out}!"
    return train_recs, test_recs


# -----------------------------------------------------------------------------
# Window LOPO Setup
# -----------------------------------------------------------------------------
def setup_fold_windows(recs_all: List[Dict[str, Any]], held_out: str):
    """Partition recordings for LOPO windows."""
    train_recs, test_recs = [], []
    for r in recs_all:
        r_copy = dict(r)
        n_frames = len(r_copy["ds"])
        if r_copy["person"] == held_out:
            r_copy["cut"] = 0
            r_copy["start_test"] = 0
            test_recs.append(r_copy)
        else:
            r_copy["cut"] = n_frames
            r_copy["start_test"] = n_frames
            train_recs.append(r_copy)

    train_persons = {r["person"] for r in train_recs}
    test_persons = {r["person"] for r in test_recs}
    assert held_out not in train_persons, f"LEAKAGE ASSERTION FAILED: {held_out} in train set!"
    assert test_persons == {held_out}, f"LEAKAGE ASSERTION FAILED: Test set has {test_persons} != {held_out}!"
    return train_recs, test_recs


# -----------------------------------------------------------------------------
# Stage 1: SiWiS Pose LOPO
# -----------------------------------------------------------------------------
def run_siwis_lopo_fold(train_recs, test_recs, unit: str, held_out: str, args, dev, n_ch, fold_dir: Path):
    """Train 4 OOF fold models on 3 participants, 1 final model predicting H."""
    K = args.oof_folds
    all_recs = train_recs + test_recs
    store = {
        r["sid"]: dict(
            mask=np.zeros((len(r["ds"]), 192), np.float32),
            kp=np.zeros((len(r["ds"]), 36), np.float32),
        )
        for r in all_recs
    }

    def put_predictions(pieces, pred):
        pooled, kp = s2.frame_features(pred)
        i = 0
        for r, _, idx in pieces:
            store[r["sid"]]["mask"][idx] = pooled[i : i + len(idx)]
            store[r["sid"]]["kp"][idx] = kp[i : i + len(idx)]
            i += len(idx)

    print(f"\n--- [Fold H={held_out}] Training 4 OOF Pose Models on 3 Participants ---")
    for k in range(K):
        tr_pieces, ho_pieces = [], []
        for r in train_recs:
            if unit == "windows":
                n_f = len(r["ds"])
                e = [round(j * n_f / K) for j in range(K + 1)]
                idx_tr = list(range(0, max(0, e[k] - args.gap))) + list(range(min(n_f, e[k + 1] + args.gap), n_f))
                idx_ho = list(range(e[k], e[k + 1]))
                tr_pieces.append((r, r["ds"], idx_tr))
                ho_pieces.append((r, r["ds"], idx_ho))
            else:
                tri = np.array(r["split"]["train"])
                g = np.array_split(tri, K)[k]
                if len(g) == 0:
                    continue
                banned = set(g.tolist()) | {int(g.min()) - 1, int(g.max()) + 1}
                tr_segs = [i for i in tri.tolist() if i not in banned]
                tr_pieces.append((r, r["ds"], s2.frames_of_segments(r["seg"], tr_segs)))
                ho_pieces.append((r, r["ds"], s2.frames_of_segments(r["seg"], g.tolist())))

        n_tr = sum(len(x) for _, _, x in tr_pieces)
        n_ho = sum(len(x) for _, _, x in ho_pieces)
        print(f"  [Fold {held_out}] OOF {k+1}/{K}: {n_tr} train frames -> {n_ho} held-out frames")
        model = s2.fit_siwis(tr_pieces, args, dev, n_ch, f"{held_out} OOF {k+1}/{K}")
        pred = tp.predict(model, ho_pieces, args.batch_size, dev)
        put_predictions(ho_pieces, pred)
        del model

    # Final Model: All frames of the 3 training persons -> predicts held-out person H
    if unit == "windows":
        tr_pieces = [(r, r["ds"], list(range(len(r["ds"])))) for r in train_recs]
        te_pieces = [(r, r["ds"], list(range(len(r["ds"])))) for r in test_recs]
    else:
        tr_pieces = [(r, r["ds"], s2.frames_of_segments(r["seg"], r["split"]["train"])) for r in train_recs]
        te_pieces = [(r, r["ds"], s2.frames_of_segments(r["seg"], r["split"]["test"])) for r in test_recs]

    n_tr = sum(len(x) for _, _, x in tr_pieces)
    n_te = sum(len(x) for _, _, x in te_pieces)
    print(f"\n--- [Fold H={held_out}] Final Pose Model: {n_tr} frames (3 persons) -> predicting {n_te} frames of {held_out} ---")
    final_model = s2.fit_siwis(tr_pieces, args, dev, n_ch, f"{held_out} Final Model")
    test_pred = tp.predict(final_model, te_pieces, args.batch_size, dev)
    put_predictions(te_pieces, test_pred)
    del final_model

    # Pose Quality Evaluation on H vs training baseline
    model_score, base_score = s2.evaluate_pose_quality(test_pred, tr_pieces, args, dev)
    print(f"\n=== [Fold H={held_out}] Pose Quality on Held-out Person ===")
    print("  MODEL     :", tp.fmt(model_score))
    print("  MEAN-POSE :", tp.fmt(base_score))

    pq_out = {"held_out": held_out, "model": model_score, "baseline": base_score}
    (fold_dir / "pose_quality.json").write_text(json.dumps(pq_out, indent=2))

    # Save cached features
    feat_file = fold_dir / f"feats_{unit}.npz"
    save_dict = {
        "names": np.array([r["sid"] for r in all_recs]),
        "persons": np.array([r["person"] for r in all_recs]),
        "acts": np.array([r["activity"] for r in all_recs]),
    }
    for r in all_recs:
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
    print(f"Saved Stage 1 features to {feat_file}")
    return store


# -----------------------------------------------------------------------------
# Stage 2: SNN LOPO for Gestures
# -----------------------------------------------------------------------------
def run_lopo_gestures_fold(train_recs, test_recs, held_out: str, args, dev, fold_dir: Path):
    """Run SNN for one LOPO fold (Gestures unit)."""
    feat_file = fold_dir / "feats_gestures.npz"
    z = np.load(feat_file, allow_pickle=False)

    # Reconstruct records
    recs_snn = []
    for j, name in enumerate(z["names"]):
        name = str(name)
        recs_snn.append({
            "sid": name,
            "person": str(z["persons"][j]),
            "act": str(z["acts"][j]),
            "activity": str(z["acts"][j]),
            "mask": z[f"mask_{name}"],
            "kp": z[f"kp_{name}"],
            "segments": z[f"segments_{name}"].tolist(),
            "split_train": z[f"split_train_{name}"].tolist(),
            "split_test": z[f"split_test_{name}"].tolist(),
        })

    fold_results = {}
    predictions_rows = []

    for feat in args.feats:
        print(f"\n>>> [Fold H={held_out}] Input Feature Set: {feat} + speed")
        # Centering: per-recording mean pose
        raw_feats = {}
        for r in recs_snn:
            parts = []
            if feat in ("mask", "both"):
                parts.append(r["mask"])
            if feat in ("kp", "both"):
                parts.append(r["kp"])
            raw_feats[r["sid"]] = np.concatenate(parts, 1)

        centered = {}
        for r in recs_snn:
            X = raw_feats[r["sid"]]
            # For training recordings use own frames; for H's recordings use own frames
            centered[r["sid"]] = X - X.mean(0, keepdims=True)

        tens, tr_items, te_items, tr_rows = [], [], [], []
        for ri, r in enumerate(recs_snn):
            X = centered[r["sid"]]
            tens.append(torch.from_numpy(X.astype(np.float32)).to(dev))
            y = ACTS.index(r["act"])
            if r["person"] != held_out:
                tr_f = s2.frames_of_segments(r["segments"], r["split_train"])
                tr_rows.append(np.concatenate([X[tr_f], np.diff(X, axis=0, prepend=X[:1])[tr_f]], 1))
                for i in r["split_train"]:
                    tr_items.append((ri, *r["segments"][i], y, i, r["person"], r["sid"]))
            else:
                for i in r["split_test"]:
                    te_items.append((ri, *r["segments"][i], y, i, r["person"], r["sid"]))

        # Scaling strictly from the 3 training persons only
        sd = np.concatenate(tr_rows).std(0) + 1e-6
        scale = torch.from_numpy(sd.astype(np.float32)).to(dev)

        print(f"  Fold {held_out}: {len(tr_items)} train gestures (3 persons), {len(te_items)} test gestures of {held_out}")
        accs_seeds = []
        cm_counts = np.zeros((3, 3), dtype=int)

        for seed in range(args.seeds):
            pred, tr_acc = s2.run_gesture_snn_once(tens, tr_items, te_items, scale, args, dev, seed)
            y_true = np.array([it[3] for it in te_items])
            acc = float(np.mean(pred == y_true))
            accs_seeds.append(acc)

            for i in range(3):
                for j in range(3):
                    cm_counts[i, j] += int(np.sum((y_true == i) & (pred == j)))

            for it, seg_tuple in enumerate(te_items):
                predictions_rows.append([
                    held_out,
                    seg_tuple[5],  # sid
                    f"seg_{seg_tuple[4]}",
                    ACTS[seg_tuple[3]],
                    ACTS[pred[it]],
                    seed,
                    feat,
                    "gestures",
                ])

        mean_acc = float(np.mean(accs_seeds))
        std_acc = float(np.std(accs_seeds))
        print(f"  Fold {held_out} Gesture Accuracy: {mean_acc * 100:5.1f}% +- {std_acc * 100:.1f}%")

        fold_results[f"gestures_{feat}"] = {
            "held_out": held_out,
            "feat": feat,
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "n_test": len(te_items),
            "cm_counts": cm_counts.tolist(),
        }

        # Plot fold confusion matrix
        cm_title = f"LOPO Fold ({held_out}) - Gestures ({feat.upper()} + speed)"
        plot_confusion_matrix(cm_counts, cm_title, fold_dir / f"cm_gestures_{feat}.png", normalize=False)
        plot_confusion_matrix(cm_counts, f"{cm_title} (Row %)", fold_dir / f"cm_gestures_{feat}_norm.png", normalize=True)

    # Save fold deliverables
    (fold_dir / "results_gestures.json").write_text(json.dumps(fold_results, indent=2))
    with open(fold_dir / "predictions_gestures.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["held_out", "recording", "item_id", "true_label", "predicted_label", "seed", "input_type", "unit"])
        w.writerows(predictions_rows)
    (fold_dir / ".finished").write_text(f"Finished at {time.ctime()}\n")
    return fold_results


def run_lopo_windows_fold(train_recs, test_recs, held_out: str, args, dev, fold_dir: Path):
    """Run SNN for one LOPO fold (Windows unit)."""
    feat_file = fold_dir / "feats_windows.npz"
    z = np.load(feat_file, allow_pickle=False)

    recs_snn = []
    for j, name in enumerate(z["names"]):
        name = str(name)
        recs_snn.append({
            "sid": name,
            "person": str(z["persons"][j]),
            "act": str(z["acts"][j]),
            "activity": str(z["acts"][j]),
            "mask": z[f"mask_{name}"],
            "kp": z[f"kp_{name}"],
            "cut": int(z[f"cut_{name}"]),
            "start_test": int(z[f"start_test_{name}"]),
        })

    fold_results = {}
    predictions_rows = []

    for feat in args.feats:
        print(f"\n>>> [Fold H={held_out}] Windows Feature Set: {feat} + speed")
        raw_feats = {}
        for r in recs_snn:
            parts = []
            if feat in ("mask", "both"):
                parts.append(r["mask"])
            if feat in ("kp", "both"):
                parts.append(r["kp"])
            raw_feats[r["sid"]] = np.concatenate(parts, 1)

        centered = {}
        for r in recs_snn:
            X = raw_feats[r["sid"]]
            centered[r["sid"]] = X - X.mean(0, keepdims=True)

        Xtr, ytr = [], []
        Xte, yte, rte, wte_id = [], [], [], []

        for r in recs_snn:
            a = centered[r["sid"]]
            y = ACTS.index(r["act"])
            sid = r["sid"]
            if r["person"] != held_out:
                for w in s2.make_windows_of(a, args.window, args.train_hop, True):
                    Xtr.append(w)
                    ytr.append(y)
            else:
                w_idx = 0
                for w in s2.make_windows_of(a, args.window, args.test_hop, True):
                    Xte.append(w)
                    yte.append(y)
                    rte.append(sid)
                    wte_id.append(f"{sid}_w{w_idx}")
                    w_idx += 1

        Xtr = np.array(Xtr, np.float32)
        ytr = np.array(ytr)
        Xte = np.array(Xte, np.float32)
        yte = np.array(yte)
        rte = np.array(rte)

        print(f"  Fold {held_out}: {len(Xtr)} train windows (3 persons), {len(Xte)} test windows from {len(set(rte))} recordings of {held_out}")
        accs_w, accs_r = [], []
        cm_counts = np.zeros((3, 3), dtype=int)

        snn_lr = getattr(args, "snn_lr", 2e-3)
        snn_epochs = getattr(args, "snn_epochs", 40)
        snn_batch = getattr(args, "snn_batch", 64)

        for seed in range(args.seeds):
            snn_args = SimpleNamespace(
                seed=seed,
                epochs=snn_epochs,
                lr=snn_lr,
                batch=snn_batch,
            )
            pred, tr_acc, _ = s2.train_eval_window_snn(Xtr, ytr, Xte, snn_args, dev)
            w_acc = float(np.mean(pred == yte))
            accs_w.append(w_acc)

            # Majority vote per recording
            ok = tot = 0
            for r_sid in sorted(set(rte)):
                s = rte == r_sid
                pred_vote = np.bincount(pred[s], minlength=3).argmax()
                ok += int(pred_vote == yte[s][0])
                tot += 1
            r_acc = ok / tot
            accs_r.append(r_acc)

            for i in range(3):
                for j in range(3):
                    cm_counts[i, j] += int(np.sum((yte == i) & (pred == j)))

            for it_id, r_sid, y_true, y_pred in zip(wte_id, rte, yte, pred):
                predictions_rows.append([
                    held_out,
                    r_sid,
                    it_id,
                    ACTS[y_true],
                    ACTS[y_pred],
                    seed,
                    feat,
                    "windows",
                ])

        mean_w = float(np.mean(accs_w))
        std_w = float(np.std(accs_w))
        mean_r = float(np.mean(accs_r))
        std_r = float(np.std(accs_r))
        print(f"  Fold {held_out} Window Acc: {mean_w * 100:5.1f}% +- {std_w * 100:.1f}%, Rec Vote: {mean_r * 100:5.1f}% +- {std_r * 100:.1f}%")

        fold_results[f"windows_{feat}"] = {
            "held_out": held_out,
            "feat": feat,
            "acc_mean": mean_w,
            "acc_std": std_w,
            "window_acc_mean": mean_w,
            "window_acc_std": std_w,
            "recording_acc_mean": mean_r,
            "recording_acc_std": std_r,
            "n_test_windows": len(yte),
            "n_test_recordings": tot,
            "cm_counts": cm_counts.tolist(),
        }

        cm_title = f"LOPO Fold ({held_out}) - Windows ({feat.upper()} + speed)"
        plot_confusion_matrix(cm_counts, cm_title, fold_dir / f"cm_windows_{feat}.png", normalize=False)
        plot_confusion_matrix(cm_counts, f"{cm_title} (Row %)", fold_dir / f"cm_windows_{feat}_norm.png", normalize=True)

    (fold_dir / "results_windows.json").write_text(json.dumps(fold_results, indent=2))
    with open(fold_dir / "predictions_windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["held_out", "recording", "item_id", "true_label", "predicted_label", "seed", "input_type", "unit"])
        w.writerows(predictions_rows)
    (fold_dir / ".finished").write_text(f"Finished at {time.ctime()}\n")
    return fold_results


# -----------------------------------------------------------------------------
# Main LOPO Pipeline Runner
# -----------------------------------------------------------------------------
def run_lopo_pipeline(args):
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = sorted({c for g in args.channels for c in v2.GROUPS[g]})

    print_config_table(args, dev, cols)

    # 1. Load Data
    if args.unit == "gestures":
        sp_path = Path(args.segments or (Path(args.data) / "segments.json")).expanduser()
        if not sp_path.exists():
            raise SystemExit(f"segments.json not found at {sp_path}!")
        segs_all = json.loads(sp_path.read_text())
        recs_all = s2.load_all_gestures(args, cols, segs_all)
    else:
        recs_all = s2.load_all_windows(args, cols)

    # Check that all 4 participants are present
    available_persons = sorted({r["person"] for r in recs_all})
    assert len(available_persons) == 4, f"Expected 4 participants, found {available_persons}!"

    # 2. Iterate over 4 LOPO folds
    all_fold_results = {}
    for fold_idx, held_out in enumerate(PERSON_NAMES):
        if args.smoke and fold_idx > 0:
            print("\n[SMOKE TEST] Successfully completed 1 fold. Exiting early as requested.")
            break
        print("\n" + "#" * 90)
        print(f"STARTING LOPO FOLD {fold_idx + 1}/4: HELD-OUT PARTICIPANT = {held_out} ({PERSON_IDS[held_out]})")
        print("#" * 90)

        fold_dir = out_dir / f"fold_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Check for resumption
        if (fold_dir / ".finished").exists() and not args.force:
            print(f"Fold {held_out} already marked .finished, skipping to next fold (use --force to rerun).")
            res_file = fold_dir / f"results_{args.unit}.json"
            if res_file.exists():
                all_fold_results[held_out] = json.loads(res_file.read_text())
            continue

        if args.unit == "gestures":
            tr_recs, te_recs = setup_fold_gestures(recs_all, held_out)
        else:
            tr_recs, te_recs = setup_fold_windows(recs_all, held_out)

        # Stage 1: Pose Model for Fold
        feat_file = fold_dir / f"feats_{args.unit}.npz"
        if not args.skip_siwis:
            if feat_file.exists() and not args.force:
                print(f"Features already exist at {feat_file}, skipping SiWiS Stage 1 for fold {held_out} (resuming SNN).")
            else:
                run_siwis_lopo_fold(tr_recs, te_recs, args.unit, held_out, args, dev, len(cols), fold_dir)

        # Stage 2: SNN for Fold
        if not args.skip_snn:
            if args.unit == "gestures":
                res = run_lopo_gestures_fold(tr_recs, te_recs, held_out, args, dev, fold_dir)
            else:
                res = run_lopo_windows_fold(tr_recs, te_recs, held_out, args, dev, fold_dir)
            all_fold_results[held_out] = res

    # 3. Aggregate across folds: Compute Average Confusion Matrix & Metrics
    print("\n" + "=" * 90)
    print("LOPO 4-FOLD AGGREGATION & AVERAGE CONFUSION MATRIX")
    print("=" * 90)

    for feat in args.feats:
        key = f"{args.unit}_{feat}"
        accs = [all_fold_results[p][key]["acc_mean"] for p in PERSON_NAMES if p in all_fold_results and key in all_fold_results[p]]
        if not accs:
            continue
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        print(f"\n--- {feat.upper()} + speed (Mean over 4 folds): {mean_acc * 100:.1f}% +- {std_acc * 100:.1f}% ---")
        for p in PERSON_NAMES:
            if p in all_fold_results and key in all_fold_results[p]:
                f_acc = all_fold_results[p][key]["acc_mean"]
                print(f"  Held-out {p:10s} ({PERSON_IDS[p]}): {f_acc * 100:.1f}%")

        # Average row-normalized confusion matrix across the 4 folds
        norm_cms = []
        sum_counts = np.zeros((3, 3), dtype=int)
        for p in PERSON_NAMES:
            if p in all_fold_results and key in all_fold_results[p]:
                counts = np.array(all_fold_results[p][key]["cm_counts"], dtype=float)
                sum_counts += counts.astype(int)
                row_sums = counts.sum(axis=1, keepdims=True)
                norm_cms.append(np.divide(counts, np.where(row_sums == 0, 1, row_sums)) * 100.0)

        avg_cm = np.mean(norm_cms, axis=0)
        # avg_cm is already in % (0-100), use dedicated function to show % labels
        plot_confusion_matrix_pct(avg_cm, f"Average Confusion Matrix: LOPO ({feat.upper()} + speed)",
                                  out_dir / f"confusion_matrix_avg_{args.unit}_{feat}.png")
        plot_confusion_matrix(sum_counts, f"Total Counts: LOPO ({feat.upper()} + speed)",
                              out_dir / f"confusion_matrix_sum_{args.unit}_{feat}.png", normalize=False)

        if args.unit == "gestures" and feat == "kp":
            plot_confusion_matrix_pct(avg_cm, "Scenario 3 LOPO: Gestures (Joints + Speed)",
                                      out_dir / "confusion_matrix_main_gestures_joints.png")

    print("\n##### Done with Scenario 3 LOPO Pipeline Execution #####")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Scenario 3 (LOPO) Wi-Fi Gesture Pipeline")
    ap.add_argument("--data", default="~/percom/siwis_data", help="Path to siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz", help="Path to converted CSI npz files")
    ap.add_argument("--segments", default="", help="Path to segments.json (default: <data>/segments.json)")
    ap.add_argument("--out", default="scenario3_lopo", help="Output directory")
    ap.add_argument("--unit", default="gestures", choices=["gestures", "windows"], help="Evaluation unit")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase", "aoa", "vel"], choices=list(v2.GROUPS))
    ap.add_argument("--persons", nargs="*", default=[], help="Filter persons")
    ap.add_argument("--exclude", nargs="*", default=["Divij"], help="Excluded persons")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Test fraction")
    ap.add_argument("--epochs", type=int, default=60, help="SiWiS pose model epochs")
    ap.add_argument("--batch_size", type=int, default=16, help="SiWiS batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="SiWiS learning rate")
    ap.add_argument("--pose_weight", type=float, default=500.0, help="SiWiS pose loss weight")
    ap.add_argument("--rate", type=float, default=150.0, help="CSI sampling rate Hz")
    ap.add_argument("--spacing", type=float, default=0.5, help="Antenna spacing fraction")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--oof_folds", type=int, default=4, help="Number of out-of-fold models per LOPO fold")
    ap.add_argument("--gap", type=int, default=3, help="Frames dropped between train and test blocks")
    ap.add_argument("--seed", type=int, default=0, help="Base random seed")
    ap.add_argument("--skip_siwis", action="store_true", help="Skip SiWiS pose training and use saved features")
    ap.add_argument("--skip_snn", action="store_true", help="Skip SNN training")
    ap.add_argument("--feats", nargs="+", default=["kp", "mask", "both"], choices=["kp", "mask", "both"])
    ap.add_argument("--window", type=int, default=10, help="Window frames")
    ap.add_argument("--train_hop", type=int, default=2, help="Train window hop")
    ap.add_argument("--test_hop", type=int, default=5, help="Test window hop")
    ap.add_argument("--snn_epochs", type=int, default=60, help="SNN training epochs")
    ap.add_argument("--snn_lr", type=float, default=1e-3, help="SNN learning rate")
    ap.add_argument("--snn_batch", type=int, default=16, help="SNN batch size")
    ap.add_argument("--seeds", type=int, default=3, help="SNN seeds (0, 1, 2)")
    ap.add_argument("--augment", action="store_true", default=True, help="Gesture augmentation")
    ap.add_argument("--smoke", action="store_true", help="Smoke test (1 epoch, 1 seed, 1 fold)")
    ap.add_argument("--force", action="store_true", help="Force rerun even if fold marked .finished")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.smoke:
        print("[SMOKE TEST MODE ENABLED]")
        args.epochs = 1
        args.snn_epochs = 1
        args.seeds = 1
        args.oof_folds = 2
        args.workers = 0

    run_lopo_pipeline(args)


if __name__ == "__main__":
    main()
