#!/usr/bin/env python3
"""
report_5p.py - Comprehensive Evaluation & Reporting for 5-Participant Protocols.
Phase 2-Ready:
  - ONE aggregation code path across all protocols.
  - Strict validation: fails loudly if any unit is incomplete (missing seed, kind, or fold).
  - No column mixing: Accuracy, F1, Precision, and Recall are ALWAYS computed from the SAME predictions.
  - Table T1: Protocol A (Participant-wise)
  - Table T2: Protocol B (Pooled)
  - Table T3: Protocol C (Leave-One-Person-Out)
  - Table T4: Protocol Comparison (A, B, C)
  - Old vs. New Segment / Frame Counts Breakdown
  - Publication-Quality Confusion Matrices (300 DPI, BHR/LHR/RHR)
  - FINAL_SUMMARY.md and results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY_ACTS = ["BHR", "LHR", "RHR"]
INTERNAL_MAP = {"bhu": "BHR", "lhu": "LHR", "rhu": "RHR"}
ALL_PERSONS = ["Priyanshu", "Sanchit", "Sir", "Vansh", "Vardaan"]
PERSON_IDS = {"Priyanshu": "P1", "Sanchit": "P2", "Sir": "P3", "Vansh": "P4", "Vardaan": "P5"}

OLD_SEGMENTS = {
    "Priyanshu": {"bhu": 98, "lhu": 93, "rhu": 116, "total": 307, "test": 64},
    "Sanchit":   {"bhu": 161, "lhu": 137, "rhu": 160, "total": 458, "test": 94},
    "Sir":       {"bhu": 97, "lhu": 93, "rhu": 99, "total": 289, "test": 58},
    "Vansh":     {"bhu": 63, "lhu": 95, "rhu": 99, "total": 257, "test": 53},
    "Vardaan":   {"bhu": 0, "lhu": 0, "rhu": 0, "total": 0, "test": 0},
}


def calc_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate accuracy, macro precision, recall, and macro F1 from the exact same predictions."""
    if len(y_true) == 0:
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
    
    classes = DISPLAY_ACTS
    y_t = [INTERNAL_MAP.get(y.lower(), y) for y in y_true]
    y_p = [INTERNAL_MAP.get(y.lower(), y) for y in y_pred]
    
    acc = float(np.mean([t == p for t, p in zip(y_t, y_p)]))
    
    precs, recs, f1s = [], [], []
    for c in classes:
        tp = sum(1 for t, p in zip(y_t, y_p) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_t, y_p) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_t, y_p) if t == c and p != c)
        
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0.0
        
        precs.append(pr)
        recs.append(rc)
        f1s.append(f)
        
    return {
        "acc": acc,
        "prec": float(np.mean(precs)),
        "rec": float(np.mean(recs)),
        "f1": float(np.mean(f1s))
    }


def plot_publication_cm(cm_pct: np.ndarray, title: str, out_path: Path):
    """Publication-quality confusion matrix with standard formatting."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=14, labelpad=8)
    cbar.ax.tick_params(labelsize=14)

    ax.set_xticks(np.arange(len(DISPLAY_ACTS)))
    ax.set_yticks(np.arange(len(DISPLAY_ACTS)))
    ax.set_xticklabels(DISPLAY_ACTS, fontsize=14)
    ax.set_yticklabels(DISPLAY_ACTS, fontsize=14)

    thresh = cm_pct.max() / 2.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=16, fontweight="bold")

    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_ylabel("True Gesture", fontsize=16, labelpad=8)
    ax.set_xlabel("Predicted Gesture", fontsize=16, labelpad=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="5-Participant Reporting Script")
    ap.add_argument("--results_dir", default="~/percom/results_5p")
    ap.add_argument("--segments", default="~/percom/siwis_data_5p/segments_5p.json")
    ap.add_argument("--allow_incomplete", action="store_true", help="Don't fail if some units are missing (preview mode)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    done_dir = results_dir / "done"
    done_files = list(done_dir.glob("*.done"))
    print(f"Loaded {len(done_files)} completed units from {done_dir}")

    # Expected units
    expected_units = []
    for p in ALL_PERSONS:
        for f in range(5):
            expected_units.append(f"A_{p}_f{f}")
    for f in range(5):
        expected_units.append(f"B_pooled_f{f}")
    for p in ALL_PERSONS:
        expected_units.append(f"C_lopo_{p}")

    missing_units = [u for u in expected_units if not (done_dir / f"{u}.done").exists()]
    if missing_units:
        msg = f"Incomplete run: {len(missing_units)} of {len(expected_units)} units are missing (.done not found): {missing_units}"
        if not args.allow_incomplete:
            print(f"\n[ERROR] {msg}")
            print("To preview partial results, pass --allow_incomplete")
            sys.exit(1)
        else:
            print(f"\n[WARNING] {msg}")

    all_preds = []
    unit_stats = {}
    for d in done_files:
        u_id = d.stem
        u_dir = results_dir / u_id
        csv_file = u_dir / "predictions.csv"
        if csv_file.exists():
            with open(csv_file, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                all_preds.extend(rows)
                seeds = {int(r["seed"]) for r in rows}
                kinds = {r["kind"] for r in rows}
                unit_stats[u_id] = {"seeds": sorted(list(seeds)), "kinds": sorted(list(kinds)), "rows": len(rows)}
                if not args.allow_incomplete and (seeds != {0, 1, 2} or kinds != {"both", "kp", "mask"}):
                    print(f"\n[ERROR] Unit {u_id} is incomplete! Found seeds={seeds}, kinds={kinds}")
                    sys.exit(1)

    if not all_preds:
        print("ERROR: No predictions found. Units must be run before generating report.")
        sys.exit(1)

    # Master predictions CSV
    master_csv = results_dir / "predictions.csv"
    with open(master_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_preds[0].keys()))
        w.writeheader()
        w.writerows(all_preds)
    print(f"Master predictions CSV written: {master_csv} ({len(all_preds)} total rows)")

    # Load canonical segments for count statistics
    segs_path = Path(args.segments).expanduser()
    segs_data = {}
    if segs_path.exists():
        segs_data = json.loads(segs_path.read_text())

    new_counts = {p: {"bhu": 0, "lhu": 0, "rhu": 0, "total": 0} for p in ALL_PERSONS}
    for sid, val in segs_data.items():
        p = val.get("person")
        act = val.get("act", "").lower()
        if p in new_counts and act in new_counts[p]:
            c = len(val.get("segments", []))
            new_counts[p][act] += c
            new_counts[p]["total"] += c

    lines = []
    lines.append("=" * 95)
    lines.append("PERCOM BTP: 5-PARTICIPANT FINAL EVALUATION RESULTS (COMPLETE GESTURES ONLY)")
    lines.append("=" * 95)

    # -------------------------------------------------------------------------
    # SEGMENT & FRAME COMPARISON (OLD VS NEW)
    # -------------------------------------------------------------------------
    lines.append("\n" + "=" * 95)
    lines.append("DATA AUDIT: UPDATED MEDIAPIPE SEGMENTS VS. PREVIOUS (OLD) HEURISTIC SEGMENTS")
    lines.append("=" * 95)
    lines.append(f"{'Participant':<12} | {'Old Gestures (BHU/LHU/RHU)':<28} | {'New Gestures (BHU/LHU/RHU)':<28} | {'Difference'}")
    lines.append("-" * 95)
    for p in ALL_PERSONS:
        old_str = f"{OLD_SEGMENTS[p]['total']:3d} ({OLD_SEGMENTS[p]['bhu']}/{OLD_SEGMENTS[p]['lhu']}/{OLD_SEGMENTS[p]['rhu']})" if OLD_SEGMENTS[p]['total'] > 0 else "  0 (new participant)"
        new_str = f"{new_counts[p]['total']:3d} ({new_counts[p]['bhu']}/{new_counts[p]['lhu']}/{new_counts[p]['rhu']})"
        diff_str = f"+{new_counts[p]['total'] - OLD_SEGMENTS[p]['total']:d} gestures" if OLD_SEGMENTS[p]['total'] > 0 else f"+{new_counts[p]['total']:d} gestures (new)"
        lines.append(f"{p:<12} | {old_str:<28} | {new_str:<28} | {diff_str}")
    lines.append("-" * 95)
    lines.append(f"Total Gestures: Old = 1,311 gestures (4 participants) -> New = {sum(new_counts[p]['total'] for p in ALL_PERSONS):,d} gestures (all 5 participants)")
    lines.append("Notice: Priyanshu, Sir, and Vansh were previously under-segmented (~22-40/rec); MediaPipe unified all 5 to ~50-65/rec.")

    # Helper to aggregate metrics across seeds and folds
    # Hierarchy: per (unit, seed, kind) -> mean over seeds -> mean +/- std over folds
    def aggregate_folds(unit_preds_by_fold: Dict[int, List[Dict[str, Any]]], kind: str) -> Dict[str, Tuple[float, float]]:
        fold_metrics = {"acc": [], "f1": [], "prec": [], "rec": []}
        for f, rows in sorted(unit_preds_by_fold.items()):
            kind_rows = [r for r in rows if r["kind"] == kind]
            if not kind_rows:
                continue
            seeds = sorted(list({int(r["seed"]) for r in kind_rows}))
            seed_res = []
            for s in seeds:
                s_rows = [r for r in kind_rows if int(r["seed"]) == s]
                m = calc_metrics([r["y_true"] for r in s_rows], [r["y_pred"] for r in s_rows])
                seed_res.append(m)
            if seed_res:
                for k in ["acc", "f1", "prec", "rec"]:
                    fold_metrics[k].append(float(np.mean([sr[k] for sr in seed_res])))
        out = {}
        for k in ["acc", "f1", "prec", "rec"]:
            vals = fold_metrics[k]
            out[k] = (float(np.mean(vals) * 100), float(np.std(vals) * 100)) if vals else (0.0, 0.0)
        return out

    # -------------------------------------------------------------------------
    # PROTOCOL A: PARTICIPANT-WISE (5 Folds x 5 Persons = 25 Units)
    # -------------------------------------------------------------------------
    lines.append("\n" + "=" * 95)
    lines.append("TABLE T1: PROTOCOL A (PARTICIPANT-WISE / INDIVIDUALLY CALIBRATED)")
    lines.append("HEADLINE: Joints + Speed (kp). All 4 metrics computed from the EXACT same predictions.")
    lines.append("=" * 95)
    lines.append(f"{'Participant':<12} | {'Accuracy (Joints+Spd)':<22} | {'Macro F1':<10} | {'Macro Prec':<12} | {'Macro Rec':<10} | {'Outline+Spd':<12} | {'Both+Spd'}")
    lines.append("-" * 95)

    proto_a_results = {}
    for p in ALL_PERSONS:
        p_folds = defaultdict(list)
        for r in all_preds:
            if r["protocol"] == "A" and r["person"] == p:
                p_folds[int(r["fold"])].append(r)
        
        m_kp = aggregate_folds(p_folds, "kp")
        m_mask = aggregate_folds(p_folds, "mask")
        m_both = aggregate_folds(p_folds, "both")
        proto_a_results[p] = {"kp": m_kp, "mask": m_mask, "both": m_both}
        lines.append(f"{p:<12} | {m_kp['acc'][0]:5.1f}% +- {m_kp['acc'][1]:4.1f}%       | {m_kp['f1'][0]:5.1f}%    | {m_kp['prec'][0]:5.1f}%      | {m_kp['rec'][0]:5.1f}%    | {m_mask['acc'][0]:5.1f}%       | {m_both['acc'][0]:5.1f}%")

    lines.append("-" * 95)
    mean_a_kp_acc = float(np.mean([proto_a_results[p]["kp"]["acc"][0] for p in ALL_PERSONS]))
    std_a_kp_acc = float(np.std([proto_a_results[p]["kp"]["acc"][0] for p in ALL_PERSONS]))
    mean_a_kp_f1 = float(np.mean([proto_a_results[p]["kp"]["f1"][0] for p in ALL_PERSONS]))
    mean_a_kp_pr = float(np.mean([proto_a_results[p]["kp"]["prec"][0] for p in ALL_PERSONS]))
    mean_a_kp_rc = float(np.mean([proto_a_results[p]["kp"]["rec"][0] for p in ALL_PERSONS]))
    mean_a_mask_acc = float(np.mean([proto_a_results[p]["mask"]["acc"][0] for p in ALL_PERSONS]))
    mean_a_both_acc = float(np.mean([proto_a_results[p]["both"]["acc"][0] for p in ALL_PERSONS]))
    lines.append(f"{'MEAN (All 5)':<12} | {mean_a_kp_acc:5.1f}% +- {std_a_kp_acc:4.1f}%       | {mean_a_kp_f1:5.1f}%    | {mean_a_kp_pr:5.1f}%      | {mean_a_kp_rc:5.1f}%    | {mean_a_mask_acc:5.1f}%       | {mean_a_both_acc:5.1f}%")

    # -------------------------------------------------------------------------
    # PROTOCOL B: POOLED (5 Folds = 5 Units)
    # -------------------------------------------------------------------------
    lines.append("\n" + "=" * 95)
    lines.append("TABLE T2: PROTOCOL B (POOLED 5 PARTICIPANTS)")
    lines.append("One model trained on pooled data from all 5 participants (Complete Gestures)")
    lines.append("=" * 95)
    lines.append(f"{'Slice':<12} | {'Accuracy (Joints+Spd)':<22} | {'Macro F1':<10} | {'Macro Prec':<12} | {'Macro Rec':<10} | {'Outline+Spd':<12} | {'Both+Spd'}")
    lines.append("-" * 95)

    b_folds = defaultdict(list)
    for r in all_preds:
        if r["protocol"] == "B":
            b_folds[int(r["fold"])].append(r)

    b_kp = aggregate_folds(b_folds, "kp")
    b_mask = aggregate_folds(b_folds, "mask")
    b_both = aggregate_folds(b_folds, "both")

    lines.append(f"{'Overall Mean':<12} | {b_kp['acc'][0]:5.1f}% +- {b_kp['acc'][1]:4.1f}%       | {b_kp['f1'][0]:5.1f}%    | {b_kp['prec'][0]:5.1f}%      | {b_kp['rec'][0]:5.1f}%    | {b_mask['acc'][0]:5.1f}%       | {b_both['acc'][0]:5.1f}%")
    lines.append("-" * 95)
    lines.append("Breakdown by participant inside pooled predictions (Joints + Speed):")
    for p in ALL_PERSONS:
        p_sub = [r for r in all_preds if r["protocol"] == "B" and r["person"] == p and r["kind"] == "kp"]
        if p_sub:
            p_acc = calc_metrics([r["y_true"] for r in p_sub], [r["y_pred"] for r in p_sub])["acc"] * 100
            lines.append(f"  {p:<12}: {p_acc:5.1f}%")

    # -------------------------------------------------------------------------
    # PROTOCOL C: LEAVE-ONE-PERSON-OUT (LOPO) (5 Folds = 5 Units)
    # -------------------------------------------------------------------------
    lines.append("\n" + "=" * 95)
    lines.append("TABLE T3: PROTOCOL C (LEAVE-ONE-PERSON-OUT / UNCALIBRATED)")
    lines.append("Trained on 4 participants, evaluated on the 5th held-out participant (Complete Gestures)")
    lines.append("=" * 95)
    lines.append(f"{'Held-Out':<12} | {'Accuracy (Joints+Spd)':<22} | {'Macro F1':<10} | {'Macro Prec':<12} | {'Macro Rec':<10} | {'Outline+Spd':<12} | {'Both+Spd'}")
    lines.append("-" * 95)

    c_results = {}
    for p in ALL_PERSONS:
        sub_c = [r for r in all_preds if r["protocol"] == "C" and r["person"] == p]
        c_p_folds = {0: sub_c}
        m_kp = aggregate_folds(c_p_folds, "kp")
        m_mask = aggregate_folds(c_p_folds, "mask")
        m_both = aggregate_folds(c_p_folds, "both")
        c_results[p] = {"kp": m_kp, "mask": m_mask, "both": m_both}
        lines.append(f"{p:<12} | {m_kp['acc'][0]:5.1f}%                 | {m_kp['f1'][0]:5.1f}%    | {m_kp['prec'][0]:5.1f}%      | {m_kp['rec'][0]:5.1f}%    | {m_mask['acc'][0]:5.1f}%       | {m_both['acc'][0]:5.1f}%")

    lines.append("-" * 95)
    mean_c_kp_acc = float(np.mean([c_results[p]["kp"]["acc"][0] for p in ALL_PERSONS]))
    std_c_kp_acc = float(np.std([c_results[p]["kp"]["acc"][0] for p in ALL_PERSONS]))
    mean_c_kp_f1 = float(np.mean([c_results[p]["kp"]["f1"][0] for p in ALL_PERSONS]))
    mean_c_kp_pr = float(np.mean([c_results[p]["kp"]["prec"][0] for p in ALL_PERSONS]))
    mean_c_kp_rc = float(np.mean([c_results[p]["kp"]["rec"][0] for p in ALL_PERSONS]))
    mean_c_mask_acc = float(np.mean([c_results[p]["mask"]["acc"][0] for p in ALL_PERSONS]))
    mean_c_both_acc = float(np.mean([c_results[p]["both"]["acc"][0] for p in ALL_PERSONS]))
    lines.append(f"{'MEAN (All 5)':<12} | {mean_c_kp_acc:5.1f}% +- {std_c_kp_acc:4.1f}%       | {mean_c_kp_f1:5.1f}%    | {mean_c_kp_pr:5.1f}%      | {mean_c_kp_rc:5.1f}%    | {mean_c_mask_acc:5.1f}%       | {mean_c_both_acc:5.1f}%")

    # -------------------------------------------------------------------------
    # TABLE T4: PROTOCOL COMPARISON (A vs B vs C)
    # -------------------------------------------------------------------------
    lines.append("\n" + "=" * 95)
    lines.append("TABLE T4: OVERALL COMPARISON OF PROTOCOLS A, B, AND C")
    lines.append("=" * 95)
    lines.append(f"{'Protocol':<34} | {'Accuracy (Joints+Spd)':<22} | {'Macro F1':<10} | {'Both (Mask+Joints+Spd)'}")
    lines.append("-" * 95)
    lines.append(f"{'Protocol A (Participant-wise)':<34} | {mean_a_kp_acc:5.1f}% +- {std_a_kp_acc:4.1f}%       | {mean_a_kp_f1:5.1f}%    | {mean_a_both_acc:5.1f}%")
    lines.append(f"{'Protocol B (Pooled Calibration)':<34} | {b_kp['acc'][0]:5.1f}% +- {b_kp['acc'][1]:4.1f}%       | {b_kp['f1'][0]:5.1f}%    | {b_both['acc'][0]:5.1f}%")
    lines.append(f"{'Protocol C (Leave-One-Person-Out)':<34} | {mean_c_kp_acc:5.1f}% +- {std_c_kp_acc:4.1f}%       | {mean_c_kp_f1:5.1f}%    | {mean_c_both_acc:5.1f}%")
    lines.append("=" * 95)

    # Footnote on macro vs micro
    lines.append("\nNote: Macro metrics compute precision/recall/F1 separately per class and average unweighted.")
    lines.append("Chance accuracy for 3 balanced gesture classes is 33.3%.")

    full_text = "\n".join(lines)
    print("\n" + full_text)

    # Save summary report
    summary_path = results_dir / "FINAL_SUMMARY.md"
    summary_path.write_text(full_text + "\n")
    print(f"\nSaved final summary report: {summary_path}")

    # Generate Publication Confusion Matrices for Headline Input (Joints + Speed)
    for proto, p_name in [("A", "Protocol A (Within-Person)"), ("B", "Protocol B (Pooled)"), ("C", "Protocol C (LOPO)")]:
        sub = [r for r in all_preds if r["protocol"] == proto and r["kind"] == "kp"]
        if not sub:
            continue
        cm = np.zeros((3, 3))
        for r in sub:
            t_idx = DISPLAY_ACTS.index(INTERNAL_MAP[r["y_true"].lower()])
            p_idx = DISPLAY_ACTS.index(INTERNAL_MAP[r["y_pred"].lower()])
            cm[t_idx, p_idx] += 1
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6) * 100.0
        cm_out = results_dir / f"cm_{proto.lower()}_kp.png"
        plot_publication_cm(cm_pct, f"{p_name}\nComplete Gestures (Joints + Speed)", cm_out)
        print(f"Saved publication confusion matrix: {cm_out}")


if __name__ == "__main__":
    main()
