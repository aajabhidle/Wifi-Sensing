#!/usr/bin/env python3
"""
replot_cm.py - Re-generates publication-quality confusion matrices from saved predictions:
1. Labels in all CAPS: BHR, LHR, RHR
2. Increased font sizes for titles, axis labels, tick labels, colorbar, and cell values.
3. Row-normalized average percentages (rows sum to 100%) and count matrices.
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mapping from internal labels to presentation labels
ACTS_ORIG = ["bhu", "lhu", "rhu"]
ACTS_DISPLAY = ["BHR", "LHR", "RHR"]  # Both Hands Raise, Left Hand Raise, Right Hand Raise


def plot_confusion_matrix(cm, title, out_path, normalize=False):
    # Generous figure size with high resolution for presentations/reports
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
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
        cbar.set_label("Total Counts across Seeds", fontsize=13, fontweight="bold", labelpad=10)

    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(ACTS_DISPLAY, fontsize=13, fontweight="bold")
    ax.set_yticklabels(ACTS_DISPLAY, fontsize=13, fontweight="bold")
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
    print(f"Generated: {out_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="scenario2_pooled", help="Directory with predictions_gestures.csv")
    args = ap.parse_args()
    d = Path(args.dir)
    pred_file = d / "predictions_gestures.csv"
    if not pred_file.exists():
        raise SystemExit(f"Error: {pred_file} not found. Please download predictions_gestures.csv from server first.")

    # Read predictions grouped by feature type
    data_by_feat = {
        "kp": np.zeros((3, 3), dtype=int),
        "mask": np.zeros((3, 3), dtype=int),
        "both": np.zeros((3, 3), dtype=int)
    }
    
    with open(pred_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat = row["input_type"]
            if feat in data_by_feat:
                y_true = ACTS_ORIG.index(row["true_label"])
                y_pred = ACTS_ORIG.index(row["predicted_label"])
                data_by_feat[feat][y_true, y_pred] += 1

    feature_labels = {
        "kp": "Joints + Speed",
        "mask": "Outline + Speed",
        "both": "Joints + Outline + Speed"
    }

    print("\n--- Generating Updated Confusion Matrices ---")
    for feat, cm in data_by_feat.items():
        fname = feature_labels[feat]
        title_count = f"Confusion Matrix: Gestures ({fname})"
        title_norm = f"Average Confusion Matrix: Gestures ({fname})"
        
        plot_confusion_matrix(cm, title_count, d / f"confusion_matrix_gestures_{feat}.png", normalize=False)
        plot_confusion_matrix(cm, title_norm, d / f"confusion_matrix_gestures_{feat}_norm.png", normalize=True)

    # Main publication figure for joints + speed
    plot_confusion_matrix(
        data_by_feat["kp"],
        "Overall Performance: Gestures (Joints + Speed)",
        d / "confusion_matrix_main_gestures_joints.png",
        normalize=True
    )
    print("\nAll confusion matrices updated with larger fonts and BHR, LHR, RHR labels!")


if __name__ == "__main__":
    main()
