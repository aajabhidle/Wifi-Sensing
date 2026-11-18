"""Regenerate all confusion matrix PNGs from the verified predictions.csv"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = "E:/percom/results_5p"

with open(f"{results_dir}/predictions.csv") as f:
    rows = list(csv.DictReader(f))

acts = ["bhu", "lhu", "rhu"]
DISPLAY = ["BHR", "LHR", "RHR"]

def plot_cm(cm_pct, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percentage (%)", fontsize=14, labelpad=8)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(DISPLAY, fontsize=14, fontweight="bold")
    ax.set_yticklabels(DISPLAY, fontsize=14, fontweight="bold")
    thresh = 50.0
    for i in range(3):
        for j in range(3):
            v = cm_pct[i, j]
            color = "white" if v > thresh else "black"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=color, fontsize=16, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=14)
    ax.set_ylabel("True Gesture", fontsize=14, fontweight="bold", labelpad=8)
    ax.set_xlabel("Predicted Gesture", fontsize=14, fontweight="bold", labelpad=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# Generate individual CMs for each (protocol, kind) combo
for proto, proto_name in [("A", "Protocol A (Within-Person)"), ("B", "Protocol B (Pooled)"), ("C", "Protocol C (LOPO)")]:
    for kind, kind_label in [("kp", "Joints + Speed"), ("both", "Outline + Joints + Speed")]:
        sub = [r for r in rows if r["protocol"] == proto and r["kind"] == kind]
        if not sub:
            continue
        cm = np.zeros((3, 3))
        for r in sub:
            t = acts.index(r["y_true"].lower())
            p = acts.index(r["y_pred"].lower())
            cm[t, p] += 1
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6) * 100.0
        total = int(cm.sum())
        correct = int(np.trace(cm))
        acc = correct / total * 100
        title = f"{proto_name}\nComplete Gestures ({kind_label})"
        out = f"{results_dir}/cm_{proto.lower()}_{kind}.png"
        plot_cm(cm_pct, title, out)
        print(f"  Verified: {correct}/{total} = {acc:.2f}%  diag=[{cm_pct[0,0]:.1f}, {cm_pct[1,1]:.1f}, {cm_pct[2,2]:.1f}]")

# Generate triptych (3-panel side-by-side) for both kinds
for kind, kind_label in [("both", "Outline + Joints + Speed"), ("kp", "Joints + Speed")]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    titles = [
        ("A", "Protocol A\n(Within-Person)"),
        ("B", "Protocol B\n(Pooled)"),
        ("C", "Protocol C\n(Leave-One-Out)")
    ]
    for idx, (proto, title) in enumerate(titles):
        ax = axes[idx]
        sub = [r for r in rows if r["protocol"] == proto and r["kind"] == kind]
        cm = np.zeros((3, 3))
        for r in sub:
            t = acts.index(r["y_true"].lower())
            p = acts.index(r["y_pred"].lower())
            cm[t, p] += 1
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6) * 100.0
        im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(DISPLAY, fontsize=14, fontweight="bold")
        ax.set_yticklabels(DISPLAY, fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("True Gesture", fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Gesture", fontsize=13, fontweight="bold")
        for i in range(3):
            for j in range(3):
                v = cm_pct[i, j]
                color = "white" if v > 50 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color=color, fontsize=15, fontweight="bold")
    plt.suptitle(f"Wi-Fi CSI Gesture Recognition — {kind_label}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = f"{results_dir}/cm_triptych_{kind}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved triptych: {out}")

print("\nDone! All CMs regenerated from verified predictions.csv")
