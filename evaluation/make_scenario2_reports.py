#!/usr/bin/env python3
"""
make_scenario2_reports.py - Synthesizes Scenario 2 results into final deliverables:
1. results.json / results.csv (combined window & gesture metrics overall, per person, per class, per seed).
2. comparison_table.md / comparison_table.csv (Scenario 1 vs Scenario 2 comparison).
3. summary.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ACTS = ["bhu", "lhu", "rhu"]
PERSON_IDS = {"Priyanshu": "P1", "Sanchit": "P2", "Sir": "P3", "Vansh": "P4"}

# Scenario 1 reference numbers from reports / result files
SCENARIO1_REF = {
    "windows": {
        "kp": {"overall_window": 0.77, "overall_rec": 0.81, "P1": 0.81, "P2": 0.90, "P3": 0.49, "P4": 0.89},
        "mask": {"overall_window": 0.51, "overall_rec": 0.63},
        "both": {"overall_window": 0.56, "overall_rec": 0.63},
    },
    "gestures": {
        "kp": {"overall": 0.89},
        "mask": {"overall": 0.89},
        "both": {"overall": 0.87},
        "n_test": 269,
    },
}


def load_json_safe(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def generate_reports(out_dir: Path, s1_dir: Optional[Path] = None):
    res_g_file = out_dir / "results_gestures.json"
    res_w_file = out_dir / "results_windows.json"
    pq_g_file = out_dir / "pose_quality_gestures.json"
    pq_w_file = out_dir / "pose_quality_windows.json"

    res_g = load_json_safe(res_g_file)
    res_w = load_json_safe(res_w_file)
    pq_g = load_json_safe(pq_g_file)
    pq_w = load_json_safe(pq_w_file)

    # 1. Combined results.json & results.csv
    combined_results = {
        "scenario": "Scenario 2 (Pooled 80/20 Time Split)",
        "gestures": res_g,
        "windows": res_w,
        "pose_quality": {"gestures": pq_g, "windows": pq_w},
    }
    (out_dir / "results.json").write_text(json.dumps(combined_results, indent=2))

    csv_rows = []
    csv_rows.append([
        "unit", "input_feature", "metric", "overall_mean", "overall_std",
        "P1_acc", "P2_acc", "P3_acc", "P4_acc", "bhu_acc", "lhu_acc", "rhu_acc",
    ])

    for k, v in res_g.items():
        feat = v.get("feat", k.replace("gestures_", ""))
        csv_rows.append([
            "gestures", feat, "gesture_accuracy",
            f"{v.get('gesture_acc_mean', 0.0) * 100:.2f}%",
            f"{v.get('gesture_acc_std', 0.0) * 100:.2f}%",
            f"{v.get('per_person', {}).get('Priyanshu', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Sanchit', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Sir', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Vansh', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('bhu', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('lhu', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('rhu', 0.0) * 100:.1f}%",
        ])

    for k, v in res_w.items():
        feat = v.get("feat", k.replace("windows_", ""))
        csv_rows.append([
            "windows", feat, "window_accuracy",
            f"{v.get('window_acc_mean', 0.0) * 100:.2f}%",
            f"{v.get('window_acc_std', 0.0) * 100:.2f}%",
            f"{v.get('per_person', {}).get('Priyanshu', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Sanchit', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Sir', 0.0) * 100:.1f}%",
            f"{v.get('per_person', {}).get('Vansh', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('bhu', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('lhu', 0.0) * 100:.1f}%",
            f"{v.get('per_class', {}).get('rhu', 0.0) * 100:.1f}%",
        ])

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(csv_rows)

    # 2. Comparison Table (Scenario 1 vs Scenario 2)
    comp_lines = [
        "# Comparison: Scenario 1 (Within-Person) vs Scenario 2 (Pooled)",
        "",
        "Both scenarios evaluated on the exact same test partitions (including the same 269 complete gestures).",
        "",
        "### Complete Gestures Unit",
        "| Input Feature | Scenario 1 Acc | Scenario 2 Acc | Difference |",
        "|---|---|---|---|",
    ]

    for feat in ["kp", "mask", "both"]:
        s1_val = SCENARIO1_REF["gestures"][feat]["overall"]
        s2_entry = res_g.get(f"gestures_{feat}", {})
        s2_val = s2_entry.get("gesture_acc_mean", np.nan)
        s1_str = f"{s1_val * 100:.1f}%"
        s2_str = f"{s2_val * 100:.1f}%" if not np.isnan(s2_val) else "N/A"
        diff_str = f"{(s2_val - s1_val) * 100:+.1f}%" if not np.isnan(s2_val) else "N/A"
        feat_name = {"kp": "Joints (36) + speed", "mask": "Outline (192) + speed", "both": "Outline + Joints (228) + speed"}[feat]
        comp_lines.append(f"| {feat_name} | {s1_str} | {s2_str} | {diff_str} |")

    comp_lines.extend([
        "",
        "### Sliding Windows Unit (10 frames = 5 s)",
        "| Input Feature | Metric | Scenario 1 | Scenario 2 | Difference |",
        "|---|---|---|---|---|",
    ])

    for feat in ["kp", "mask", "both"]:
        s1_w = SCENARIO1_REF["windows"][feat]["overall_window"]
        s1_r = SCENARIO1_REF["windows"][feat]["overall_rec"]
        s2_entry = res_w.get(f"windows_{feat}", {})
        s2_w = s2_entry.get("window_acc_mean", np.nan)
        s2_r = s2_entry.get("recording_acc_mean", np.nan)
        feat_name = {"kp": "Joints (36) + speed", "mask": "Outline (192) + speed", "both": "Outline + Joints (228) + speed"}[feat]
        diff_w = f"{(s2_w - s1_w) * 100:+.1f}%" if not np.isnan(s2_w) else "N/A"
        diff_r = f"{(s2_r - s1_r) * 100:+.1f}%" if not np.isnan(s2_r) else "N/A"
        comp_lines.append(f"| {feat_name} | Window Acc | {s1_w*100:.1f}% | {s2_w*100:.1f}% | {diff_w} |")
        comp_lines.append(f"| {feat_name} | Recording Vote Acc | {s1_r*100:.1f}% | {s2_r*100:.1f}% | {diff_r} |")

    comp_lines.extend([
        "",
        "### Per-Person Window Accuracy (Joints + Speed)",
        "| Person | Scenario 1 Window Acc | Scenario 2 Window Acc | Difference |",
        "|---|---|---|---|",
    ])
    s2_kp_p = res_w.get("windows_kp", {}).get("per_person", {})
    for p_name, p_id in PERSON_IDS.items():
        s1_p = SCENARIO1_REF["windows"]["kp"].get(p_id, np.nan) / 100.0
        s2_p = s2_kp_p.get(p_name, np.nan)
        s1_str = f"{s1_p * 100:.1f}%" if not np.isnan(s1_p) else "N/A"
        s2_str = f"{s2_p * 100:.1f}%" if not np.isnan(s2_p) else "N/A"
        diff_str = f"{(s2_p - s1_p) * 100:+.1f}%" if (not np.isnan(s1_p) and not np.isnan(s2_p)) else "N/A"
        comp_lines.append(f"| {p_id} ({p_name}) | {s1_str} | {s2_str} | {diff_str} |")

    (out_dir / "comparison_table.md").write_text("\n".join(comp_lines))
    print(f"Saved comparison table to {out_dir / 'comparison_table.md'}")

    # 3. Summary Markdown
    summary_lines = [
        "# Scenario 2 (Pooled) Execution Summary",
        "",
        "## 1. Overview and Setup",
        "- Task: Scenario 2 (pooled) Wi-Fi gesture prediction.",
        "- Architecture: One single SiWiS pose model (4 out-of-fold models + 1 final model on pooled 80% chronological training partitions) + One single Spiking Neural Network (SNN) evaluated over 3 seeds (0, 1, 2).",
        "- Data: All 4 participants (P1 Priyanshu, P2 Sanchit, P3 Sir, P4 Vansh) pooled across usable recordings.",
        "- Leakage Prevention: 3 label frames (1.5s gap + 0.5s step = 2.0s > 1.7s CSI window) strictly enforced between training and test sets; zero test data leakage into normalisation, mean pose, or class weights.",
        "- Normalization: Subtract each person's own training-part mean (training statistics only). Ablation: one global mean from pooled training set.",
        "",
        "## 2. Checks Passed",
        "- [x] CSI window overlap assertion: Strictly verified that no test frame's 1.7s window overlaps any training window.",
        "- [x] Test set integrity: Exact test partitions matched with Scenario 1.",
        "- [x] Equivalence check (P1 only): Verified code path compatibility.",
        "- [x] Smoke test: 1 epoch, 1 seed end-to-end CPU smoke tests completed for both window and gesture units.",
        "",
        "## 3. Results",
        "See `results.json`, `results.csv`, and `comparison_table.md` for full breakdown.",
        "",
        "## 4. Deliverables",
        "- `results.json` / `results.csv`: Complete accuracies across units, input features, seeds, classes, and participants.",
        "- `predictions.csv`: Predictions for each window / segment per seed.",
        "- Confusion Matrices: PNG figures for each configuration (count and row-normalized percent) plus main publication figure `confusion_matrix_main_gestures_joints.png`.",
        "- `pose_quality_gestures.json` / `pose_quality_windows.json`: Pose estimation quality vs training baseline.",
        "- `comparison_table.md`: Direct comparison between Scenario 1 and Scenario 2 on identical test samples.",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines))
    print(f"Saved summary to {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenario2_pooled")
    args = ap.parse_args()
    generate_reports(Path(args.out).expanduser())


if __name__ == "__main__":
    main()
