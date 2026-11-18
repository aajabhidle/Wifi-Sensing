import csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

with open("E:/percom/results_5p/predictions.csv") as f:
    rows = list(csv.DictReader(f))

ALL_PERSONS = ["Priyanshu", "Sanchit", "Sir", "Vansh", "Vardaan"]

def calc_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0)
    }

def aggregate_folds(unit_preds_by_fold, kind):
    fold_metrics = {"acc": [], "f1": [], "prec": [], "rec": []}
    for f, frows in sorted(unit_preds_by_fold.items()):
        kind_rows = [r for r in frows if r["kind"] == kind]
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

print("=" * 80)
print("TABLE T1 (PROTOCOL A): BOTH + SPEED (OUTLINE + JOINTS + SPEED)")
print("=" * 80)
print(f"{'Participant':<12} | {'Accuracy (Both+Spd)':<22} | {'Macro F1':<10} | {'Macro Prec':<12} | {'Macro Rec':<10}")
print("-" * 80)

both_results = {}
for p in ALL_PERSONS:
    p_folds = defaultdict(list)
    for r in rows:
        if r["protocol"] == "A" and r["person"] == p:
            p_folds[int(r["fold"])].append(r)
    m = aggregate_folds(p_folds, "both")
    both_results[p] = m
    print(f"{p:<12} | {m['acc'][0]:5.1f}% +- {m['acc'][1]:4.1f}%       | {m['f1'][0]:5.1f}%    | {m['prec'][0]:5.1f}%      | {m['rec'][0]:5.1f}%")

print("-" * 80)
mean_acc = np.mean([both_results[p]["acc"][0] for p in ALL_PERSONS])
std_acc = np.std([both_results[p]["acc"][0] for p in ALL_PERSONS])
mean_f1 = np.mean([both_results[p]["f1"][0] for p in ALL_PERSONS])
mean_pr = np.mean([both_results[p]["prec"][0] for p in ALL_PERSONS])
mean_rc = np.mean([both_results[p]["rec"][0] for p in ALL_PERSONS])
print(f"{'MEAN (All 5)':<12} | {mean_acc:5.1f}% +- {std_acc:4.1f}%       | {mean_f1:5.1f}%    | {mean_pr:5.1f}%      | {mean_rc:5.1f}%")
print("=" * 80)
