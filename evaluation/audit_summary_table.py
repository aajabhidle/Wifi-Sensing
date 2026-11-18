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

print("=" * 90)
print("VERIFICATION OF TABLE T1 (PROTOCOL A)")
print("=" * 90)
p_res = {}
for p in ALL_PERSONS:
    p_folds = defaultdict(list)
    for r in rows:
        if r["protocol"] == "A" and r["person"] == p:
            p_folds[int(r["fold"])].append(r)
    m_kp = aggregate_folds(p_folds, "kp")
    m_mask = aggregate_folds(p_folds, "mask")
    m_both = aggregate_folds(p_folds, "both")
    p_res[p] = (m_kp, m_mask, m_both)
    print(f"{p:<10s} | kp: {m_kp['acc'][0]:.1f}% +- {m_kp['acc'][1]:.1f}% | F1: {m_kp['f1'][0]:.1f}% | Prec: {m_kp['prec'][0]:.1f}% | Rec: {m_kp['rec'][0]:.1f}% | Outline: {m_mask['acc'][0]:.1f}% | Both: {m_both['acc'][0]:.1f}%")

m_a_kp_acc = np.mean([p_res[p][0]["acc"][0] for p in ALL_PERSONS])
s_a_kp_acc = np.std([p_res[p][0]["acc"][0] for p in ALL_PERSONS])
m_a_f1 = np.mean([p_res[p][0]["f1"][0] for p in ALL_PERSONS])
m_a_pr = np.mean([p_res[p][0]["prec"][0] for p in ALL_PERSONS])
m_a_rc = np.mean([p_res[p][0]["rec"][0] for p in ALL_PERSONS])
m_a_mask = np.mean([p_res[p][1]["acc"][0] for p in ALL_PERSONS])
m_a_both = np.mean([p_res[p][2]["acc"][0] for p in ALL_PERSONS])
print("-" * 90)
print(f"MEAN (All 5)| kp: {m_a_kp_acc:.1f}% +- {s_a_kp_acc:.1f}% | F1: {m_a_f1:.1f}% | Prec: {m_a_pr:.1f}% | Rec: {m_a_rc:.1f}% | Outline: {m_a_mask:.1f}% | Both: {m_a_both:.1f}%")

print("\n" + "=" * 90)
print("VERIFICATION OF TABLE T2 (PROTOCOL B)")
print("=" * 90)
b_folds = defaultdict(list)
for r in rows:
    if r["protocol"] == "B":
        b_folds[int(r["fold"])].append(r)
b_kp = aggregate_folds(b_folds, "kp")
b_mask = aggregate_folds(b_folds, "mask")
b_both = aggregate_folds(b_folds, "both")
print(f"Overall Mean| kp: {b_kp['acc'][0]:.1f}% +- {b_kp['acc'][1]:.1f}% | F1: {b_kp['f1'][0]:.1f}% | Prec: {b_kp['prec'][0]:.1f}% | Rec: {b_kp['rec'][0]:.1f}% | Outline: {b_mask['acc'][0]:.1f}% | Both: {b_both['acc'][0]:.1f}%")

print("\nBreakdown by person in Protocol B (Joints + Speed):")
for p in ALL_PERSONS:
    sub = [r for r in rows if r["protocol"] == "B" and r["person"] == p and r["kind"] == "kp"]
    acc = accuracy_score([r["y_true"] for r in sub], [r["y_pred"] for r in sub]) * 100
    print(f"  {p:<10s}: {acc:.1f}%")

print("\n" + "=" * 90)
print("VERIFICATION OF TABLE T3 (PROTOCOL C)")
print("=" * 90)
c_res = {}
for p in ALL_PERSONS:
    sub_c = [r for r in rows if r["protocol"] == "C" and r["person"] == p]
    c_p_folds = {0: sub_c}
    m_kp = aggregate_folds(c_p_folds, "kp")
    m_mask = aggregate_folds(c_p_folds, "mask")
    m_both = aggregate_folds(c_p_folds, "both")
    c_res[p] = (m_kp, m_mask, m_both)
    print(f"{p:<10s} | kp: {m_kp['acc'][0]:.1f}% | F1: {m_kp['f1'][0]:.1f}% | Prec: {m_kp['prec'][0]:.1f}% | Rec: {m_kp['rec'][0]:.1f}% | Outline: {m_mask['acc'][0]:.1f}% | Both: {m_both['acc'][0]:.1f}%")

m_c_kp_acc = np.mean([c_res[p][0]["acc"][0] for p in ALL_PERSONS])
s_c_kp_acc = np.std([c_res[p][0]["acc"][0] for p in ALL_PERSONS])
m_c_f1 = np.mean([c_res[p][0]["f1"][0] for p in ALL_PERSONS])
m_c_pr = np.mean([c_res[p][0]["prec"][0] for p in ALL_PERSONS])
m_c_rc = np.mean([c_res[p][0]["rec"][0] for p in ALL_PERSONS])
m_c_mask = np.mean([c_res[p][1]["acc"][0] for p in ALL_PERSONS])
m_c_both = np.mean([c_res[p][2]["acc"][0] for p in ALL_PERSONS])
print("-" * 90)
print(f"MEAN (All 5)| kp: {m_c_kp_acc:.1f}% +- {s_c_kp_acc:.1f}% | F1: {m_c_f1:.1f}% | Prec: {m_c_pr:.1f}% | Rec: {m_c_rc:.1f}% | Outline: {m_c_mask:.1f}% | Both: {m_c_both:.1f}%")
