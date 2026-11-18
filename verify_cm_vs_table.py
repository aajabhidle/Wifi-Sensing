import csv
import numpy as np

with open("E:/percom/results_5p/predictions.csv") as f:
    rows = list(csv.DictReader(f))

acts = ["bhu", "lhu", "rhu"]
labels = ["BHR", "LHR", "RHR"]

for proto, proto_name in [("A", "Protocol A"), ("B", "Protocol B"), ("C", "Protocol C")]:
    for kind in ["kp", "both"]:
        sub = [r for r in rows if r["protocol"] == proto and r["kind"] == kind]
        if not sub:
            continue
        seeds = sorted({int(r["seed"]) for r in sub})

        print(f"\n{'='*70}")
        print(f"{proto_name} | kind={kind} | {len(sub)} prediction rows | seeds={seeds}")
        print(f"{'='*70}")

        # Per-seed confusion matrices and accuracy
        for s in seeds:
            s_rows = [r for r in sub if int(r["seed"]) == s]
            cm = np.zeros((3, 3), dtype=int)
            for r in s_rows:
                t = acts.index(r["y_true"].lower())
                p = acts.index(r["y_pred"].lower())
                cm[t, p] += 1
            total = cm.sum()
            correct = np.trace(cm)
            acc = correct / total * 100 if total > 0 else 0
            print(f"  Seed {s}: {correct}/{total} = {acc:.2f}%")

        # Aggregated CM across all seeds
        cm_all = np.zeros((3, 3), dtype=int)
        for r in sub:
            t = acts.index(r["y_true"].lower())
            p = acts.index(r["y_pred"].lower())
            cm_all[t, p] += 1
        cm_pct = cm_all / np.maximum(cm_all.sum(axis=1, keepdims=True), 1) * 100

        total_all = cm_all.sum()
        correct_all = np.trace(cm_all)
        acc_all = correct_all / total_all * 100

        print(f"\n  Aggregated CM (all {len(seeds)} seeds pooled):")
        print(f"  {'':12s} {'BHR':>8s} {'LHR':>8s} {'RHR':>8s}  |  Row Total")
        for i in range(3):
            row_total = cm_all[i].sum()
            print(f"  {labels[i]:<12s} {cm_all[i,0]:8d} {cm_all[i,1]:8d} {cm_all[i,2]:8d}  |  {row_total}")
        print(f"  Raw accuracy from CM: {correct_all}/{total_all} = {acc_all:.2f}%")
        print(f"  Diagonal percentages: BHR={cm_pct[0,0]:.1f}%, LHR={cm_pct[1,1]:.1f}%, RHR={cm_pct[2,2]:.1f}%")

        # Mean-of-seed-accuracies (what the table reports for Protocol A)
        seed_accs = []
        for s in seeds:
            s_rows = [r for r in sub if int(r["seed"]) == s]
            correct_s = sum(1 for r in s_rows if r["y_true"] == r["y_pred"])
            seed_accs.append(correct_s / len(s_rows) * 100)
        print(f"  Mean of per-seed accuracies: {np.mean(seed_accs):.2f}% +- {np.std(seed_accs):.2f}%")

        # For Protocol A: mean-of-fold-mean-of-seed (the actual table computation)
        if proto == "A":
            persons = sorted({r["person"] for r in sub})
            for p in persons:
                p_rows = [r for r in sub if r["person"] == p]
                folds = sorted({int(r["fold"]) for r in p_rows})
                fold_accs = []
                for fd in folds:
                    f_rows = [r for r in p_rows if int(r["fold"]) == fd]
                    f_seeds = sorted({int(r["seed"]) for r in f_rows})
                    s_accs = []
                    for s in f_seeds:
                        sr = [r for r in f_rows if int(r["seed"]) == s]
                        s_accs.append(sum(1 for r in sr if r["y_true"] == r["y_pred"]) / len(sr) * 100)
                    fold_accs.append(np.mean(s_accs))
                print(f"    {p}: fold_accs={[f'{x:.1f}' for x in fold_accs]}, mean={np.mean(fold_accs):.2f}%")
