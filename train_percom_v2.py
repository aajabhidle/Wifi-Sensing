#!/usr/bin/env python3
"""
train_percom_v2.py - SiWiS trained again, with the two problems of the first run fixed (repo code untouched).

What changed against train_percom.py
  1. LOSS BALANCE.  The repo's mask loss SUMS the error over every pixel (values of thousands) while the keypoint loss is a
     mean (about 0.002), so the keypoint branch hardly influenced training.  Here the mask loss is divided by the number of
     pixels and the keypoint loss is multiplied by --pose_weight (default 500), so both matter.
  2. RICHER INPUT.  5 channels per packet instead of 2 (std of the amplitude of each antenna, log level of each antenna,
     and the PHASE DIFFERENCE between the two antennas - a direction cue), each z-scored inside its own recording
     (no labels used, removes person/session offsets).  --channels picks a subset, e.g. --channels std for the old input.
  3. OPTIMISER.  AdamW, lr 3e-4 with cosine decay, gradient clipping 1.0 (the first run had one exploding epoch),
     more epochs, and parallel data loading.
It reads the labels that prepare_percom.py already made (nothing is relabelled) and the converted CSI (npz).
Output: the same files as train_percom.py (pred_person_<Person>.npz etc.) in --out, so pose_snn.py can use them:

    python train_percom_v2.py --split_test Vansh --epochs 3          # quick timing test, one fold
    python train_percom_v2.py                                         # all 4 folds (leave-one-person-out)
    python pose_snn.py --results ~/percom/siwis_results_v2
Run from the siwis repo folder (needs config.py, models/, losses/).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
import percom_common as pc
import train_percom as tp
from data.dataset import SiWiSDataset
from losses.losses import mask_loss, pose_loss
from models.signal_encoder import SignalEncoder
from models.siwis_net import SiWiSNet

CHANNELS = ["std_a", "std_b", "logamp_a", "logamp_b", "phase_diff"]
GROUPS = {"std": [0, 1], "amp": [2, 3], "phase": [4]}


# ----------------------------------------------------------------------------- features
def stream_features(npz_path, rate):
    """-> grid (absolute epoch s), (K, 5) float32, each column z-scored over the recording."""
    z = np.load(npz_path, allow_pickle=False)
    data, t = z["data"], z["t"].astype(np.float64)
    t0 = float(z["t0_epoch"])
    o = np.argsort(t)
    data, t = data[o], t[o]
    A, B = data[:, 0::2], data[:, 1::2]
    aA, aB = np.abs(A).astype(np.float32), np.abs(B).astype(np.float32)
    mA, mB = aA.mean(1), aB.mean(1)
    ok = (mA > 0) & (mB > 0)
    stdA = aA.std(1) / np.where(mA == 0, 1, mA)
    stdB = aB.std(1) / np.where(mB == 0, 1, mB)
    pd = np.unwrap(np.angle((A * np.conj(B)).sum(1))[ok])            # unwrap on the packet series, then resample
    tt = (t0 + t)[ok]
    grid = np.arange(tt[0], tt[-1], 1.0 / rate)
    cols = [stdA[ok], stdB[ok], np.log(mA[ok]), np.log(mB[ok]), pd]
    F = np.stack([np.interp(grid, tt, c) for c in cols], 1)
    n = max(int(2 * rate), 3)                                         # remove the slow phase drift (2 s moving average)
    pad = n // 2
    xp = np.pad(F[:, 4], (pad, n - 1 - pad), mode="edge")
    F[:, 4] = F[:, 4] - np.convolve(xp, np.ones(n) / n, mode="valid")
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    return grid, F.astype(np.float32)


def get_features(rec_dir, meta, npz_dir, rate):
    fg, ff = rec_dir / "features_v2_t.npy", rec_dir / "features_v2.npy"
    if fg.exists() and ff.exists():
        return np.load(fg), np.load(ff)
    grid, F = stream_features(Path(npz_dir) / meta["source_npz"], rate)
    np.save(fg, grid)
    np.save(ff, F)
    return grid, F


def load_recording_v2(rec_dir, meta, args, cols):
    grid, F = get_features(rec_dir, meta, args.npz_dir, args.rate)
    feats = np.column_stack([grid, F[:, cols].astype(np.float64)])
    ts = np.load(rec_dir / "frame_timestamps.npy")
    mask = np.load(rec_dir / "mask_labels.npy")
    kp_xy, kp_vis = np.load(rec_dir / "kp_xy.npy"), np.load(rec_dir / "kp_vis.npy")
    coords = np.zeros(kp_xy.shape[:3] + (2,), np.int64)
    coords[..., 0] = np.clip(kp_xy[..., 1].astype(np.int64), 0, config.HEATMAP_H - 1)
    coords[..., 1] = np.clip(kp_xy[..., 0].astype(np.int64), 0, config.HEATMAP_W - 1)
    return SiWiSDataset(feats, ts, mask, pc.LazyHeatmaps(kp_xy, kp_vis), coords, kp_vis)


def load_all_v2(args, cols):
    recs = []
    for m in sorted((Path(args.data).expanduser() / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in args.exclude}:
            continue
        print(f"  features {m.parent.name}", flush=True)
        meta["ds"] = load_recording_v2(m.parent, meta, args, cols)
        recs.append(meta)
    if not recs:
        raise SystemExit("no prepared recordings - run prepare_percom.py first")
    return recs


# ----------------------------------------------------------------------------- training
def run_fold(recs, args, test_person, dev, tag, n_ch):
    tr, te = tp.make_split(recs, "person", test_person)
    n_tr, n_te = sum(len(i) for _, _, i in tr), sum(len(i) for _, _, i in te)
    print(f"\n=== {tag}: test person {test_person} | train {n_tr} frames ({len(tr)} recordings) / test {n_te} frames ===")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SiWiSNet()
    model.signal_encoder = SignalEncoder(in_channels=n_ch)            # the repo default is 2 channels
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loader = DataLoader(ConcatDataset([Subset(d, idx) for _, d, idx in tr]), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    npix = config.HEATMAP_H * config.HEATMAP_W
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        run = dict(mask=0.0, pose=0.0, n=0)
        for b in loader:
            o = model(b["signal"].to(dev))
            B = o["mask"].shape[0]
            lm = mask_loss(torch.sigmoid(o["mask"]), b["mask_label"].to(dev)) / (B * npix)
            lp = pose_loss(o["pose_heatmaps"], b["keypoint_label"].to(dev), o["pose_tags"],
                           b["keypoint_coords"].to(dev), b["keypoint_visible"].to(dev))
            loss = lm + args.pose_weight * lp
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run["mask"] += float(lm.detach())
            run["pose"] += float(lp.detach())
            run["n"] += 1
        sched.step()
        if ep == 0 or (ep + 1) % args.print_every == 0 or ep == args.epochs - 1:
            nb = max(run["n"], 1)
            print(f"  epoch {ep + 1:3d}  mask_loss(mean) {run['mask'] / nb:.4f}  pose_loss {run['pose'] / nb:.5f}"
                  f"   ({time.time() - t0:.0f} s)", flush=True)
    pred = tp.predict(model, te, args.batch_size, dev)
    pts = pc.argmax_points(pred["pp"])
    res = dict(model=tp.score(pred["pm"], pred["gm"], pts, pred["gc"], pred["gv"]), baseline=tp.baseline(tr, pred),
               by_activity={}, test_person=test_person, n_train=n_tr, n_test=n_te)
    print("  MODEL     :", tp.fmt(res["model"]))
    print("  MEAN-POSE :", tp.fmt(res["baseline"]), " <- predict the average mask/skeleton, no CSI used")
    for a in sorted(set(pred["act"])):
        s = pred["act"] == a
        res["by_activity"][a] = tp.score(pred["pm"][s], pred["gm"][s], pts[s], pred["gc"][s], pred["gv"][s])
        print(f"    {a:4s}    :", tp.fmt(res["by_activity"][a]))
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"siwisv2_person_{test_person}.pt")
    np.savez_compressed(out / f"pred_person_{test_person}.npz", masks=pred["pm"].astype(np.float16),
                        heatmaps=pred["pp"].astype(np.float16), gt_masks=pred["gm"], activity=pred["act"],
                        person=pred["person"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--npz_dir", default="~/percom/work/npz")
    ap.add_argument("--out", default="~/percom/siwis_results_v2")
    ap.add_argument("--channels", nargs="+", default=["std", "amp", "phase"], choices=list(GROUPS),
                    help="input groups: std amp phase (default all; 'std' alone = the first run's input)")
    ap.add_argument("--split_test", default="", help="run only this held-out person (default: all, leave-one-person-out)")
    ap.add_argument("--exclude", nargs="*", default=["Divij"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pose_weight", type=float, default=500.0)
    ap.add_argument("--rate", type=float, default=150.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--print_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    pc.set_sampling_rate(args.rate)
    dev = torch.device(args.device)
    cols = sorted({c for g in args.channels for c in GROUPS[g]})
    print(f"device {dev} | input channels {[CHANNELS[c] for c in cols]} | pose_weight {args.pose_weight} | lr {args.lr}")
    recs = load_all_v2(args, cols)
    persons = sorted({r["person"] for r in recs})
    print(f"{len(recs)} recordings | persons {persons}")
    todo = [args.split_test] if args.split_test else persons
    results = [run_fold(recs, args, p, dev, f"fold {i}/{len(todo)}", len(cols)) for i, p in enumerate(todo, 1)]
    if len(results) > 1:
        print("\n=== leave-one-person-out mean ===")
        keys = results[0]["model"].keys()
        for name in ("model", "baseline"):
            print(f"  {name:9s}:", tp.fmt({k: float(np.nanmean([r[name][k] for r in results])) for k in keys}))
    pc.save_json(Path(args.out).expanduser() / "results_v2.json", results)
    print(f"\nSaved in {Path(args.out).expanduser()}")


if __name__ == "__main__":
    main()
