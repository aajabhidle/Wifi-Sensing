#!/usr/bin/env python3
"""
pose_snn.py - stage 2: the SiWiS output (a body-mask "image" per time step, predicted from CSI ONLY) -> SNN -> activity.

    CSI --(SiWiS, trained once with video labels)--> predicted body mask per frame --> SNN --> bhu / lhu / rhu

At this stage NO video is used. The input is the sequence of masks that SiWiS predicted from the radio signal.
The predictions come from the leave-one-person-out run you already did (train_percom.py --split person --loso):
every person's frames were predicted by a SiWiS model that never saw that person, so nothing leaks.

Each window of T consecutive frames (default 10 frames = 5 s at 2 fps) is a small "pose spectrogram":
time x (pooled mask pixels). --velocity adds the frame-to-frame change (how fast the pose moves).

    python pose_snn.py --results ~/percom/siwis_results --data ~/percom/siwis_data                 # CSI-only input
    python pose_snn.py ... --velocity                                                              # + velocity
    python pose_snn.py ... --source gt                                                             # CONTROL: masks from video

--source gt feeds the video-derived masks. It is NOT a real result (it uses the video); it is the upper bound that
tells you whether the problem is the SNN stage or the quality of the SiWiS masks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ACTS = ["bhu", "lhu", "rhu"]


# ----------------------------------------------------------------------------- data
def load_frames(results, data, source, persons_exclude):
    """-> list of recordings: dict(person, act, rec, X (n, 12, 16) pooled mask, in time order)."""
    results, data = Path(results).expanduser(), Path(data).expanduser()
    metas = []
    for m in sorted((data / "rec").glob("*/meta.json")):
        meta = json.loads(m.read_text())
        if meta["person"].lower() in {e.lower() for e in persons_exclude}:
            continue
        meta["dir"] = m.parent
        meta["n"] = len(np.load(m.parent / "frame_timestamps.npy"))
        metas.append(meta)
    recs = []
    for pf in sorted(results.glob("pred_person_*.npz")):
        z = np.load(pf, allow_pickle=False)
        person = pf.stem.replace("pred_person_", "")
        mine = [m for m in metas if m["person"] == person]
        arr = z["masks"] if source == "pred" else z["gt_masks"]
        arr = arr.astype(np.float32)
        if sum(m["n"] for m in mine) != len(arr):
            raise SystemExit(f"{pf.name}: {len(arr)} predicted frames but the recordings of {person} have "
                             f"{sum(m['n'] for m in mine)} - the order/count does not match, tell me before going on")
        i = 0
        for m in mine:
            a = arr[i:i + m["n"]]
            i += m["n"]
            n, h, w = a.shape
            pooled = a[:, : h // 4 * 4, : w // 4 * 4].reshape(n, h // 4, 4, w // 4, 4).mean((2, 4))
            recs.append(dict(person=person, act=m["activity"], rec=m["dir"].name, X=pooled))
    if not recs:
        raise SystemExit(f"no pred_person_*.npz in {results}")
    return recs


def make_windows(recs, T, hop, velocity, center):
    X, y, person, rec = [], [], [], []
    for r in recs:
        a = r["X"].reshape(len(r["X"]), -1)
        if center:
            a = a - a.mean(0, keepdims=True)        # per-recording centring, no labels used
        if velocity:
            v = np.diff(a, axis=0, prepend=a[:1])
            a = np.concatenate([a, v], axis=1)
        for s in range(0, len(a) - T + 1, hop):
            X.append(a[s:s + T])
            y.append(ACTS.index(r["act"]))
            person.append(r["person"])
            rec.append(r["rec"])
    return np.array(X, np.float32), np.array(y), np.array(person), np.array(rec)


# ----------------------------------------------------------------------------- spiking network
class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1 + 10 * x.abs()) ** 2          # fast-sigmoid surrogate gradient


class SNN(nn.Module):
    def __init__(self, n_in, h1=256, h2=128, n_out=3, beta=0.9, thr=1.0):
        super().__init__()
        self.f1, self.f2, self.f3 = nn.Linear(n_in, h1), nn.Linear(h1, h2), nn.Linear(h2, n_out)
        self.beta, self.thr = beta, thr
        self.rate2 = 0.0

    def forward(self, x):                           # x: (B, T, F), fed as a current at every time step
        B, T, _ = x.shape
        v1 = x.new_zeros(B, self.f1.out_features)
        v2 = x.new_zeros(B, self.f2.out_features)
        out = x.new_zeros(B, self.f3.out_features)
        s2_sum = 0.0
        for t in range(T):
            v1 = self.beta * v1 + self.f1(x[:, t])
            s1 = Spike.apply(v1 - self.thr)
            v1 = v1 - s1 * self.thr
            v2 = self.beta * v2 + self.f2(s1)
            s2 = Spike.apply(v2 - self.thr)
            v2 = v2 - s2 * self.thr
            out = self.beta * out + self.f3(s2)     # leaky readout
            s2_sum = s2_sum + s2.mean().item()
        self.rate2 = s2_sum / T
        return out


def train_eval(Xtr, ytr, Xte, args, dev):
    mu, sd = Xtr.mean((0, 1), keepdims=True), Xtr.std((0, 1), keepdims=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    torch.manual_seed(args.seed)
    net = SNN(Xtr.shape[2]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    w = torch.tensor(len(ytr) / (3 * np.bincount(ytr, minlength=3).clip(1)), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), args.batch):
            b = perm[i:i + args.batch]
            loss = lossf(net(Xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        tr_acc = np.mean(np.concatenate([net(Xt[i:i + 256]).argmax(1).cpu().numpy() for i in range(0, len(Xt), 256)]) == ytr)
        pred = np.concatenate([net(torch.tensor(Xte[i:i + 256], device=dev)).argmax(1).cpu().numpy()
                               for i in range(0, len(Xte), 256)])
    return pred, tr_acc, net.rate2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="~/percom/siwis_results")
    ap.add_argument("--data", default="~/percom/siwis_data")
    ap.add_argument("--source", choices=["pred", "gt"], default="pred")
    ap.add_argument("--velocity", action="store_true")
    ap.add_argument("--no_center", action="store_true")
    ap.add_argument("--window", type=int, default=10, help="frames per window (2 fps -> 10 = 5 s)")
    ap.add_argument("--hop", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", nargs="*", default=["Divij"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    recs = load_frames(args.results, args.data, args.source, args.exclude)
    X, y, person, rec = make_windows(recs, args.window, args.hop, args.velocity, not args.no_center)
    persons = sorted(set(person))
    tag = f"source={args.source}{' (VIDEO-DERIVED CONTROL, not a real result)' if args.source == 'gt' else ' (CSI only)'}" \
          f" | velocity={'on' if args.velocity else 'off'} | {len(X)} windows of {args.window} frames | persons {persons}"
    print(tag)
    P = np.zeros(len(y), int)
    for p in persons:
        te = person == p
        pred, tr_acc, rate = train_eval(X[~te], y[~te], X[te], args, dev)
        P[te] = pred
        print(f"  test {p:10s} window acc {np.mean(pred == y[te]) * 100:5.1f}%   (train acc {tr_acc * 100:5.1f}%, layer-2 spike rate {rate * 100:4.1f}%)",
              flush=True)
    ok = tot = 0
    for r in sorted(set(rec)):
        s = rec == r
        ok += np.bincount(P[s], minlength=3).argmax() == y[s][0]
        tot += 1
    print(f"\n=== leave-one-person-out, pooled ===")
    print(f"window accuracy     {np.mean(P == y) * 100:5.1f}%   chance 33.3%")
    print(f"recording accuracy  {ok}/{tot} = {ok / tot * 100:5.1f}%   chance 33.3%")
    print("\nconfusion matrix, row % (true -> predicted)")
    print("            " + " ".join(f"{a:>7s}" for a in ACTS))
    for i, a in enumerate(ACTS):
        s = y == i
        print(f"true {a:5s}  " + " ".join(f"{np.mean(P[s] == j) * 100:7.1f}" for j in range(3)))


if __name__ == "__main__":
    main()
