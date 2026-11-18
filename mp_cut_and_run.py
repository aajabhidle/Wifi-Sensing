"""
mp_cut_and_run.py - cut the recordings with MediaPipe hand tracking, then re-run the whole
within-person experiment (pose model + SNN with velocity) on those cuts.

Put this file in ~/percom and run it from there. It follows the supervisor's instructions:
  1. MediaPipe Hands finds the hand landmarks in every video frame.
  2. The average height (Y) of the 21 landmarks gives the height of the hand in that frame.
     From this curve we find the highest points (hand up) and the lowest points (hand down), each with its time stamp.
  3. A cut goes from one lowest point to the next lowest point, so every segment contains exactly one raise of the hand.
  4. The video times of the cuts are converted to the CSI time axis with timing.txt (the same conversion
     video_segments.py always used), and the existing segment_pipeline.py is started on these cuts:
     SiWiS pose model (out-of-fold for the training part) -> SNN with velocity, 3 seeds,
     first ~80 % of the segments of every recording train, one segment is dropped, the last ~20 % test.

Modes (--mode)
  check   only run MediaPipe, write the detection rates, the curves and a few pictures with the landmarks.
          Run this FIRST and look at the pictures: do the dots sit on the hands?
  cut     check + write the cuts (segments_mp.json, plots, a csv with highest / lowest points and times).
  all     cut + start the training and testing (default; it asks before the long part unless you pass --yes).

Typical use
    # one time: pip install mediapipe opencv-python
    python mp_cut_and_run.py --mode check
    python mp_cut_and_run.py --mode cut --expect 48
    tmux new -s mp
    source ~/siwis_env/bin/activate; export PATH=$HOME/bin:$PATH
    python mp_cut_and_run.py --mode all --launcher gpurun --yes 2>&1 | tee ~/percom/run_mp.log

Where things are (all can be changed with options; the defaults are the ones of our setup)
    videos          ~/percom/video_small/<BHU|LHU|RHU>/<ACT>_S<n>_<Person>/video.mp4 + timing.txt   (other folders are ignored)
    pose data       ~/percom/siwis_data          (made by prepare_percom.py; needed: rec/<id>/meta.json, frame_timestamps.npy)
    converted CSI   ~/percom/work/npz
    SiWiS repo      ~/percom/siwis               (with our scripts in it: segment_pipeline.py, train_percom_v2.py, ...)
    results         ~/percom/siwis_results_seg_mp   (the old results in siwis_results_seg are not touched)

Nothing existing is overwritten: the old segments.json stays as it is, the new cuts go to segments_mp.json.

MediaPipe versions
  * old API (mp.solutions.hands, e.g. pip install mediapipe==0.10.14): the model is inside the pip package.
  * new API (mediapipe >= 0.10.30 removed mp.solutions): needs the file hand_landmarker.task. The script looks for it
    in ~/percom (or --model) and tries to download it once. If the server cannot download, download it on the laptop
    from  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    and copy it to ~/percom/hand_landmarker.task.
  --backend fake is a self-test only (it follows a bright blob instead of a hand). It never gives real results.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/"
             "hand_landmarker.task")
NEEDED = ["segment_pipeline.py", "train_percom_v2.py", "train_percom.py", "percom_common.py", "config.py"]


# ----------------------------------------------------------------------------- finding our own code
def setup_paths(args):
    """Make config.py, percom_common.py, video_segments.py ... importable. Returns the folder of the SiWiS repo."""
    percom = Path(args.percom).expanduser()
    siwis = Path(args.siwis).expanduser()
    extra = [percom / "segment_kit", percom / "siwis_percom"]       # only used when a file is not in the repo folder
    for d in [percom] + extra + [siwis]:                            # the last one inserted ends up first on the path
        if d.is_dir():
            sys.path.insert(0, str(d))
    missing = [n for n in NEEDED if not any((d / n).exists() for d in [siwis, percom] + extra)]
    if not (siwis / "config.py").exists():
        sys.exit(f"Error: {siwis}/config.py not found. This script needs the SiWiS repository folder (--siwis).")
    if missing:
        sys.exit("Error: these files were not found in the SiWiS folder or in ~/percom: " + ", ".join(missing))
    if not any((d / "video_segments.py").exists() for d in [percom, siwis] + extra):
        sys.exit("Error: video_segments.py not found (it holds the curve -> cuts code that is reused here).")
    return siwis


def find_script(name, siwis, percom):
    for d in [siwis, percom, percom / "segment_kit", percom / "siwis_percom"]:
        if (d / name).exists():
            return d / name
    sys.exit(f"Error: {name} not found")


# ----------------------------------------------------------------------------- the hand detector
class Hands:
    """detect(rgb, t_ms) -> list of (21, 2) arrays with x, y in 0..1 (y = 0 is the top of the picture)."""

    def __init__(self, backend, model, det_conf, trk_conf, complexity, percom):
        self.backend = backend
        if backend == "fake":
            return
        import mediapipe as mp
        self.mp = mp
        if backend == "auto":
            self.backend = backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        if backend == "solutions":
            self.det = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=complexity,
                                                min_detection_confidence=det_conf, min_tracking_confidence=trk_conf)
        else:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            path = self._find_model(model, percom)
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(path)), running_mode=vision.RunningMode.VIDEO,
                num_hands=2, min_hand_detection_confidence=det_conf, min_hand_presence_confidence=det_conf,
                min_tracking_confidence=trk_conf)
            self.det = vision.HandLandmarker.create_from_options(opts)
        self.last_ms = -1

    @staticmethod
    def _find_model(model, percom):
        for p in [Path(model).expanduser() if model else None, percom / "hand_landmarker.task",
                  Path("hand_landmarker.task")]:
            if p is not None and p.exists():
                return p
        target = percom / "hand_landmarker.task"
        try:
            import urllib.request
            print(f"downloading the MediaPipe hand model to {target} ...", flush=True)
            urllib.request.urlretrieve(MODEL_URL, target)
            return target
        except Exception as e:
            sys.exit("Error: the new MediaPipe API needs the file hand_landmarker.task and it could not be downloaded "
                     f"({e}).\nDownload it on a machine with internet from\n  {MODEL_URL}\nand copy it to {target} "
                     "(or pass --model /path/to/hand_landmarker.task).\nAlternative: pip install mediapipe==0.10.14, "
                     "which still has the model inside the package.")

    def detect(self, rgb, t_ms):
        if self.backend == "fake":
            g = rgb.mean(2)
            ys, xs = np.where(g > 200)
            if len(ys) < 20:
                return []
            h, w = g.shape
            cx, cy = xs.mean() / w, ys.mean() / h
            pts = np.stack([np.full(21, cx), cy + np.linspace(-0.02, 0.02, 21)], 1)
            return [pts]
        if self.backend == "solutions":
            res = self.det.process(rgb)
            lms = res.multi_hand_landmarks or []
            return [np.array([[p.x, p.y] for p in h.landmark], np.float32) for h in lms]
        t_ms = max(int(t_ms), self.last_ms + 1)                       # the tasks API wants strictly rising time stamps
        self.last_ms = t_ms
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.det.detect_for_video(img, t_ms)
        return [np.array([[p.x, p.y] for p in h], np.float32) for h in res.hand_landmarks]


def run_video(video, det, target_fps):
    """Read the video, find the hands in every (sampled) frame.
    Returns sig (n, 2) hand height [left-most hand, right-most hand] in 0..1 (up = large, nan = no hand),
    lm (n, 2, 21, 2) the landmarks (nan = none), fps of the sampled frames."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    step = max(1, int(round(src_fps / target_fps)))
    fps = src_fps / step
    sig, lms, k = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if k % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands = det.detect(rgb, 1000.0 * k / src_fps)
            row, lm = [np.nan, np.nan], np.full((2, 21, 2), np.nan, np.float32)
            hands = sorted(hands, key=lambda p: float(p[:, 0].mean()))[:2]   # left-most first
            for j, p in enumerate(hands):
                slot = j if len(hands) == 2 else (0 if p[:, 0].mean() < 0.5 else 1)
                row[slot] = 1.0 - float(p[:, 1].mean())              # the average Y of all 21 landmarks -> hand height
                lm[slot] = p
            sig.append(row)
            lms.append(lm)
        k += 1
    cap.release()
    return np.array(sig, np.float32), np.array(lms, np.float16), fps


def fill_curve(sig, fps, vs, gap_s=0.4, drop=0.25):
    """Short gaps (<= gap_s) are interpolated (the tracker flickers for a frame or two).
    A longer time without any hand means the hand is down or out of view: MediaPipe usually loses a hanging hand.
    Those frames get a "hand down" level that lies clearly below the heights seen while the hand was up
    (5th percentile of the detected heights minus `drop`, in picture heights). It must not be taken from the
    detected heights alone: if the hand is only detected while it is up, their lowest value is still a raised hand.
    Returns the curve columns without nan."""
    gap = max(int(gap_s * fps), 1)
    cols = [vs.fill_nan(sig[:, c], gap) for c in range(2)]
    seen = sig[~np.isnan(sig)]
    if seen.size < 5 * fps:
        return None
    low = max(0.0, float(np.percentile(seen, 5)) - drop)
    return np.stack([np.where(np.isnan(c), low, c) for c in cols], 1).astype(np.float32)


# ----------------------------------------------------------------------------- pictures for the check
def contact_sheet(path, video, lm, fps, idx, step_src):
    import cv2
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step_src))
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        for hand in lm[i].astype(np.float32):
            if np.isnan(hand).all():
                continue
            for x, y in hand:
                cv2.circle(fr, (int(x * w), int(y * h)), max(2, h // 120), (0, 255, 0), -1)
        cv2.putText(fr, f"t = {i / fps:.1f} s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(cv2.resize(fr, (int(w * 320 / h), 320)))
    cap.release()
    if tiles:
        cv2.imwrite(str(path), np.concatenate(tiles, 1))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="MediaPipe cuts + the within-person experiment on them",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--mode", choices=["check", "cut", "all"], default="all")
    ap.add_argument("--percom", default="~/percom")
    ap.add_argument("--video_dir", default="", help="default <percom>/video_small (BHU/LHU/RHU folders)")
    ap.add_argument("--data", default="", help="default <percom>/siwis_data")
    ap.add_argument("--npz_dir", default="", help="default <percom>/work/npz")
    ap.add_argument("--siwis", default="", help="default <percom>/siwis")
    ap.add_argument("--out", default="", help="default <percom>/siwis_results_seg_mp")
    ap.add_argument("--persons", nargs="*", default=[])
    ap.add_argument("--exclude", nargs="*", default=["Divij", "Nikita"])
    ap.add_argument("--only", nargs="*", default=[], help="only these recording ids, e.g. rhu_Vansh_S3")
    ap.add_argument("--backend", choices=["auto", "solutions", "tasks", "fake"], default="auto")
    ap.add_argument("--model", default="", help="hand_landmarker.task (new MediaPipe API only)")
    ap.add_argument("--det_conf", type=float, default=0.5)
    ap.add_argument("--trk_conf", type=float, default=0.5)
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1], help="old API only: 1 = more accurate")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second that MediaPipe looks at")
    ap.add_argument("--recompute", action="store_true", help="run MediaPipe again even if a saved curve exists")
    # cutting (the same meaning and defaults as in video_segments.py)
    ap.add_argument("--smooth", type=float, default=0.2, help="smoothing of the curve in seconds")
    ap.add_argument("--min_period", type=float, default=1.5, help="shortest time between two lowest points (s)")
    ap.add_argument("--prom", type=float, default=0.35, help="a raise must be this fraction of the hand range")
    ap.add_argument("--rest_frac", type=float, default=0.30)
    ap.add_argument("--expect", type=int, default=0, help="about how many repetitions per recording (flags outliers)")
    ap.add_argument("--min_segments", type=int, default=6, help="recordings with fewer cuts are left out of the experiment")
    # training
    ap.add_argument("--launcher", default="", help='prefix for the long job, e.g. "gpurun"')
    ap.add_argument("--yes", action="store_true", help="do not ask before the long training part")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--extra", default="", help='extra options for segment_pipeline.py, e.g. "--epochs 3 --device cpu"')
    args = ap.parse_args()

    percom = Path(args.percom).expanduser()
    video_dir = Path(args.video_dir or percom / "video_small").expanduser()
    data = Path(args.data or percom / "siwis_data").expanduser()
    npz_dir = Path(args.npz_dir or percom / "work" / "npz").expanduser()
    out_dir = Path(args.out or percom / "siwis_results_seg_mp").expanduser()
    args.siwis = args.siwis or str(percom / "siwis")
    siwis = setup_paths(args)
    import percom_common as pc
    import video_segments as vs

    recs = pc.find_recordings(video_dir, acts=("BHU", "LHU", "RHU"), persons=args.persons,
                              exclude_persons=args.exclude)
    if args.only:
        recs = [r for r in recs if r["sid"] in args.only]
    if not recs:
        sys.exit(f"Error: no recordings found under {video_dir} (expected BHU/ LHU/ RHU/ <ACT>_S<n>_<Person>/video.mp4)")
    print(f"{len(recs)} recordings under {video_dir} | persons {sorted({r['person'] for r in recs})}")

    sig_dir = data / "mp_signal"
    plot_dir = data / "segments_mp_plots"
    check_dir = data / "mp_check"
    for d in (sig_dir, plot_dir, check_dir):
        d.mkdir(parents=True, exist_ok=True)

    det = None
    out, rows, lines, det_stats, peak_rows = {}, [], [], [], []
    lines.append(f"{'recording':22s} {'hand found':>10s} {'2 hands':>8s} {'segs':>5s} {'len s (min/med/max)':>22s}  note")
    for rec in recs:
        sid, folder = rec["sid"], rec["folder"]
        video = folder / "video.mp4"
        meta_p = data / "rec" / sid / "meta.json"
        if not video.exists():
            print(f"{sid}: no video.mp4 - skipped")
            continue
        if meta_p.exists():
            meta = json.loads(meta_p.read_text())
        elif args.mode == "check":
            meta = None                                            # fine for --mode check: no mapping to CSI is needed
        else:
            print(f"{sid}: no {meta_p} (prepare_percom.py has not processed this recording) - skipped")
            continue
        cache = sig_dir / f"{sid}.npz"
        if cache.exists() and not args.recompute:
            z = np.load(cache)
            sig, lm, fps = z["sig"], z["lm"], float(z["fps"])
        else:
            if det is None:
                det = Hands(args.backend, args.model, args.det_conf, args.trk_conf, args.complexity, percom)
                print(f"hand detector: {det.backend}", flush=True)
            t0 = time.time()
            sig, lm, fps = run_video(video, det, args.fps)
            np.savez_compressed(cache, sig=sig, lm=lm, fps=fps)
            print(f"{sid}: MediaPipe on {len(sig)} frames in {time.time() - t0:.0f} s", flush=True)
            if det.backend != "solutions":
                det = None                                           # a fresh tracker for the next video
        found = float(np.mean(~np.isnan(sig).all(1)))
        two = float(np.mean((~np.isnan(sig)).all(1)))
        det_stats.append((rec["act"], found, two))

        curve = fill_curve(sig, fps, vs)
        if curve is None:
            note, res, frames = "failed (hardly any hand found)", None, []
            lines.append(f"{sid:22s} {found * 100:9.0f}% {two * 100:7.0f}% {0:5d} {'':>22s}  {note}")
            print(lines[-1], flush=True)
            continue
        res = vs.segment_signal(curve, fps, args.smooth, args.min_period, args.prom, args.rest_frac)
        timing = pc.read_timing(folder)
        vstart = timing.get("video_start_epoch")
        if meta is None:                                            # check mode without prepared data: count the cuts in video time
            ts, t_off, label_fps = None, None, 2.0
            frames = [[int(a * label_fps), int(b * label_fps)] for a, b in res["segments"]]
            note = "mediapipe (check only, no meta.json)"
        else:
            ts = np.load(data / "rec" / sid / "frame_timestamps.npy")
            label_fps = float(meta.get("label_fps", 2.0))
            t_off = float(vstart) + float(meta.get("sync_shift_s", 0.0)) if vstart is not None else None
            frames = vs.to_frames(res["segments"], ts - t_off) if t_off is not None else []
            note = "mediapipe"
            if vstart is None:
                note = "no video_start_epoch in timing.txt"
        if res["reason"]:
            note = f"failed ({res['reason']})"
        if args.expect and not (0.6 * args.expect <= len(frames) <= 1.5 * args.expect):
            note += f"  (expected about {args.expect}, please check the plot)"
        if len(frames) < args.min_segments:
            note += f"  -> fewer than {args.min_segments} cuts, left out"
        elif meta is not None:
            out[sid] = dict(person=meta["person"], act=meta["activity"], n_frames=int(len(ts)),
                            fps=label_fps, method="mediapipe", segments=frames,
                            up_runs=[[a, b] for a, b in frames])
        ln = np.array([(b - a) / label_fps for a, b in frames]) if frames else np.zeros(1)
        lines.append(f"{sid:22s} {found * 100:9.0f}% {two * 100:7.0f}% {len(frames):5d} "
                     f"{ln.min():7.1f}/{np.median(ln):5.1f}/{ln.max():5.1f}  {note}")
        print(lines[-1], flush=True)
        # highest and lowest points with their time stamps (the table the supervisor asked for)
        for i, ((a, b), pk) in enumerate(zip(res["segments"], res["peaks"])):
            peak_rows.append([sid, i + 1, f"{a:.2f}", f"{pk / fps:.2f}", f"{b:.2f}",
                              f"{res['curve'][int(round(a * fps))]:.3f}", f"{res['curve'][pk]:.3f}",
                              f"{res['curve'][min(int(round(b * fps)), len(res['curve']) - 1)]:.3f}"])
        vs.plot(plot_dir / f"{sid}.png", sid, curve, fps, res, f"[{note}]")
        pk = res["peaks"][:4] or list(np.linspace(0, len(lm) - 1, 4).astype(int))
        step_src = 1
        try:
            import cv2
            cap = cv2.VideoCapture(str(video))
            step_src = max(1, int(round((cap.get(cv2.CAP_PROP_FPS) or fps) / fps)))
            cap.release()
        except Exception:
            pass
        contact_sheet(check_dir / f"{sid}.jpg", video, lm, fps, pk, step_src)

    # ---------------------------------------------------------------- summary
    lines.append("")
    lines.append("hand detection by gesture (share of video frames with at least one hand found):")
    for a in ("bhu", "lhu", "rhu"):
        v = [f for (act, f, _) in det_stats if act == a]
        if v:
            lines.append(f"  {a}: mean {100 * np.mean(v):.0f} %   worst recording {100 * np.min(v):.0f} %")
    overall = float(np.mean([f for _, f, _ in det_stats])) if det_stats else 0.0
    if overall < 0.7:
        lines.append("WARNING: MediaPipe finds a hand in less than 70 % of the frames. The 480p videos in video_small may be "
                     "too small. Try the original videos (--video_dir <SSD>/percom), --complexity 1 or --det_conf 0.3.")
    lines.append("")
    lines.append("per person: segments in total, and about how many go to test with the 80/20 split")
    for p in sorted({v["person"] for v in out.values()}):
        tot = [len(v["segments"]) for v in out.values() if v["person"] == p]
        te = [max(1, int(round(0.2 * k))) for k in tot]
        lines.append(f"  {p:10s} {sum(tot):4d} segments in {len(tot)} recordings -> about {sum(te)} test segments")
    text = "\n".join(lines)
    print("\n" + text)
    (data / "segments_mp_summary.txt").write_text(text + "\n")
    with open(data / "hand_extremes_mp.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "segment", "start_low_s", "highest_s", "end_low_s", "height_at_start",
                    "height_at_highest", "height_at_end"])
        w.writerows(peak_rows)
    print(f"\ndetection table:   {data / 'segments_mp_summary.txt'}")
    print(f"pictures to look at: {check_dir}/<recording>.jpg  (green dots = MediaPipe landmarks)")
    print(f"hand curves + cuts:  {plot_dir}/<recording>.png")
    print(f"highest / lowest points with times: {data / 'hand_extremes_mp.csv'}")
    if args.mode == "check":
        return
    seg_path = data / "segments_mp.json"
    seg_path.write_text(json.dumps(out, indent=1))
    print(f"cuts saved: {seg_path}  ({len(out)} recordings)")
    if args.mode == "cut":
        return

    # ---------------------------------------------------------------- the experiment
    if len(out) < 3:
        sys.exit("Too few recordings with cuts to run the experiment. Check the pictures and plots first.")
    if not args.yes and sys.stdin.isatty():
        if input("Start the pose model + SNN on these cuts now (about 1.5-2 h on the GPU)? [y/N] ").strip().lower() != "y":
            print("Stopped. Run with --mode all --yes later (saved curves are reused, MediaPipe is not run again).")
            return
    script = find_script("segment_pipeline.py", siwis, percom)
    cmd = shlex.split(args.launcher) + [sys.executable, str(script), "--data", str(data), "--npz_dir", str(npz_dir),
                                         "--out", str(out_dir), "--segments", str(seg_path), "--control",
                                         "--seeds", str(args.seeds), "--exclude"] + args.exclude
    if args.persons:
        cmd += ["--persons"] + args.persons
    cmd += shlex.split(args.extra)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(siwis), str(percom), env.get("PYTHONPATH", "")])
    print("\n$ " + " ".join(cmd) + "\n", flush=True)
    code = subprocess.call(cmd, cwd=str(siwis), env=env)
    if code != 0:
        sys.exit(f"segment_pipeline.py stopped with exit code {code}")

    # ---------------------------------------------------------------- final table
    res_p = out_dir / "seg_results.json"
    if not res_p.exists():
        return
    new = json.loads(res_p.read_text())
    old_p = percom / "siwis_results_seg" / "seg_results.json"
    old = json.loads(old_p.read_text()) if old_p.exists() else {}
    names = {"kp": "6 arm joints + velocity", "both": "mask + joints + velocity", "mask": "body mask + velocity"}
    rows = ["", "=" * 78, f"FINAL: within person, MediaPipe cuts, SNN with velocity, Wi-Fi only ({args.seeds} seeds)", "=" * 78,
            f"{'input':28s} {'accuracy':>16s} {'test segs':>10s} {'95 % range':>14s}   previous cuts (video AI)"]
    for k in ("kp", "both", "mask"):
        r = new.get(f"pred_{k}")
        if not r:
            continue
        o = old.get(f"pred_{k}")
        prev = f"{100 * o['accuracy']:.1f} % ({o['n_test']} segs)" if o else "-"
        rows.append(f"{names[k]:28s} {100 * r['accuracy']:8.1f} +- {100 * r['std']:3.1f} % {r['n_test']:10d} "
                    f"{100 * r['ci95'][0]:6.0f}-{100 * r['ci95'][1]:3.0f} %   {prev}")
    rows.append("")
    rows.append("control (poses from the video, not a result):")
    for k in ("kp", "both", "mask"):
        r = new.get(f"gt_{k}")
        if r:
            rows.append(f"  {names[k]:26s} {100 * r['accuracy']:8.1f} +- {100 * r['std']:3.1f} %")
    rows.append(f"\nper person (arm joints): " + ", ".join(
        f"{p} {100 * a:.0f} %" for p, a in new.get("pred_kp", {}).get("per_person", {}).items()))
    rows.append(f"files: {out_dir}/seg_results.json, seg_predictions.csv")
    print("\n".join(rows))
    (out_dir / "mp_run_summary.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()