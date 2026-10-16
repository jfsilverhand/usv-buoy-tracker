#!/usr/bin/env python3
"""
Offline replay evaluation of the USV buoy detector on recorded footage.

Runs the same YOLOv8 ONNX model used on the vehicle over a video file and reports:
  - detection rate (frames with >= 1 buoy above threshold)
  - per-class detection counts
  - target-lock continuity (lock segments, losses, re-acquisition time)
  - the normalized center error (ex, ey) the PID loop would receive
  - inference latency on THIS machine (label the device when you report it)

No ground-truth labels are used, so this does NOT give precision/recall.
Spot-check sampled frames (--samples) to estimate false positives by eye.

Usage:
  pip install onnxruntime opencv-python matplotlib   # or onnxruntime-gpu
  python3 tools/offline_eval.py --video videos/aura_2025_run.mp4 \
      --model models/best.onnx --out eval_out --demo
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ["kirmiziduba", "sariduba", "siyahduba", "turuncuduba", "yesilduba"]
COLORS = [(40, 40, 230), (40, 220, 230), (60, 60, 60), (30, 140, 255), (60, 200, 60)]


def letterbox(img, size=640):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def preprocess(frame, size=640):
    lb, r, px, py = letterbox(frame, size)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), r, px, py


def postprocess(out, r, px, py, w, h, conf_th, iou_th):
    p = out[0].T  # (8400, 4 + nc)
    boxes, scores = p[:, :4], p[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(scores)), cls]
    keep = conf >= conf_th
    boxes, conf, cls = boxes[keep], conf[keep], cls[keep]
    if len(boxes) == 0:
        return []
    xy = np.empty_like(boxes)
    xy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - px) / r
    xy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - py) / r
    xy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - px) / r
    xy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - py) / r
    xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w)
    xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h)
    rects = [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xy]
    idx = cv2.dnn.NMSBoxesBatched(rects, conf.tolist(), cls.tolist(), conf_th, iou_th)
    idx = np.array(idx).flatten() if len(idx) else []
    return [(xy[i].tolist(), float(conf[i]), int(cls[i])) for i in idx]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="models/best.onnx")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--conf", type=float, default=0.40)  # same as ds_main.py
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--demo", action="store_true", help="write annotated demo mp4")
    ap.add_argument("--samples", type=int, default=12, help="annotated frames to save for spot-check")
    ap.add_argument("--select", choices=["conf", "sticky"], default="conf",
                    help="conf = highest confidence (as in ds_main.py); sticky = keep current target if still visible")
    ap.add_argument("--lost-after", type=int, default=5, help="frames without target before lock counts as lost")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(args.model, providers=providers)
    inp = sess.get_inputs()[0].name
    used = sess.get_providers()[0]

    cap = cv2.VideoCapture(args.video)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sample_at = set(np.linspace(0, max(total - 1, 0), args.samples, dtype=int).tolist())

    writer = None
    if args.demo:
        writer = cv2.VideoWriter(os.path.join(args.out, "demo_annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps_src, (W, H))

    rows, infer_ms, e2e_ms = [], [], []
    class_counts = [0] * len(NAMES)
    prev_box, miss, switches = None, 0, 0
    i = 0
    # warm-up
    dummy = np.zeros((1, 3, 640, 640), np.float32)
    for _ in range(3):
        sess.run(None, {inp: dummy})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        x, r, px, py = preprocess(frame)
        t1 = time.perf_counter()
        out = sess.run(None, {inp: x})[0]
        t2 = time.perf_counter()
        dets = postprocess(out, r, px, py, W, H, args.conf, args.iou)
        t3 = time.perf_counter()
        infer_ms.append((t2 - t1) * 1000)
        e2e_ms.append((t3 - t0) * 1000)

        # target selection
        best = max(dets, key=lambda d: d[1]) if dets else None  # ds_main.py behaviour
        if args.select == "sticky" and dets and prev_box:
            cand = max(dets, key=lambda d: iou(d[0], prev_box))
            if iou(cand[0], prev_box) > 0.3:
                best = cand
        switched = bool(best and prev_box and iou(best[0], prev_box) <= 0.1)
        switches += int(switched)
        for d in dets:
            class_counts[d[2]] += 1
        same_target = bool(best and prev_box and iou(best[0], prev_box) > 0.1)
        if best:
            miss = 0
            prev_box = best[0]
            cx, cy = (best[0][0] + best[0][2]) / 2, (best[0][1] + best[0][3]) / 2
            ex, ey = (cx - W / 2) / (W / 2), (cy - H / 2) / (H / 2)
        else:
            miss += 1
            if miss >= args.lost_after:
                prev_box = None
            ex = ey = None
        rows.append(dict(frame=i, t=i / fps_src, n_det=len(dets),
                         cls=NAMES[best[2]] if best else "", conf=round(best[1], 3) if best else "",
                         ex=round(ex, 4) if ex is not None else "", ey=round(ey, 4) if ey is not None else "",
                         same_target=int(same_target), switched=int(switched), infer_ms=round(infer_ms[-1], 2)))

        if writer is not None or i in sample_at:
            vis = frame.copy()
            for (b, c, k) in dets:
                col = COLORS[k]
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
                cv2.putText(vis, f"{NAMES[k]} {c:.2f}", (int(b[0]), max(20, int(b[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            cv2.drawMarker(vis, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
            if best:
                cv2.line(vis, (W // 2, H // 2), (int(cx), int(cy)), (0, 255, 255), 2)
            status = f"TRACK {NAMES[best[2]]}  ex={ex:+.2f} ey={ey:+.2f}" if best else "SEARCH"
            cv2.putText(vis, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if writer is not None:
                writer.write(vis)
            if i in sample_at:
                cv2.imwrite(os.path.join(args.out, f"sample_{i:05d}.jpg"), vis)
        i += 1

    cap.release()
    if writer is not None:
        writer.release()

    with open(os.path.join(args.out, "per_frame.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # lock segments: consecutive frames with a target, gaps < lost_after tolerated
    has = np.array([r["n_det"] > 0 for r in rows])
    segs, start, gap = [], None, 0
    for k, v in enumerate(has):
        if v:
            if start is None:
                start = k
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= args.lost_after:
                segs.append((start, k - gap))
                start, gap = None, 0
    if start is not None:
        segs.append((start, len(has) - 1 - gap))
    gaps = [(segs[j + 1][0] - segs[j][1] - 1) / fps_src for j in range(len(segs) - 1)]

    ea = np.array([abs(r["ex"]) for r in rows if r["ex"] != ""])
    summary = dict(
        video=os.path.basename(args.video), frames=len(rows), duration_s=round(len(rows) / fps_src, 1),
        resolution=f"{W}x{H}", conf_threshold=args.conf, device=used, target_select=args.select,
        target_switches=switches,
        detection_rate=round(float(has.mean()), 3),
        detections_per_class={NAMES[k]: class_counts[k] for k in range(len(NAMES))},
        lock_segments=len(segs),
        longest_lock_s=round(max(((b - a + 1) / fps_src for a, b in segs), default=0), 1),
        losses=max(len(segs) - 1, 0),
        median_reacquire_s=round(float(np.median(gaps)), 2) if gaps else None,
        mean_abs_ex=round(float(ea.mean()), 3) if len(ea) else None,
        infer_ms_avg=round(float(np.mean(infer_ms)), 2), infer_ms_p95=round(float(np.percentile(infer_ms, 95)), 2),
        e2e_ms_avg=round(float(np.mean(e2e_ms)), 2), e2e_ms_p95=round(float(np.percentile(e2e_ms, 95)), 2),
        fps_equiv=round(1000 / float(np.mean(e2e_ms)), 1),
    )
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = np.array([r["t"] for r in rows])
        ex = np.array([r["ex"] if r["ex"] != "" else np.nan for r in rows], dtype=float)
        ey = np.array([r["ey"] if r["ey"] != "" else np.nan for r in rows], dtype=float)
        fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
        ax[0].plot(t, ex, lw=1, label="ex (yaw error)")
        ax[0].plot(t, ey, lw=1, label="ey (pitch error)")
        ax[0].axhline(0, color="gray", lw=0.5)
        ax[0].set_ylabel("normalized error")
        ax[0].legend(loc="upper right")
        ax[0].set_title("Target center error fed to the PID loop (offline replay)")
        ax[1].fill_between(t, 0, has.astype(int), step="mid", alpha=0.6)
        ax[1].set_yticks([0, 1], ["search", "track"])
        ax[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "tracking_error.png"), dpi=130)
    except ImportError:
        print("matplotlib not installed, skipping plot")


if __name__ == "__main__":
    main()
