"""İki aşamalı transistör denetimi.

  AŞAMA 1 — YOLO + geometri  : parça VAR mı, DOĞRU yerde ve doğru yönde mi?
  AŞAMA 2 — PatchCore        : yerindeki parçada yüzey/şekil kusuru var mı?

Bir parça herhangi bir aşamada kalırsa NG. Geometri eşikleri yalnızca train/good'dan kalibre edilir.

  python scripts/inspect_two_stage.py --calibrate                     # bir kez (YOLO eğitiminden sonra)
  python scripts/inspect_two_stage.py --input datasets/MVTecAD/transistor/test
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from inspect_parts import IMG_EXT, defect_boxes, find_ckpt, image_dirs  # aynı klasördeki betik
from ultralytics import YOLO

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

YOLO_W = Path("results/yolo/transistor/weights/best.pt")
GEOM = Path("results/yolo/transistor/geometry.json")
TRAIN_GOOD = Path("datasets/MVTecAD/transistor/train/good")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input")
    p.add_argument("--calibrate", action="store_true", help="Geometri referansını train/good'dan çıkar")
    p.add_argument("--out", default="results/inspection_2stage/transistor")
    p.add_argument("--threshold", type=float, default=0.5, help="PatchCore skor eşiği")
    p.add_argument("--yolo-conf", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=200)
    return p.parse_args()


# ---------------------------------------------------------------- AŞAMA 1
def yolo_geometry(det: YOLO, img_path: Path, conf: float) -> dict:
    """En güvenilir body/leads kutuları + normalize merkez ve gövde→bacak açısı."""
    r = det.predict(str(img_path), conf=conf, verbose=False, device=0)[0]
    W = r.orig_shape[1]
    best = {}
    for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        if c not in best or s > best[c][0]:
            best[int(c)] = (s, [int(v) for v in b])
    g = {"body": best.get(0), "leads": best.get(1), "center": None, "angle": None}
    if g["body"]:
        x1, y1, x2, y2 = g["body"][1]
        g["center"] = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / W)
        if g["leads"]:
            lx1, ly1, lx2, ly2 = g["leads"][1]
            g["angle"] = math.degrees(math.atan2((ly1 + ly2) / 2 / W - g["center"][1],
                                                 (lx1 + lx2) / 2 / W - g["center"][0]))
    return g


def calibrate(det: YOLO, conf: float):
    feats = [yolo_geometry(det, f, conf) for f in sorted(TRAIN_GOOD.glob("*.png"))]
    feats = [f for f in feats if f["center"] and f["angle"] is not None]
    c = np.array([f["center"] for f in feats])
    ang = np.array([f["angle"] for f in feats])
    ref = c.mean(0)
    dist = np.linalg.norm(c - ref, axis=1)
    geom = {
        "ref_center": ref.round(4).tolist(),
        "max_dist": round(float(dist.max() * 1.10), 4),      # en uç sağlam örnek + %10 pay
        "ref_angle": round(float(ang.mean()), 2),
        "max_angle_dev": 15.0,                               # train std≈2.4° → ~6σ
        "n_train": len(feats),
        "train_angle_range": [round(float(ang.min()), 1), round(float(ang.max()), 1)],
        "train_max_dist": round(float(dist.max()), 4),
    }
    GEOM.write_text(json.dumps(geom, indent=2))
    print(json.dumps(geom, indent=2))


def stage1(g: dict, geom: dict) -> tuple[bool, str]:
    if not g["body"]:
        return False, "parca_yok"
    if not g["leads"]:
        return False, "bacak_yok"
    d = math.dist(g["center"], geom["ref_center"])
    dev = abs((g["angle"] - geom["ref_angle"] + 180) % 360 - 180)
    if dev > geom["max_angle_dev"]:
        return False, f"yon_hatali({g['angle']:.0f}°)"
    if d > geom["max_dist"]:
        return False, f"kaymis({d:.3f})"
    return True, "ok"


# ---------------------------------------------------------------- görselleştirme
def render(img_path, g, s1_ok, s1_reason, amap, score, boxes, final_ng, reason, out_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    lw = max(2, w // 200)

    p1 = img.copy()
    for key, col in (("body", (255, 120, 0)), ("leads", (255, 255, 0))):
        if g[key]:
            x1, y1, x2, y2 = g[key][1]
            cv2.rectangle(p1, (x1, y1), (x2, y2), col, lw)
    _banner(p1, f"1) YOLO: {s1_reason}", (0, 160, 0) if s1_ok else (0, 0, 255))

    if amap is not None:
        heat = cv2.applyColorMap((np.clip(cv2.resize(amap, (w, h)), 0, 1) * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        p2 = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
        for x, y, bw, bh, _ in boxes:
            cv2.rectangle(p2, (x, y), (x + bw, y + bh), (0, 0, 255), lw)
        _banner(p2, f"2) PatchCore: {score:.2f}", (0, 0, 255) if score >= 0.5 else (0, 160, 0))
    else:
        p2 = (img * 0.3).astype(np.uint8)
        _banner(p2, "2) PatchCore: atlandi", (90, 90, 90))

    p3 = img.copy()
    _banner(p3, f"{'NG' if final_ng else 'OK'}  {reason}", (0, 0, 255) if final_ng else (0, 160, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))


def _banner(img, text, color):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, h // 12), color, -1)
    cv2.putText(img, text, (w // 60, h // 17), cv2.FONT_HERSHEY_SIMPLEX, w / 1100, (255, 255, 255),
                max(2, w // 450))


# ---------------------------------------------------------------- ana akış
def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    det = YOLO(str(YOLO_W))
    if args.calibrate:
        calibrate(det, args.yolo_conf)
        return
    if not args.input:
        raise SystemExit("--input gerekli (ya da --calibrate)")
    if not GEOM.exists():
        raise SystemExit("Önce: python scripts/inspect_two_stage.py --calibrate")
    geom = json.loads(GEOM.read_text())
    inp = Path(args.input).resolve()
    out_dir = Path(args.out)

    # Aşama 2'yi toplu çalıştır (PatchCore klasör bazında çok daha hızlı)
    engine = Engine(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
    model, ckpt = Patchcore(), find_ckpt("transistor")
    pc = {}
    for k, d in enumerate(image_dirs(inp)):
        for batch in engine.predict(model=model, dataset=PredictDataset(path=d),
                                    ckpt_path=ckpt if k == 0 else None):
            for i, p in enumerate(batch.image_path):
                pc[Path(p).resolve()] = (float(batch.pred_score[i]),
                                         batch.anomaly_map[i].squeeze().cpu().numpy(),
                                         batch.pred_mask[i].squeeze().cpu().numpy())

    rows = []
    for path in sorted(pc):
        g = yolo_geometry(det, path, args.yolo_conf)
        s1_ok, s1_reason = stage1(g, geom)
        score, amap, mask = pc[path]
        s2_ng = score >= args.threshold
        boxes = []
        if s1_ok:
            if s2_ng:
                h, w = cv2.imread(str(path)).shape[:2]
                boxes = defect_boxes(cv2.resize(mask.astype(np.uint8), (w, h),
                                                interpolation=cv2.INTER_NEAREST), args.min_area)
            final_ng, reason = s2_ng, ("yuzey_kusuru" if s2_ng else "ok")
            stage = 2 if s2_ng else 0
        else:  # parça yok / yanlış yerde: yüzey analizi anlamsız, kararı aşama 1 verir
            final_ng, reason, stage, amap = True, s1_reason, 1, None
        rel = path.relative_to(inp) if inp.is_dir() else Path(path.name)
        render(path, g, s1_ok, s1_reason, amap, score, boxes, final_ng, reason, out_dir / rel)
        rows.append({"image": str(path), "truth": path.parent.name, "verdict": "NG" if final_ng else "OK",
                     "caught_by_stage": stage, "reason": reason, "patchcore_score": round(score, 4),
                     "body_conf": round(g["body"][0], 3) if g["body"] else 0,
                     "angle": round(g["angle"], 1) if g["angle"] is not None else ""})
        print(f"{rows[-1]['verdict']}  {path.parent.name}/{path.name:<8} {reason}")

    if not rows:  # mevcut raporun üzerine boş dosya yazma
        raise SystemExit(f"Görüntü bulunamadı: {inp} (desteklenen: {', '.join(sorted(IMG_EXT))})")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)

    stats = defaultdict(lambda: {"n": 0, "ng": 0, "s1": 0, "s2": 0, "pc_only": 0})
    for r in rows:
        s = stats[r["truth"]]
        s["n"] += 1
        s["ng"] += r["verdict"] == "NG"
        s["s1"] += r["caught_by_stage"] == 1
        s["s2"] += r["caught_by_stage"] == 2
        s["pc_only"] += r["patchcore_score"] >= args.threshold
    if "good" in stats and len(stats) > 1:
        print(f"\n{'tür':<14}{'adet':>5}{'NG':>5}{'1.aşama':>9}{'2.aşama':>9}   | sadece PatchCore NG")
        for t, s in sorted(stats.items()):
            print(f"{t:<14}{s['n']:>5}{s['ng']:>5}{s['s1']:>9}{s['s2']:>9}   | {s['pc_only']:>3}")
        bad = [s for t, s in stats.items() if t != "good"]
        bn, bng = sum(s["n"] for s in bad), sum(s["ng"] for s in bad)
        g = stats["good"]
        pc_bng = sum(s["pc_only"] for s in bad)
        print(f"\n                    İKİ AŞAMA   SADECE PATCHCORE")
        print(f"Kaçırılan kusurlu : {bn - bng:>3}/{bn}      {bn - pc_bng:>3}/{bn}")
        print(f"Yanlış alarm      : {g['ng']:>3}/{g['n']}      {g['pc_only']:>3}/{g['n']}")
        acc2 = (bng + g["n"] - g["ng"]) / len(rows)
        acc1 = (pc_bng + g["n"] - g["pc_only"]) / len(rows)
        print(f"Doğruluk          : {acc2:>7.1%}      {acc1:>7.1%}")
    print(f"\nRapor: {out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
