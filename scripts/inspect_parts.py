"""Eğitilmiş PatchCore modeliyle parça denetimi: OK / NG kararı + kusur konumu.

Örnekler:
  # Tek görüntü
  python scripts/inspect_parts.py --input datasets/MVTecAD/transistor/test/bent_lead/000.png
  # Klasör (alt klasörler dahil). Klasör adı 'good' ise gerçek etiket OK sayılır, rapor üretilir
  python scripts/inspect_parts.py --input datasets/MVTecAD/transistor/test
"""
import argparse
import csv
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from anomalib.data import PredictDataset  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Patchcore  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--category", default="transistor")
    p.add_argument("--input", required=True, help="Görüntü dosyası ya da klasör")
    p.add_argument("--ckpt", default=None, help="Varsayılan: results/Patchcore/MVTecAD/<kategori>/latest")
    p.add_argument("--out", default=None, help="Varsayılan: results/inspection/<kategori>")
    p.add_argument("--threshold", type=float, default=0.5, help="Normalize skor eşiği (0-1)")
    p.add_argument("--min-area", type=int, default=200, help="Bu pikselden küçük kusur bölgelerini yok say")
    return p.parse_args()


def find_ckpt(category: str) -> Path:
    base = Path("results/Patchcore/MVTecAD") / category
    ckpts = sorted(base.glob("*/weights/lightning/model.ckpt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise SystemExit(f"Checkpoint yok. Önce: python scripts/train_patchcore.py --category {category}")
    return ckpts[-1]


def image_dirs(inp: Path) -> list[Path]:
    """Görüntü içeren tüm klasörler (PredictDataset alt klasörlere inmiyor)."""
    if inp.is_file():
        return [inp]
    return sorted({p.parent for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT})


def defect_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int, int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h, area))
    return sorted(boxes, key=lambda b: -b[4])


def render(img_path: Path, amap: np.ndarray, mask: np.ndarray, score: float, is_ng: bool, boxes, out_path: Path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    amap = cv2.resize(amap, (w, h))
    amap = np.clip(amap, 0, 1)
    heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.55, heat, 0.45, 0)

    boxed = img.copy()
    color = (0, 0, 255) if is_ng else (0, 200, 0)
    for x, y, bw, bh, _ in boxes:
        cv2.rectangle(boxed, (x, y), (x + bw, y + bh), (0, 0, 255), max(2, w // 256))
    label = f"{'NG' if is_ng else 'OK'}  skor={score:.2f}  bolge={len(boxes)}"
    cv2.rectangle(boxed, (0, 0), (w, h // 12), color, -1)
    cv2.putText(boxed, label, (w // 60, h // 16), cv2.FONT_HERSHEY_SIMPLEX, w / 900, (255, 255, 255), max(2, w // 400))

    panel = np.hstack([img, overlay, boxed])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    inp = Path(args.input).resolve()
    ckpt = Path(args.ckpt) if args.ckpt else find_ckpt(args.category)
    out_dir = Path(args.out or f"results/inspection/{args.category}")
    print(f"Model: {ckpt}\nÇıktı: {out_dir}\n")

    engine = Engine(accelerator="gpu", devices=1, logger=False, enable_progress_bar=False)
    model = Patchcore()
    rows = []
    loaded = False

    for d in image_dirs(inp):
        preds = engine.predict(model=model, dataset=PredictDataset(path=d), ckpt_path=None if loaded else ckpt)
        loaded = True  # ağırlıklar ilk çağrıda yüklendi
        for batch in preds:
            for i in range(len(batch.image_path)):
                path = Path(batch.image_path[i])
                score = float(batch.pred_score[i])
                is_ng = score >= args.threshold
                amap = batch.anomaly_map[i].squeeze().cpu().numpy()
                mask = batch.pred_mask[i].squeeze().cpu().numpy()
                mask = cv2.resize(mask.astype(np.uint8), cv2.imread(str(path)).shape[1::-1],
                                  interpolation=cv2.INTER_NEAREST)
                boxes = defect_boxes(mask, args.min_area) if is_ng else []
                truth = path.parent.name
                rel = path.relative_to(inp) if inp.is_dir() else Path(path.name)
                render(path, amap, mask, score, is_ng, boxes, out_dir / rel)
                rows.append({"image": str(path), "truth": truth, "score": round(score, 4),
                             "verdict": "NG" if is_ng else "OK", "regions": len(boxes),
                             "largest_region_px": boxes[0][4] if boxes else 0})
                print(f"{'NG' if is_ng else 'OK'}  {score:.3f}  {truth}/{path.name}  bölge={len(boxes)}")

    if not rows:  # mevcut raporun üzerine boş dosya yazma
        raise SystemExit(f"Görüntü bulunamadı: {inp} (desteklenen: {', '.join(sorted(IMG_EXT))})")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Klasör adları gerçek etiket gibi kullanılabiliyorsa özet çıkar
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        by_type[r["truth"]][0] += 1
        by_type[r["truth"]][1] += r["verdict"] == "NG"
    if "good" in by_type and len(by_type) > 1:
        print("\nKusur türü        adet   NG denilen   oran")
        for t, (n, ng) in sorted(by_type.items()):
            print(f"{t:<17} {n:>4}   {ng:>10}   {ng / n:5.0%}")
        good_n, good_ng = by_type["good"]
        bad = [(n, ng) for t, (n, ng) in by_type.items() if t != "good"]
        bad_n, bad_ng = sum(n for n, _ in bad), sum(ng for _, ng in bad)
        print(f"\nKaçırılan kusurlu : {bad_n - bad_ng}/{bad_n}")
        print(f"Yanlış alarm      : {good_ng}/{good_n} sağlam parça NG dendi")
        print(f"Doğruluk          : {(bad_ng + good_n - good_ng) / len(rows):.1%}")
    print(f"\nRapor: {out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()
