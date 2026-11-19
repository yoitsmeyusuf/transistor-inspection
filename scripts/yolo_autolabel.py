"""MVTec AD transistor/train/good görüntülerinden YOLO etiketlerini otomatik üretir.

Sınıflar: 0 = body (siyah plastik gövde), 1 = leads (gümüş bacaklar)
  - Gövde: koyu pikseller (V düşük) → morfolojik açma → en büyük bileşen
  - Bacaklar: renkle güvenilir ayrılamıyor (bakırı yansıtıyorlar). Sağlam parçada bacaklar hep gövdenin
    hemen altında, gövde yüksekliğinin ~0.8'i kadar uzanır; kutu gövdeden geometrik olarak türetilir.

Sadece train/good kullanılır; test kümesi değerlendirme için ayrı kalır.
Çıktı: datasets/yolo_transistor/{images,labels}/{train,val} + data.yaml + preview/
"""
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path("datasets/MVTecAD/transistor/train/good")
DST = Path("datasets/yolo_transistor")
VAL_RATIO = 0.15
random.seed(0)


def body_box(hsv):
    dark = (hsv[..., 2] < 90).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (111, 111)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[i]
    return x, y, x + w, y + h


def leads_box(body, w, h):
    bx1, by1, bx2, by2 = body
    bw, bh = bx2 - bx1, by2 - by1
    return max(0, int(bx1 - 0.06 * bw)), by2, min(w, int(bx2 + 0.06 * bw)), min(h, int(by2 + 0.8 * bh))


def to_yolo(cls, box, w, h):
    x1, y1, x2, y2 = box
    return f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    if DST.exists():
        shutil.rmtree(DST)
    files = sorted(SRC.glob("*.png"))
    random.shuffle(files)
    n_val = int(len(files) * VAL_RATIO)
    skipped = []
    for k, f in enumerate(files):
        split = "val" if k < n_val else "train"
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        body = body_box(hsv)
        leads = leads_box(body, w, h) if body else None
        if body is None or leads is None:
            skipped.append(f.name)
            continue
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(f, DST / "images" / split / f.name)
        (DST / "labels" / split / f"{f.stem}.txt").write_text(
            to_yolo(0, body, w, h) + "\n" + to_yolo(1, leads, w, h) + "\n")
        if k < 12:  # kontrol için önizleme
            cv2.rectangle(img, body[:2], body[2:], (0, 255, 0), 6)
            cv2.rectangle(img, leads[:2], leads[2:], (255, 0, 255), 6)
            (DST / "preview").mkdir(exist_ok=True)
            cv2.imwrite(str(DST / "preview" / f"{f.stem}.jpg"), cv2.resize(img, (320, 320)))

    (DST / "data.yaml").write_text(
        f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: body\n  1: leads\n")
    n_tr = len(list((DST / "labels/train").glob("*.txt")))
    n_va = len(list((DST / "labels/val").glob("*.txt")))
    print(f"train={n_tr} val={n_va} atlanan={len(skipped)} {skipped}")


if __name__ == "__main__":
    main()
