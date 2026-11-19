"""Transistör gövde/bacak dedektörü (YOLO11n) eğitimi.

Augmentasyon bilinçli olarak geniş: eğitimde sadece düzgün takılmış parçalar var, ama dedektörün
kaymış/dönmüş/ters parçayı da BULABİLMESİ gerekiyor. "Doğru yerde mi?" kararını YOLO değil,
inspect_two_stage.py içindeki geometri kuralları veriyor.
"""
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="datasets/yolo_transistor/data.yaml",
    epochs=80, imgsz=640, batch=16, device=0, workers=4, seed=0,
    degrees=180, flipud=0.5, fliplr=0.5, translate=0.3, scale=0.3, mosaic=1.0,
    hsv_h=0.01, hsv_s=0.3, hsv_v=0.3,
    project=str(Path("results/yolo").resolve()), name="transistor", exist_ok=True,
    plots=True, verbose=False,
)
