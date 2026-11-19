"""Ultralytics YOLO hızlı test: önceden eğitilmiş modelle örnek görüntüde tahmin."""
from pathlib import Path

from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.predict("https://ultralytics.com/images/bus.jpg", device=0, save=True,
                        project=str(Path("results/yolo").resolve()), name="smoke", exist_ok=True)
for r in results:
    print(r.verbose())
