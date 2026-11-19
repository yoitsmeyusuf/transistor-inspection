"""MVTec AD üzerinde PatchCore eğitimi + testi (Anomalib 2.x).

Kullanım: python scripts/train_patchcore.py --category bottle
"""
import argparse

import torch

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

p = argparse.ArgumentParser()
p.add_argument("--category", default="bottle")
p.add_argument("--root", default="./datasets/MVTecAD")
args = p.parse_args()
torch.set_float32_matmul_precision("high")

datamodule = MVTecAD(root=args.root, category=args.category,
                     train_batch_size=16, eval_batch_size=16, num_workers=4)
model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                  coreset_sampling_ratio=0.1)
engine = Engine(default_root_dir="./results", accelerator="gpu", devices=1)

engine.fit(model=model, datamodule=datamodule)
engine.test(model=model, datamodule=datamodule)
