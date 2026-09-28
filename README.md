# ExplainAD

Factory defect inspection from an ordinary camera image, built from three published models:

1. **SAM** (Kirillov et al., *Segment Anything*, ICCV 2023) separates the object from the background.
2. **SegAD** (Baitieva et al., *Supervised Anomaly Detection for Complex Industrial Images*, CVPR 2024),
   with **PatchCore** (Roth et al., CVPR 2022) anomaly maps, decides whether the part is defective and where.
3. **Qwen3-VL-2B-Instruct** describes the flagged defect in one sentence (demo only; not part of the score).

## Result (MVTec 3D-AD, RGB images, 10 classes × 10 seeds, image AUROC)

| Setup | 3D sensor needed | AUROC |
|---|---|---|
| SegAD, 1 segment (whole image) | no | 78.6 ± 1.2 |
| SegAD + SAM parts (colour-clustered) | no | 78.7 ± 1.1 |
| **SegAD + SAM object/background** | **no** | **82.4 ± 0.9** |
| SegAD + real 3D-sensor depth (same protocol, separate run) | yes | 90.4 ± 1.0 |

SAM object/background adds +3.8 AUROC and closes about a third of the gap to a 3D sensor.
Per-class numbers: `results/report.txt`; per seed: `results/per_seed.csv`.
Protocol: SegAD head trained on validation/good plus 10 defective test images per seed (official SegAD seeds),
evaluated on the remaining test images.

Known limitations: colour-clustered parts are not consistent across images (`results/parts_carrot_gland.png`);
tire (black on black) cannot be separated from the background; the VLM usually localises correctly but often
names the wrong defect type.

## Run

Data: MVTec 3D-AD, expected at `../GeoAD/data/mvtec3d/<class>/{train,validation,test}/...`.

```
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1   # optional laptop GPU guard
python pipeline.py fit         # SAM part maps + PatchCore per class (~5 min/class on an 8 GB laptop GPU)
python pipeline.py eval        # SegAD heads -> results/report.txt
python pipeline.py explain ../GeoAD/data/mvtec3d/dowel/test/bent/rgb/000.png   # -> results/explain_*.png
```

Models download from Hugging Face on first use (`facebook/sam-vit-base`, `Qwen/Qwen3-VL-2B-Instruct`).
