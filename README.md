# ExplainAD

Factory defect inspection from an ordinary camera image, built from published models:

1. **SAM** (Kirillov et al., *Segment Anything*, ICCV 2023) separates the object from the background.
2. **SegAD** (Baitieva et al., *Supervised Anomaly Detection for Complex Industrial Images*, CVPR 2024)
   combines three **PatchCore** (Roth et al., CVPR 2022) anomaly maps, on WideResNet-50 features, on
   **DINO** ViT-B/8 features (Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*,
   ICCV 2021) and on **CLIP** ViT-B/16 features (Radford et al., *Learning Transferable Visual Models From
   Natural Language Supervision*, ICML 2021), and decides whether the part is defective and where.
3. **Qwen3-VL-2B-Instruct** describes the flagged defect in one sentence (demo only; not part of the score).

## Result (MVTec 3D-AD, RGB images, 10 classes × 10 seeds, image AUROC)

| Setup | 3D sensor needed | AUROC |
|---|---|---|
| SegAD, 1 segment (whole image) | no | 78.6 ± 1.2 |
| SegAD + SAM parts (colour-clustered) | no | 78.7 ± 1.1 |
| SegAD + SAM object/background (v1) | no | 82.4 ± 0.9 |
| SegAD + SAM object/background + DINO (v2) | no | 85.2 ± 1.1 |
| **SegAD + SAM object/background + DINO + CLIP (v3)** | **no** | **87.5 ± 0.9** |
| SegAD + real 3D-sensor depth (same protocol, separate run) | yes | 90.4 ± 1.0 |

SAM object/background adds +3.8 AUROC, DINO as a second detector +2.8, CLIP as a third +2.3. Together they
close about 76 % of the gap to a 3D sensor.
Per-class numbers: `results/report.txt`; per seed: `results/per_seed.csv`.
Protocol: SegAD head trained on validation/good plus 10 defective test images per seed (official SegAD seeds),
evaluated on the remaining test images.

Ablation (`python improve.py fit && python improve.py eval`, `results/report_improve.txt`): starting from
SAM object/background, cropping each image to the SAM object ("zoom") gives 79.4, DINO 85.2, both 84.2.
Round 2 (`improve2.py`, `results/report_improve2.txt`), from v2: 384 px input 83.9, CLIP 87.5, both 86.2.

Known limitations: colour-clustered parts are not consistent across images (`results/parts_carrot_gland.png`);
tire (black on black) cannot be separated from the background; the VLM usually localises correctly but often
names the wrong defect type.

## Run

Data: MVTec 3D-AD, expected at `../GeoAD/data/mvtec3d/<class>/{train,validation,test}/...`.

```
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1   # optional laptop GPU guard
python pipeline.py fit         # SAM part maps + PatchCore (WRN50, DINO, CLIP) per class (~5 min/class, 8 GB laptop GPU)
python pipeline.py eval        # SegAD heads -> results/report.txt
python pipeline.py explain ../GeoAD/data/mvtec3d/dowel/test/bent/rgb/000.png   # -> results/explain_*.png
```

Models download from Hugging Face on first use (`facebook/sam-vit-base`, `facebook/dino-vitb8`,
`openai/clip-vit-base-patch16`, `Qwen/Qwen3-VL-2B-Instruct`).
