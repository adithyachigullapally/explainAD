# ExplainAD

Factory defect inspection that works from the colour image of a fixed inspection camera and becomes more accurate when a 3D
scanner is also available, built from published models:

1. **SAM** (Kirillov et al., *Segment Anything*, ICCV 2023) separates the object from the background.
2. **SegAD** (Baitieva et al., *Supervised Anomaly Detection for Complex Industrial Images*, CVPR 2024)
   combines three **PatchCore** (Roth et al., CVPR 2022) anomaly maps, on WideResNet-50 features, on
   **DINO** ViT-B/8 features (Caron et al., *Emerging Properties in Self-Supervised Vision Transformers*,
   ICCV 2021) and on **CLIP** ViT-B/16 features (Radford et al., *Learning Transferable Visual Models From
   Natural Language Supervision*, ICML 2021), and decides whether the part is defective and where.
   With a 3D scanner (3D mode), a fourth PatchCore map on the measured depth image is added.
3. **Qwen3-VL-2B-Instruct** describes the flagged defect in one sentence (demo only; not part of the score).

## Result (MVTec 3D-AD, 10 classes × 10 seeds, image AUROC)

| Setup | 3D sensor needed | AUROC |
|---|---|---|
| SegAD, 1 segment (whole image) | no | 78.6 ± 1.2 |
| SegAD + SAM parts (colour-clustered) | no | 78.7 ± 1.1 |
| SegAD + SAM object/background (v1) | no | 82.4 ± 0.9 |
| SegAD + SAM object/background + DINO (v2) | no | 85.2 ± 1.1 |
| **ExplainAD camera mode: + CLIP (v3)** | **no** | **87.5 ± 0.9** |
| **ExplainAD 3D mode: + real depth (v4)** | **yes** | **92.1 ± 1.3** |
| SegAD, 1 segment + real 3D-sensor depth | yes | 90.3 ± 0.9 |

SAM object/background adds +3.8 AUROC, DINO as a second detector +2.8, CLIP as a third +2.3. Adding the 3D
scanner's depth adds another +4.6 (largest on potato, cookie and cable gland).
Per-class numbers: `results/report.txt`; per seed: `results/per_seed.csv`.
Protocol: SegAD head trained on validation/good plus 10 defective test images per seed (official SegAD seeds),
evaluated on the remaining test images.

Ablation (`python improve.py fit && python improve.py eval`, `results/report_improve.txt`): starting from
SAM object/background, cropping each image to the SAM object ("zoom") gives 79.4, DINO 85.2, both 84.2.
Round 2 (`improve2.py`, `results/report_improve2.txt`), from v2: 384 px input 83.9, CLIP 87.5, both 86.2.
More labelled defects (`python more_defects.py`, `results/more_defects.txt`; same held-out 30 % of defects in every
setting, so not the SegAD protocol): 10 / 60 % / 70 % of the defective images give 87.1 / 89.2 / 89.4 in camera mode
and 92.0 / 93.6 / 93.9 in 3D mode. A fitted threshold is no better than 0.5.

Known limitations: colour-clustered parts are not consistent across images (`results/parts_carrot_gland.png`);
tire (black on black) cannot be separated from the background; the VLM explanation is weak: on 316 held-out
defective images (`python vlm_eval.py`, `results/vlm_eval.txt`) the box centre lies on the defect in 49.1 % (box from
the DINO map; 37.7 % with the WRN50 map used before, map chosen on 632 other defective images), and a keyword match
finds the right type in 40.8 % (lenient; bent, cut and open are near 0 %). The better box did not improve the
description, and neither did showing the VLM a defect-free reference image (34.6 % vs 41.0 %, 78 images) or the
original-resolution image with a close-up of the box (46.2 % vs 41.0 %, 78 images, not significant). At threshold 0.5 the false-alarm
rate is high on some classes (tire 66 %, cookie 48 %; `results/operating_point.txt`). Each class's detector is tied
to its capture setup: on 74 phone photos of carrots (other camera, lighting and carrot variety) it scored 45.2 AUROC,
chance level. It suits a fixed inspection station and has to be refitted for a new setup (expected to work, not tested).
Zero-shot Qwen3-VL as a yes/no detector (`python vlm_detect_probe.py`, `results/vlm_detect.txt`) reaches 73.0 on the
10 classes, so the language model alone is not a substitute.

## Second dataset: VisA (12 classes × 10 seeds, image AUROC, camera mode only)

Protocol fixed in `visa.py` before any number was seen (the SegAD paper's VisA benchmark, `1cls` split); run once.

| Setup | AUROC |
|---|---|
| SegAD, 1 segment (whole image) | 86.9 ± 0.9 |
| SegAD + SAM object/background | 88.4 ± 0.7 |
| + DINO | 91.5 ± 0.8 |
| **+ CLIP (ExplainAD camera mode)** | **93.9 ± 0.4** |

The MVTec 3D-AD ladder holds: SAM object/background +1.5, DINO +3.0, CLIP +2.4.
Per-class numbers: `results/report_visa.txt`; per seed: `results/per_seed_visa.csv`.
Not comparable row by row with the SegAD paper's VisA numbers (96.1-98.4), which use other base detectors.

Known limitations: the object mask is wrong on candle and capsules (`results/visa_masks.png`: "object" covers
93 % and 87 % of the image, the border rule picked the wrong cluster); not corrected, since the protocol was
fixed beforehand. Weakest classes: capsules 78.5, macaroni2 86.4, macaroni1 90.2. At threshold 0.5 recall is low
on capsules (39 %), pcb3 (50 %) and macaroni2 (50 %) with false alarms under 7 % everywhere
(`results/operating_point_visa.txt`).

## Run

Data: MVTec 3D-AD (CC BY-NC-SA 4.0, download from mvtec.com), extracted to `data/mvtec3d/<class>/{train,validation,test}/...`.

Needs an NVIDIA GPU with CUDA (8 GB is enough) for `fit`, `explain`, the web demo and the VLM scripts; `eval`,
`calibrate` and `more_defects.py` run on the CPU from the cache that `fit` writes.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130   # plain PyPI gives CPU-only torch on Windows
powershell -ExecutionPolicy Bypass -File scripts\watchdog.ps1   # optional laptop GPU guard
python pipeline.py fit         # SAM part maps + PatchCore (WRN50, DINO, CLIP, real depth) per class (~5 min/class, 8 GB laptop GPU)
python pipeline.py eval        # SegAD heads for camera and 3D modes -> results/report.txt
python pipeline.py calibrate   # per-class demo threshold -> cache/thresholds.json (else 0.5)
python pipeline.py explain data/mvtec3d/dowel/test/bent/rgb/000.png   # camera-mode demo -> results/explain_*.png (add --3d for 3D mode)
streamlit run app.py           # web demo: pick a dataset, class and image (or upload), camera or 3D mode
```

VisA (optional second dataset; extract `VisA_20220922.tar` to `data/visa_raw`):

```
python visa.py prepare     # data/visa_raw -> data/visa in the pipeline's folder layout (hard links)
python visa.py fit         # SAM part maps + PatchCore (WRN50, DINO, CLIP) per class (~10 min/class)
python visa.py masks       # results/visa_masks.png, looked at before any score
python visa.py eval        # SegAD camera heads -> results/report_visa.txt
python visa.py calibrate   # per-class demo threshold -> cache_visa/thresholds.json
```

Models download from Hugging Face on first use (`facebook/sam-vit-base`, `facebook/dino-vitb8`,
`openai/clip-vit-base-patch16`, `Qwen/Qwen3-VL-2B-Instruct`).
