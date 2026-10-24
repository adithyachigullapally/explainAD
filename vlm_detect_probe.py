"""Can the VLM alone say defective / not, zero-shot, with no per-class fit? Probe on MVTec 3D-AD test images.
Score = logit(Yes) - logit(No) of the first answer token. Run: python vlm_detect_probe.py [class ...]
-> results/vlm_detect.csv/.txt"""
import sys

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForImageTextToText, AutoProcessor

import pipeline as pl

proc = AutoProcessor.from_pretrained(pl.VLM_ID)
model = AutoModelForImageTextToText.from_pretrained(pl.VLM_ID, dtype=torch.bfloat16).to("cuda").eval()
yes, no = proc.tokenizer.encode("Yes")[0], proc.tokenizer.encode("No")[0]

rows = []
for cls in sys.argv[1:] or pl.CLASSES:
    q = (f"This is a photo of a {cls.replace('_', ' ')}. Does it have any defect or damage "
         "(crack, cut, hole, rot, contamination, discolouration, deformation)? Answer Yes or No.")
    for p in pl.paths(cls, "test"):
        img = Image.open(p).convert("RGB").resize((448, 448), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                       return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**inp).logits[0, -1].float()
        rows.append((cls, p.parts[-3], pl.uid(p), float(lg[yes] - lg[no])))
    print(cls, len(rows), flush=True)
r = pd.DataFrame(rows, columns=["cls", "kind", "uid", "score"])
r["bad"] = r.kind != "good"
r.to_csv(pl.RESULTS / "vlm_detect.csv", index=False)

per = r.groupby("cls").apply(lambda g: pd.Series({
    "n": len(g), "auroc": 100 * roc_auc_score(g.bad, g.score),
    "good_flagged": 100 * (g[~g.bad].score > 0).mean(), "bad_flagged": 100 * (g[g.bad].score > 0).mean()}),
    include_groups=False).round(1)
txt = "\n".join([f"Zero-shot VLM ({pl.VLM_ID}), MVTec 3D-AD test, flagged = score > 0 (%)", per.to_string(),
                 f"mean AUROC over classes: {per.auroc.mean():.1f}"])
(pl.RESULTS / "vlm_detect.txt").write_text(txt, encoding="utf-8")
print(txt)
