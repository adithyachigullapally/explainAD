"""Web demo: streamlit run app.py"""
import json
import random

import streamlit as st

import pipeline as pl
from vlm_eval import first_type

st.set_page_config(page_title="ExplainAD", layout="wide")
st.title("ExplainAD: find the defect, show where, say what it is")

# (images, fitted models, fit/eval script) per dataset; the pipeline reads MV and CACHE at call time (as visa.py does)
DATASETS = {"MVTec 3D-AD": (pl.ROOT / "data" / "mvtec3d", pl.ROOT / "cache", "pipeline.py"),
            "VisA": (pl.ROOT / "data" / "visa", pl.ROOT / "cache_visa", "visa.py")}
ds = st.sidebar.radio("Dataset", list(DATASETS), horizontal=True)
pl.MV, pl.CACHE, script = DATASETS[ds]
has_depth = ds == "MVTec 3D-AD"   # VisA has no 3D scans: camera mode only
classes = sorted(p.name for p in pl.MV.iterdir() if p.is_dir()) if pl.MV.is_dir() else []
cls = st.sidebar.selectbox("Object class", classes)
if cls is None or not (pl.CACHE / f"head_{cls}_train.json").exists():
    st.error(f"No fitted model found. Run `python {script} fit` and `python {script} eval` first (see README).")
    st.stop()
src = st.sidebar.radio("Image", [f"{ds} test sample", "Upload"])
rgb = depth = truth = None

if src != "Upload":
    # only test images the SegAD head never saw, shuffled so the position does not give the answer away;
    # the ground truth is shown after the verdict, not before
    used = set(json.loads((pl.CACHE / f"head_{cls}_train.json").read_text()))
    files = [p for p in pl.paths(cls, "test") if pl.uid(p) not in used]
    random.Random(0).shuffle(files)
    f = files[st.sidebar.selectbox("Image", range(len(files)), format_func=lambda i: f"image {i + 1}")]
    scanner = has_depth and st.sidebar.toggle("3D mode (use scanner depth)")
    rgb, truth = pl.load_rgb(f), f.parts[-3]
    if scanner:
        try:
            depth = pl.load_depth(f)
        except Exception as e:   # empty or unreadable scan: fall back instead of crashing
            st.warning(f"No usable depth for this image ({e}); running in camera mode.")
else:
    up = st.sidebar.file_uploader("Photo of the object", type=["png", "jpg", "jpeg"])
    if up:
        rgb = pl.load_rgb(up)   # same resize as the dataset images (was bicubic here, bilinear there)
    st.sidebar.caption(f"Camera mode only. The model is fitted per class on {ds} images: "
                       "results are only meaningful for the same object and a similar camera setup.")

mode = "3D" if depth is not None else "camera"
# starts at the class's calibrated value (python pipeline.py calibrate); the key resets it when class or mode changes
threshold = st.sidebar.slider("Defect threshold", 0.05, 0.95, pl.demo_threshold(cls, mode), 0.05, key=f"thr_{cls}_{mode}",
                              help="Probability at or above this is called BAD. Default: calibrated per class on the "
                                   "head's training images; lower it to catch more defects, raise it for fewer false alarms.")
if rgb is None:
    st.info("Choose an image.")
    st.stop()
st.sidebar.image(rgb, caption="selected image")
if st.button("Inspect", type="primary"):
    with st.spinner("SAM, three detectors, SegAD head, then the VLM if a defect is found..."):
        out = pl.RESULTS / "app_panel.png"
        res = pl.inspect(rgb, cls, depth, threshold, out)
    verdict = f"BAD: {first_type(res['description'])}" if res["defective"] else "GOOD"
    (st.error if res["defective"] else st.success)(
        f"Model says: {verdict}  |  p = {res['defect_probability']:.2f}  |  {res['mode']} mode")
    if truth:
        st.info(f"Actual (dataset label): {truth}")
    if res.get("description"):
        st.write("**Description (Qwen3-VL):** " + res["description"])
        st.caption("The detector decides and localises; the language model only describes. "
                   "Measured: the box lands on the defect ~49 % of the time and the type is right ~41 % (README). Treat it as a hint.")
    st.image(str(out), width="stretch")
