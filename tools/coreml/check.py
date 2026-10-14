"""Parity (TFLite vs Core ML), compute-plan device breakdown, and timing for FaceLandmarks.mlpackage.

Usage: python check.py [model.tflite] [model.mlpackage]
Defaults: the repository's crates/eye-tracker/models files (as in convert.py).
"""

import collections
import pathlib
import subprocess
import sys
import tempfile
import time
import warnings

import coremltools as ct
import numpy as np
import tensorflow as tf

warnings.filterwarnings("ignore")
models = (
    pathlib.Path(__file__).resolve().parents[2] / "crates" / "eye-tracker" / "models"
)
tflite = (
    sys.argv[1] if len(sys.argv) > 1 else str(models / "face_landmarks_detector.tflite")
)
pkg = sys.argv[2] if len(sys.argv) > 2 else str(models / "FaceLandmarks.mlpackage")
OUTS = {
    "Identity": "landmarks",
    "Identity_1": "presence_logit",
    "Identity_2": "aux_score",
}


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---- 20 synthetic RGB inputs in [0, 1], NHWC (the app feeds 8-bit pixels / 255, so they are quantised the same way)
yy, xx = np.mgrid[0:256, 0:256] / 255.0
rng = np.random.default_rng(0)


def blob(shift):
    img = np.full((256, 256, 3), 0.25)
    cx, cy = 0.5 + shift[0], 0.5 + shift[1]
    img[((xx - cx) / 0.32) ** 2 + ((yy - cy) / 0.42) ** 2 < 1] = (0.85, 0.68, 0.58)
    for ex in (-0.12, 0.12):
        img[((xx - cx - ex) / 0.05) ** 2 + ((yy - cy + 0.08) / 0.025) ** 2 < 1] = 0.1
    img[((xx - cx) / 0.1) ** 2 + ((yy - cy - 0.18) / 0.02) ** 2 < 1] = (0.6, 0.2, 0.2)
    return img


inputs = [rng.random((256, 256, 3)) for _ in range(6)]
inputs += [
    np.stack([a * xx + b * yy + c] * 3, -1).clip(0, 1)
    for a, b, c in [
        (1, 0, 0),
        (0, 1, 0),
        (0.5, 0.5, 0),
        (-0.5, 0.5, 0.5),
        (0.3, -0.7, 0.8),
        (1, 1, -0.5),
    ]
]
inputs += [
    blob(s)
    for s in [
        (0, 0),
        (0.05, 0),
        (-0.05, 0.03),
        (0, -0.05),
        (0.04, 0.04),
        (-0.03, -0.04),
        (0, 0.06),
        (0.06, -0.02),
    ]
]
tensors = [
    ((np.clip(a, 0, 1) * 255).round() / 255).astype(np.float32)[None] for a in inputs
]
assert len(tensors) == 20

it = tf.lite.Interpreter(model_path=tflite)
it.allocate_tensors()
tin = it.get_input_details()[0]["index"]
tout = {d["name"]: d["index"] for d in it.get_output_details()}


def run_tflite(x):
    it.set_tensor(tin, x)
    it.invoke()
    return {OUTS[k]: it.get_tensor(i).reshape(-1) for k, i in tout.items()}


ml = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.ALL)
diffs = collections.defaultdict(list)
groups = ["noise"] * 6 + ["gradient"] * 6 + ["face-blob"] * 8
by_group = collections.defaultdict(list)  # group -> [max landmark abs diff per input]
for g, x in zip(groups, tensors, strict=False):
    t = run_tflite(x)
    c = ml.predict({"image": x})
    for k in t:
        diffs[k].append(np.abs(t[k] - np.asarray(c[k]).reshape(-1)))
    by_group[g].append(diffs["landmarks"][-1].max())
    diffs["presence=sigmoid(presence_logit)"].append(
        np.abs(
            sigmoid(t["presence_logit"])
            - sigmoid(np.asarray(c["presence_logit"]).reshape(-1))
        )
    )
print("== parity over 20 inputs (abs diff, TFLite fp32 vs Core ML fp16, units ALL)")
for k, v in diffs.items():
    v = np.concatenate(v)
    print(f"{k:34s} max {v.max():.5f}  mean {v.mean():.5f}")
lm = np.concatenate(diffs["landmarks"])
for g, v in by_group.items():
    print(
        f"landmarks max abs diff, {g:9s} inputs: worst {max(v):.3f} px, per-input {np.round(v, 2).tolist()}"
    )
pres_ok = np.concatenate(diffs["presence=sigmoid(presence_logit)"]).max() < 0.05
print(
    "presence (<0.05):",
    "PASS" if pres_ok else "FAIL",
    "| landmarks (<1 px): face-blob",
    "PASS" if max(by_group["face-blob"]) < 1 else "FAIL",
    "| all 20",
    "PASS" if lm.max() < 1 else "FAIL",
)

# ---- compute plan (needs xcrun coremlcompiler, from the Xcode command line tools)
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", pkg, tmp],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    compiled = str(pathlib.Path(tmp) / pathlib.Path(pkg).with_suffix(".mlmodelc").name)
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        path=compiled, compute_units=ct.ComputeUnit.ALL
    )
    counts = collections.defaultdict(collections.Counter)
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        dev = (
            "no-device (const/metadata)"
            if usage is None
            else type(usage.preferred_compute_device)
            .__name__.replace("ML", "")
            .replace("ComputeDevice", "")
        )
        counts[dev][op.operator_name] += 1
print("== compute plan (preferred device per MIL op, units ALL)")
for dev, c in counts.items():
    print(f"{dev:28s} {sum(c.values()):4d}  {dict(c.most_common())}")

# ---- timing
frame = tensors[-1]
for name, cu in (("ALL", ct.ComputeUnit.ALL), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)):
    m = ct.models.MLModel(pkg, compute_units=cu)
    for _ in range(20):
        m.predict({"image": frame})
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        m.predict({"image": frame})
        ts.append((time.perf_counter() - t0) * 1e3)
    print(
        f"== timing {name:9s} median {np.median(ts):.2f} ms  p95 {np.percentile(ts, 95):.2f} ms (200 runs after 20 warm-up; includes python overhead)"
    )
