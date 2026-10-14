"""face_landmarks_detector.tflite -> FaceLandmarks.mlpackage (tf2onnx -> onnx2torch -> coremltools).

Usage: python convert.py [model.tflite] [out.mlpackage]
Defaults: the repository's crates/eye-tracker/models/face_landmarks_detector.tflite and
crates/eye-tracker/models/FaceLandmarks.mlpackage.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx2torch import convert as onnx_to_torch
from onnx2torch.node_converters.activations import OnnxPReLU

warnings.filterwarnings("ignore")
models = (
    pathlib.Path(__file__).resolve().parents[2] / "crates" / "eye-tracker" / "models"
)
tflite = (
    sys.argv[1] if len(sys.argv) > 1 else str(models / "face_landmarks_detector.tflite")
)
out = sys.argv[2] if len(sys.argv) > 2 else str(models / "FaceLandmarks.mlpackage")

with tempfile.TemporaryDirectory() as tmp:
    onnx_path = str(pathlib.Path(tmp) / "face.onnx")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tf2onnx.convert",
            "--tflite",
            tflite,
            "--output",
            onnx_path,
            "--opset",
            "17",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # onnx2torch misses its fast path for (1,C,1,1) slopes and emits a masked-scatter graph (~17 ops per PRELU);
    # a plain per-channel F.prelu maps to one Core ML prelu op.
    OnnxPReLU.forward = lambda self, x, slope: torch.nn.functional.prelu(
        x, slope.view(-1)
    )
    net = onnx_to_torch(onnx.load(onnx_path)).eval()

# The converted graph takes NHWC like the tflite (1x256x256x3, RGB in [0, 1]), and so does the Core ML input:
# the app hands Core ML the same float buffer it hands tract.
traced = torch.jit.trace(net, torch.rand(1, 256, 256, 3))
ml = ct.convert(
    traced,
    inputs=[ct.TensorType(name="image", shape=(1, 256, 256, 3), dtype=np.float32)],
    outputs=[
        ct.TensorType(name=n, dtype=np.float32)
        for n in ("landmarks", "presence_logit", "aux_score")
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS13,
)
ml.short_description = "MediaPipe Face Landmarker landmark model (478x3 landmarks in 256x256 crop pixels + face presence logit)"
shutil.rmtree(out, ignore_errors=True)
ml.save(out)
print("saved", out)
