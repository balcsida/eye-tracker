# FaceLandmarks: MediaPipe `face_landmarks_detector.tflite` -> Core ML

Converts the facemesh backend's landmark network into
`crates/eye-tracker/models/FaceLandmarks.mlpackage`, which the macOS build embeds and runs through
Core ML (see `docs/architecture.md`, "Inference engines"). The package is committed; rerun the
conversion only when the `.tflite` or this script changes.

Reproduce (macOS, Python 3.12, from the repository root):

```bash
python3.12 -m venv /tmp/coreml-venv && /tmp/coreml-venv/bin/pip install -r tools/coreml/requirements.txt
/tmp/coreml-venv/bin/python tools/coreml/convert.py   # rewrites crates/eye-tracker/models/FaceLandmarks.mlpackage
/tmp/coreml-venv/bin/python tools/coreml/check.py     # parity vs TFLite, compute plan, timing
```

Both scripts take optional `[model.tflite] [model.mlpackage]` arguments. `check.py` needs
`xcrun coremlcompiler` (Xcode command line tools) for the compute plan; the app itself compiles the
package in-process and needs no Xcode.

Route: tf2onnx `--tflite` -> onnx2torch (with `OnnxPReLU` patched to a per-channel `F.prelu`, see
below) -> `torch.jit.trace` -> coremltools (ML Program, float16 compute and weights, macOS 13).
Versions: `requirements.txt` (coremltools 9.0, tensorflow 2.21.0, tf2onnx 1.17.0, onnx 1.23.2,
onnx2torch 1.5.15, torch 2.14.1, numpy 2.5.4). coremltools warns that TF and torch are newer than
it was tested with; the conversion is exact in float32 (see below).

## TFLite model (no quantisation on its inputs and outputs)

| | name | shape | dtype |
|---|---|---|---|
| in | `input_12` | 1x256x256x3 NHWC | float32 |
| out | `Identity` | 1x1x1x1434 | float32 |
| out | `Identity_1` | 1x1x1x1 | float32 |
| out | `Identity_2` | 1x1 | float32 |

Input: an upright face crop, RGB, NHWC, 256x256, float32 in **[0, 1]** (pixel / 255).

## Core ML package contract

Input `image`: **TensorType** float32, shape [1, 256, 256, 3], NHWC, RGB in [0, 1], exactly the
TFLite input. The app passes the crop buffer it builds for tract unchanged; there is no scaling or
layout op in the model.

Outputs, all float32 `MLMultiArray`s (internal compute is float16), same layout and units as TFLite:

| Core ML name | shape | = TFLite | content |
|---|---|---|---|
| `landmarks` | [1,1,1,1434] | `Identity` | 478 x (x, y, z) interleaved: index `i*3 + {0,1,2}`; x, y in crop pixels of the 256x256 input (points may fall slightly outside), z relative depth in similar units |
| `presence_logit` | [1,1,1,1] | `Identity_1` | face-presence logit, no sigmoid; presence = 1 / (1 + exp(-v)), the app's threshold is 0.5 |
| `aux_score` | [1,1] | `Identity_2` | a second sigmoid head; not a reliable presence score (0.9999 on flat grey), unused |

## Results (`check.py`, macOS 27.0.1, arm64)

Parity over 20 synthetic inputs (6 noise, 6 gradients, 8 drawn face blobs), TFLite float32 vs
Core ML with compute units ALL:

| output | max abs diff | mean abs diff |
|---|---|---|
| landmarks (px, per coordinate) | 4.98 | 0.292 |
| presence_logit | 0.392 | 0.101 |
| sigmoid(presence_logit) | 0.00016 | 0.00001 |
| aux_score | 0.199 | 0.018 |

Worst landmark coordinate per group: face blobs 0.20 px, noise 0.77 px, gradients 4.98 px. A
float32 build of the same graph matches TFLite to 8e-4 px, so the conversion is exact; the rest is
float16 arithmetic through about 100 layers, largest on inputs far from any face. The app's tests
(`facemesh_backend_tests.rs`) hold drawn faces to 2 px against tract (measured 0.1 px).

Compute plan (`MLComputePlan`, preferred device per MIL op, units ALL): all 227 ops on the
Neural Engine (106 conv, 69 prelu, 34 add, 6 max_pool, 4 cast, 4 pad, 2 reshape, 1 transpose,
1 sigmoid), none on CPU or GPU, plus 860 `const` ops without a device. The stock onnx2torch PReLU
expands into gather/scatter ops that run on the GPU (132 ms per prediction); the patch in
`convert.py` maps it to one native `prelu` op.

Timing (200 predictions after 20 warm-ups, Python `MLModel.predict`, including the Python and
NumPy overhead): ALL median 0.67 ms, p95 0.83 ms; CPU_ONLY median 1.20 ms, p95 1.37 ms.
