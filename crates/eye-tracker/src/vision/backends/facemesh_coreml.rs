//! Core ML engine for the facemesh backend's landmark network (macOS only).
//!
//! `models/FaceLandmarks.mlpackage` is `face_landmarks_detector.tflite` converted by
//! `tools/coreml/convert.py`: an ML Program with float16 weights whose input `image` is the same
//! 1x256x256x3 NHWC float crop in 0..1 that tract gets, so the crop buffer is handed over as it
//! is. Its outputs `landmarks` (1434 floats: 478 x, y, z in crop pixels) and `presence_logit` are
//! the TFLite model's `Identity` and `Identity_1`.
//!
//! At first use the embedded package is written to `<data dir>/coreml/FaceLandmarks-<hash>/`,
//! compiled there by Core ML in-process (no Xcode needed), and the compiled
//! `FaceLandmarks.mlmodelc` is kept next to it, so later starts only load it. A changed model has
//! another hash and is compiled again. The model is loaded with compute units "all": the Neural
//! Engine where the Mac has one.

use std::cell::RefCell;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::mpsc;
use std::time::Duration;

use block2::{RcBlock, StackBlock};
use objc2::AnyThread;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType,
};
use objc2_foundation::{
    NSArray, NSDictionary, NSError, NSFileManager, NSInteger, NSNumber, NSString, NSURL,
};
use sha2::{Digest, Sha256};
use tract_tflite::prelude::Tensor;

use super::COREML_LANDMARK_PACKAGE;
use super::facemesh_backend::INPUT_SIZE;

const PACKAGE_DIR: &str = "FaceLandmarks.mlpackage";
const COMPILED_DIR: &str = "FaceLandmarks.mlmodelc";
const LANDMARK_VALUES: usize = 478 * 3;
/// Compiling takes a second or two; a compiler that has not answered by then never will.
const COMPILE_TIMEOUT: Duration = Duration::from_secs(120);

/// The landmark network loaded into Core ML.
pub struct CoreMlNet {
    model: Retained<MLModel>,
}

impl CoreMlNet {
    /// Load the embedded package from its compiled copy under `root` (the data directory),
    /// writing and compiling it first when there is none or it does not load.
    pub fn load(root: &Path) -> Result<Self, String> {
        let dir = model_dir(root);
        let compiled = dir.join(COMPILED_DIR);
        if compiled.is_dir() {
            match load_compiled(&compiled) {
                Ok(model) => return Ok(Self { model }),
                Err(err) => {
                    log::debug!("Compiling the Core ML model again: {err}");
                    let _ = std::fs::remove_dir_all(&compiled);
                }
            }
        }
        let package = dir.join(PACKAGE_DIR);
        for (name, contents) in COREML_LANDMARK_PACKAGE {
            let path = package.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            std::fs::write(&path, contents).map_err(|e| format!("{}: {e}", path.display()))?;
        }
        compile(&package, &compiled)?;
        load_compiled(&compiled).map(|model| Self { model })
    }

    /// Raw landmarks (478 x 3, network-input pixels) and the presence logit for an NHWC RGB
    /// input in 0..1, like `LandmarkNet::forward`.
    pub fn forward(&self, input: &Tensor) -> Result<(Vec<f32>, f32), String> {
        let view = input
            .to_plain_array_view::<f32>()
            .map_err(|e| e.to_string())?;
        let data = view
            .to_slice()
            .filter(|d| d.len() == INPUT_SIZE * INPUT_SIZE * 3)
            .ok_or_else(|| format!("unexpected input shape {:?}", view.shape()))?;
        let number = |n: usize| NSNumber::new_usize(n);
        let shape = NSArray::from_retained_slice(&[1, INPUT_SIZE, INPUT_SIZE, 3].map(number));
        let strides = NSArray::from_retained_slice(
            &[INPUT_SIZE * INPUT_SIZE * 3, INPUT_SIZE * 3, 3, 1].map(number),
        );
        // Everything Core ML autoreleases goes before `data` (borrowed from `input`) can.
        autoreleasepool(|_| {
            let pointer = NonNull::from(data).cast::<c_void>();
            // SAFETY: `pointer` is `data`, float32 in exactly the given shape and strides, and
            // outlives the array: it is released when this pool ends. Core ML only reads inputs,
            // and without a deallocator it never frees the buffer.
            let array = unsafe {
                MLMultiArray::initWithDataPointer_shape_dataType_strides_deallocator_error(
                    MLMultiArray::alloc(),
                    pointer,
                    &shape,
                    MLMultiArrayDataType::Float32,
                    &strides,
                    None,
                )
            }
            .map_err(|e| describe(&e))?;
            // SAFETY: a feature value wrapping a valid multi-array.
            let value = unsafe { MLFeatureValue::featureValueWithMultiArray(&array) };
            let features = NSDictionary::from_retained_objects(
                &[&*NSString::from_str("image")],
                &[Retained::into_super(Retained::into_super(value))],
            );
            // SAFETY: the dictionary maps feature names to feature values, as required.
            let provider = unsafe {
                MLDictionaryFeatureProvider::initWithDictionary_error(
                    MLDictionaryFeatureProvider::alloc(),
                    &features,
                )
            }
            .map_err(|e| describe(&e))?;
            // SAFETY: a loaded model and a provider with its one input.
            let outputs = unsafe {
                self.model
                    .predictionFromFeatures_error(ProtocolObject::from_ref(&*provider))
            }
            .map_err(|e| describe(&e))?;
            let landmarks = read_output(&outputs, "landmarks", LANDMARK_VALUES)?;
            let logit = read_output(&outputs, "presence_logit", 1)?;
            Ok((landmarks, logit[0]))
        })
    }
}

/// Where the package and its compiled copy live: named by the package's hash, so a new model
/// is never mistaken for the cached one.
fn model_dir(root: &Path) -> PathBuf {
    let mut hash = Sha256::new();
    for (name, contents) in COREML_LANDMARK_PACKAGE {
        hash.update(name.as_bytes());
        hash.update(contents);
    }
    let hex = hex::encode(hash.finalize());
    root.join("coreml")
        .join(format!("FaceLandmarks-{}", &hex[..16]))
}

fn file_url(path: &Path) -> Retained<NSURL> {
    NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()))
}

fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// Compile `package` and move the result to `compiled`.
fn compile(package: &Path, compiled: &Path) -> Result<(), String> {
    let (done, finished) = mpsc::channel();
    let target = file_url(compiled);
    let handler = RcBlock::new(move |url: *mut NSURL, error: *mut NSError| {
        // SAFETY: Core ML passes the compiled model's temporary location, or nil and an error;
        // both are valid during this call.
        let (url, error) = unsafe { (url.as_ref(), error.as_ref()) };
        // The temporary copy may be removed once this handler returns, so it moves now.
        let result = match url {
            Some(url) => NSFileManager::defaultManager()
                .moveItemAtURL_toURL_error(url, &target)
                .map_err(|e| describe(&e)),
            None => Err(error.map_or_else(|| "compilation failed".into(), describe)),
        };
        let _ = done.send(result);
    });
    // SAFETY: a file URL and a completion block, which Core ML keeps alive until it calls it.
    unsafe { MLModel::compileModelAtURL_completionHandler(&file_url(package), &handler) };
    finished
        .recv_timeout(COMPILE_TIMEOUT)
        .map_err(|_| "Core ML did not finish compiling the model".to_string())?
}

fn load_compiled(compiled: &Path) -> Result<Retained<MLModel>, String> {
    // SAFETY: a fresh configuration, set up before the model takes it.
    let configuration = unsafe { MLModelConfiguration::new() };
    // SAFETY: as above.
    unsafe { configuration.setComputeUnits(MLComputeUnits::All) };
    let url = NSURL::fileURLWithPath(&NSString::from_str(&compiled.to_string_lossy()));
    // SAFETY: a file URL of a compiled model and a valid configuration.
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &configuration) }
        .map_err(|e| describe(&e))
}

/// The first `len` values of a float32 output whose innermost dimension is contiguous (all
/// outer dimensions of this model's outputs are 1).
fn read_output(
    outputs: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    // SAFETY: looking up a feature by name; nil when the model has no such output.
    let value = unsafe { outputs.featureValueForName(&NSString::from_str(name)) };
    // SAFETY: nil unless the feature is a multi-array.
    let array = value
        .and_then(|v| unsafe { v.multiArrayValue() })
        .ok_or_else(|| format!("no output {name}"))?;
    // SAFETY: plain property reads of a valid multi-array.
    let (data_type, count, strides) = unsafe { (array.dataType(), array.count(), array.strides()) };
    let innermost = strides.lastObject().map(|s| s.as_isize());
    if data_type != MLMultiArrayDataType::Float32
        || count != len as NSInteger
        || innermost != Some(1)
    {
        return Err(format!("unexpected output {name} (not the landmark model)"));
    }
    let values = RefCell::new(Vec::with_capacity(len));
    let read = StackBlock::new(|bytes: NonNull<c_void>, size: NSInteger| {
        let available = usize::try_from(size).unwrap_or(0) / size_of::<f32>();
        // SAFETY: Core ML hands the array's float32 storage, `size` bytes, valid during the call.
        let floats = unsafe { std::slice::from_raw_parts(bytes.cast::<f32>().as_ptr(), available) };
        values
            .borrow_mut()
            .extend_from_slice(&floats[..len.min(available)]);
    });
    // SAFETY: the block only reads within the size Core ML reports.
    unsafe { array.getBytesWithHandler(&read) };
    let values = values.into_inner();
    if values.len() == len {
        Ok(values)
    } else {
        Err(format!("output {name} is too short"))
    }
}
