//! Tests for `fetch_models` (ports of `tests/test_fetch_models_archive.py`,
//! `tests/test_fetch_models_licences.py` and the fetch_models part of
//! `tests/test_build_scripts.py`). Nothing here touches the network: the transport is a fake.
//!
//! Not ported: the PyInstaller-spec half of `test_model_pins_agree_everywhere` (there is no spec
//! any more; the app's `MODEL_FILES` half is kept) and `test_sdist_without_build_tooling_*`
//! (there is no sdist).

use std::collections::VecDeque;
use std::fs;
use std::io::{Cursor, Read, Write};
use std::path::Path;
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::*;

fn sha(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

const MEMBERS: [(&str, &[u8]); 2] = [
    ("weights.tflite", b"landmark weights"),
    ("geometry.binarypb", b"canonical face"),
];

fn zip_bytes(members: &[(&str, &[u8])]) -> Vec<u8> {
    let mut buffer = Cursor::new(Vec::new());
    let mut bundle = zip::ZipWriter::new(&mut buffer);
    let options =
        zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    for (name, data) in members {
        bundle.start_file(*name, options).unwrap();
        bundle.write_all(data).unwrap();
    }
    bundle.finish().unwrap();
    buffer.into_inner()
}

fn bundle_model(archive: &[u8], members: &[(&str, &[u8])]) -> Model {
    Model {
        filename: "bundle.task".into(),
        url: "https://example.invalid/bundle.task".into(),
        sha256: sha(archive),
        license: "Apache-2.0".into(),
        homepage: "https://example.invalid".into(),
        description: "test bundle".into(),
        members: members
            .iter()
            .map(|(n, d)| Member::new(n, &sha(d)))
            .collect(),
    }
}

fn plain_model(payload: &[u8], filename: &str) -> Model {
    Model {
        filename: filename.into(),
        url: format!("https://example.invalid/{filename}"),
        sha256: sha(payload),
        license: "MIT".into(),
        homepage: "https://example.invalid".into(),
        description: "test model".into(),
        members: vec![],
    }
}

/// Replaces `urlopen` and `time.sleep`: each call pops the next outcome.
#[derive(Default)]
struct FakeTransport {
    outcomes: VecDeque<Result<Vec<u8>, FetchError>>,
    calls: Vec<String>,
}

impl FakeTransport {
    fn serving(bodies: Vec<Vec<u8>>) -> Self {
        Self {
            outcomes: bodies.into_iter().map(Ok).collect(),
            calls: vec![],
        }
    }
}

impl Transport for FakeTransport {
    fn open(&mut self, url: &str, _timeout: Duration) -> Result<Box<dyn Read>, FetchError> {
        self.calls.push(url.to_string());
        let outcome = self.outcomes.pop_front().expect("unexpected request");
        outcome.map(|body| Box::new(Cursor::new(body)) as Box<dyn Read>)
    }

    fn sleep(&mut self, _duration: Duration) {}
}

fn fetch(
    model: &Model,
    dest: &Path,
    transport: &mut FakeTransport,
) -> Result<PathBuf, DownloadError> {
    download(
        model,
        dest,
        transport,
        DownloadOptions::default(),
        &mut Vec::new(),
    )
}

fn names(dir: &Path) -> Vec<String> {
    let mut found: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

fn args(check: bool, force: bool, dest: &Path) -> Args {
    Args {
        check,
        force,
        dest: Some(dest.to_path_buf()),
        timeout: 60.0,
    }
}

/// Runs `fetch-models`; returns `(exit code, stdout, stderr)`.
fn run_main(args: &Args, models: &[Model], transport: &mut FakeTransport) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = run(args, models, transport, &mut out, &mut err);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

// ============================================================================ archive
#[test]
fn test_installed_files_of_a_bundle() {
    let model = bundle_model(&zip_bytes(&MEMBERS), &MEMBERS);
    let installed: Vec<&str> = model.installed().iter().map(|(n, _)| *n).collect();
    assert_eq!(
        installed,
        MEMBERS.iter().map(|(n, _)| *n).collect::<Vec<_>>()
    );
}

#[test]
fn test_bundle_download_installs_only_the_verified_members() {
    let tmp = tempfile::tempdir().unwrap();
    let archive = zip_bytes(&[MEMBERS[0], MEMBERS[1], ("unused.tflite", b"not needed")]);
    let model = bundle_model(&archive, &MEMBERS);
    let mut transport = FakeTransport::serving(vec![archive]);
    let first = fetch(&model, tmp.path(), &mut transport).unwrap();
    assert_eq!(first, tmp.path().join("weights.tflite"));
    assert_eq!(names(tmp.path()), ["geometry.binarypb", "weights.tflite"]);
    for (name, data) in MEMBERS {
        assert_eq!(fs::read(tmp.path().join(name)).unwrap(), data);
    }
    assert_eq!(verify(&model, tmp.path()), "ok");
}

#[test]
fn test_member_checksum_mismatch_installs_nothing() {
    let tmp = tempfile::tempdir().unwrap();
    let archive = zip_bytes(&[
        ("weights.tflite", b"tampered"),
        ("geometry.binarypb", b"canonical face"),
    ]);
    let model = bundle_model(&archive, &MEMBERS); // pins the genuine members
    let mut transport = FakeTransport::serving(vec![archive]);
    let error = fetch(&model, tmp.path(), &mut transport).unwrap_err();
    assert!(
        error.0.contains("checksum mismatch for weights.tflite"),
        "{error}"
    );
    assert!(names(tmp.path()).is_empty());
}

#[test]
fn test_missing_member() {
    let tmp = tempfile::tempdir().unwrap();
    let archive = zip_bytes(&[MEMBERS[0]]);
    let model = bundle_model(&archive, &MEMBERS);
    let mut transport = FakeTransport::serving(vec![archive]);
    let error = fetch(&model, tmp.path(), &mut transport).unwrap_err();
    assert!(error.0.contains("no member 'geometry.binarypb'"), "{error}");
    assert!(names(tmp.path()).is_empty());
}

#[test]
fn test_not_a_zip() {
    let tmp = tempfile::tempdir().unwrap();
    let archive = tmp.path().join("bundle.task");
    fs::write(&archive, b"definitely not a zip file").unwrap();
    let model = bundle_model(&fs::read(&archive).unwrap(), &MEMBERS);
    let dest = tmp.path().join("models");
    fs::create_dir(&dest).unwrap();
    let error = extract_members(&model, &archive, &dest).unwrap_err();
    assert!(error.0.contains("not a readable zip archive"), "{error}");
    assert!(names(&dest).is_empty());
}

#[test]
fn test_member_paths_never_escape_the_model_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let members: [(&str, &[u8]); 1] = [("../outside.bin", b"payload")];
    let archive_bytes = zip_bytes(&members);
    let archive = tmp.path().join("bundle.task");
    fs::write(&archive, &archive_bytes).unwrap();
    let dest = tmp.path().join("models");
    fs::create_dir(&dest).unwrap();
    let installed =
        extract_members(&bundle_model(&archive_bytes, &members), &archive, &dest).unwrap();
    assert_eq!(installed, [dest.join("outside.bin")]);
    assert!(!tmp.path().join("outside.bin").exists());
}

#[test]
fn test_real_bundle_pins_match_the_committed_models() {
    let all = models();
    let bundles: Vec<&Model> = all.iter().filter(|m| !m.members.is_empty()).collect();
    assert_eq!(bundles.len(), 1);
    let task = bundles[0];
    assert_eq!(task.filename, "face_landmarker.task");
    let member_names: Vec<&str> = task.members.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        member_names,
        [
            "face_landmarks_detector.tflite",
            "geometry_pipeline_metadata_landmarks.binarypb"
        ]
    );
    assert_eq!(verify(task, &default_dest()), "ok");
}

// ============================================================================ licences
fn model_dir() -> PathBuf {
    default_dest()
}

fn copy_models(dest: &Path, licences: bool) {
    for (name, _) in eye_tracker::vision::backends::MODEL_FILES {
        fs::copy(model_dir().join(name), dest.join(name)).unwrap();
    }
    if licences {
        fs::create_dir(dest.join("licenses")).unwrap();
        for text in &LICENCE_TEXTS {
            fs::copy(model_dir().join(text.path), dest.join(text.path)).unwrap();
        }
    }
}

#[test]
fn test_licence_texts_are_committed_next_to_the_models() {
    for text in &LICENCE_TEXTS {
        let path = model_dir().join(text.path);
        assert!(path.is_file(), "{}", text.path);
        assert_eq!(sha(&fs::read(&path).unwrap()), text.sha256, "{}", text.path);
        assert_eq!(verify_licence(text, &model_dir()), "ok");
    }
}

#[test]
fn test_every_model_file_is_covered_by_exactly_one_licence_text() {
    let mut covered: Vec<&str> = LICENCE_TEXTS
        .iter()
        .flat_map(|t| t.covers.iter().copied())
        .collect();
    covered.sort_unstable();
    let mut files: Vec<&str> = eye_tracker::vision::backends::MODEL_FILES
        .iter()
        .map(|(name, _)| *name)
        .collect();
    files.sort_unstable();
    assert_eq!(covered, files);
}

#[test]
fn test_licence_texts_say_what_they_must() {
    let by_name: Vec<(&str, String)> = LICENCE_TEXTS
        .iter()
        .map(|t| {
            let name = Path::new(t.path).file_name().unwrap().to_str().unwrap();
            // Bytes, not a newline-translating read: that would hide a CRLF.
            (
                name,
                String::from_utf8(fs::read(model_dir().join(t.path)).unwrap()).unwrap(),
            )
        })
        .collect();
    let get = |name: &str| &by_name.iter().find(|(n, _)| *n == name).unwrap().1;
    let apache = get("LICENSE-APACHE-2.0.txt");
    assert!(apache.contains("Apache License"));
    assert!(apache.contains("Version 2.0, January 2004"));
    assert!(apache.contains("END OF TERMS AND CONDITIONS"));
    let yunet = get("LICENSE-YUNET.txt");
    assert!(yunet.starts_with("MIT License\n"));
    assert!(yunet.contains("Copyright (c) 2020 Shiqi Yu"));
    assert!(
        yunet.contains("The above copyright notice and this permission notice shall be included")
    );
    for (_, text) in &by_name {
        // LF everywhere, so the pinned checksums hold on every OS
        assert!(!text.contains('\r'));
    }
}

#[test]
fn test_notice_references_every_licence_text() {
    let notice = fs::read_to_string(model_dir().join("NOTICE.md")).unwrap();
    for text in &LICENCE_TEXTS {
        // a working relative link
        assert!(
            notice.contains(&format!("]({})", text.path)),
            "{}",
            text.path
        );
        assert!(notice.contains(text.sha256), "{}", text.path);
    }
}

#[test]
fn test_check_passes_with_the_committed_files() {
    let tmp = tempfile::tempdir().unwrap();
    copy_models(tmp.path(), true);
    let (code, out, _err) = run_main(
        &args(true, false, tmp.path()),
        &models(),
        &mut FakeTransport::default(),
    );
    assert_eq!(code, 0);
    for text in &LICENCE_TEXTS {
        assert!(out.contains(&format!("licence  {}", text.path)));
    }
}

#[test]
fn test_check_fails_without_the_licence_texts() {
    let tmp = tempfile::tempdir().unwrap();
    copy_models(tmp.path(), false);
    let (code, out, err) = run_main(
        &args(true, false, tmp.path()),
        &models(),
        &mut FakeTransport::default(),
    );
    assert_eq!(code, 1);
    assert!(out.contains("missing  licenses/LICENSE-APACHE-2.0.txt"));
    assert!(out.contains("missing  licenses/LICENSE-YUNET.txt"));
    assert!(err.contains(&format!(
        "{} licence text(s) missing or altered",
        LICENCE_TEXTS.len()
    )));
    // the models themselves are fine
    assert!(!err.contains("model(s) missing"));
}

#[test]
fn test_check_fails_when_a_licence_text_was_altered() {
    let tmp = tempfile::tempdir().unwrap();
    copy_models(tmp.path(), true);
    let yunet = tmp.path().join("licenses/LICENSE-YUNET.txt");
    let text = fs::read_to_string(&yunet)
        .unwrap()
        .replace("Shiqi Yu", "Somebody Else");
    fs::write(&yunet, text).unwrap();
    let (code, out, err) = run_main(
        &args(true, false, tmp.path()),
        &models(),
        &mut FakeTransport::default(),
    );
    assert_eq!(code, 1);
    assert!(out.contains("altered  licenses/LICENSE-YUNET.txt"));
    assert!(err.contains("1 licence text(s) missing or altered"));
}

/// Fetching models into another folder is fine; the texts come from git, not the web.
#[test]
fn test_download_mode_only_warns_about_licence_texts() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"weights", "model.bin");
    fs::write(tmp.path().join("model.bin"), b"weights").unwrap(); // present: nothing to download
    // an empty queue panics on any request: licence texts are never downloaded
    let (code, _out, err) = run_main(
        &args(false, false, tmp.path()),
        &[model],
        &mut FakeTransport::default(),
    );
    assert_eq!(code, 0);
    assert!(err.contains("warning: 2 licence text(s) missing or altered"));
}

// ============================================================================ build scripts
#[test]
fn test_pinned_models_match_the_spec() {
    let pinned: Vec<(String, String)> = models()
        .into_iter()
        .map(|m| (m.filename, m.sha256))
        .collect();
    assert_eq!(
        pinned,
        [
            (
                "face_landmarker.task".to_string(),
                "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff".to_string()
            ),
            (
                "face_detection_yunet_2023mar.onnx".to_string(),
                "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4".to_string()
            ),
        ]
    );
    assert!(default_dest().ends_with("crates/eye-tracker/models"));
    assert!(models().iter().all(|m| m.url.starts_with("https://")));
}

/// fetch_models and the backends pin the same files.
#[test]
fn test_model_pins_agree_everywhere() {
    let all = models();
    let installed: Vec<(&str, &str)> = all.iter().flat_map(|m| m.installed()).collect();
    assert_eq!(installed, eye_tracker::vision::backends::MODEL_FILES);
    // The MediaPipe bundle is only a download source now; it never ships.
    assert!(
        !eye_tracker::vision::backends::MODEL_FILES
            .iter()
            .any(|(n, _)| *n == "face_landmarker.task")
    );
}

#[test]
fn test_verify_statuses() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"weights", "model.bin");
    assert_eq!(verify(&model, tmp.path()), "missing");
    fs::write(tmp.path().join(&model.filename), b"tampered").unwrap();
    assert_eq!(verify(&model, tmp.path()), "mismatch");
    fs::write(tmp.path().join(&model.filename), b"weights").unwrap();
    assert_eq!(verify(&model, tmp.path()), "ok");
}

#[test]
fn test_committed_models_are_valid() {
    let tmp = tempfile::tempdir().unwrap();
    let args = Args {
        dest: None,
        ..args(true, false, tmp.path())
    };
    let (code, out, _err) = run_main(&args, &models(), &mut FakeTransport::default());
    assert_eq!(code, 0);
    assert_eq!(out.matches("  ok ").count(), models().len());
}

#[test]
fn test_check_mode_reports_missing_models_without_downloading() {
    let tmp = tempfile::tempdir().unwrap();
    let mut transport = FakeTransport::default();
    let (code, out, err) = run_main(&args(true, false, tmp.path()), &models(), &mut transport);
    assert_eq!(code, 1);
    assert!(transport.calls.is_empty());
    assert!(out.contains("missing"));
    assert!(err.contains(&format!("{} model(s) missing or invalid", models().len())));
}

#[test]
fn test_download_writes_verified_file() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"good weights", "model.bin");
    let mut transport = FakeTransport::serving(vec![b"good weights".to_vec()]);
    let path = fetch(&model, tmp.path(), &mut transport).unwrap();
    assert_eq!(path, tmp.path().join(&model.filename));
    assert_eq!(fs::read(&path).unwrap(), b"good weights");
    assert_eq!(transport.calls, vec![model.url.clone()]);
    assert_eq!(names(tmp.path()), ["model.bin"]); // no *.part left
}

#[test]
fn test_download_rejects_checksum_mismatch_and_keeps_old_file() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"expected", "model.bin");
    fs::write(tmp.path().join(&model.filename), b"previous").unwrap();
    let mut transport = FakeTransport::serving(vec![b"evil".to_vec()]);
    let error = fetch(&model, tmp.path(), &mut transport).unwrap_err();
    assert!(error.0.contains("checksum mismatch"), "{error}");
    // Never retried, never replaced, no temporary file left behind.
    assert_eq!(transport.calls.len(), 1);
    assert_eq!(
        fs::read(tmp.path().join(&model.filename)).unwrap(),
        b"previous"
    );
    assert_eq!(names(tmp.path()), ["model.bin"]);
}

#[test]
fn test_download_retries_transient_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"weights", "model.bin");
    let mut transport = FakeTransport::default();
    transport
        .outcomes
        .push_back(Err(FetchError::Other("temporary failure".into())));
    transport
        .outcomes
        .push_back(Err(FetchError::Other("slow".into())));
    transport.outcomes.push_back(Ok(b"weights".to_vec()));
    let path = fetch(&model, tmp.path(), &mut transport).unwrap();
    assert_eq!(fs::read(path).unwrap(), b"weights");
    assert_eq!(transport.calls.len(), 3);
}

#[test]
fn test_download_gives_up_after_attempts() {
    let tmp = tempfile::tempdir().unwrap();
    let mut transport = FakeTransport::default();
    for _ in 0..2 {
        transport
            .outcomes
            .push_back(Err(FetchError::Other("down".into())));
    }
    let options = DownloadOptions {
        attempts: 2,
        ..DownloadOptions::default()
    };
    let error = download(
        &plain_model(b"x", "model.bin"),
        tmp.path(),
        &mut transport,
        options,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(error.0.contains("download failed"), "{error}");
    assert_eq!(transport.calls.len(), 2);
    assert!(names(tmp.path()).is_empty());
}

#[test]
fn test_download_does_not_retry_client_errors() {
    let tmp = tempfile::tempdir().unwrap();
    let model = plain_model(b"x", "model.bin");
    let mut transport = FakeTransport::default();
    transport.outcomes.push_back(Err(FetchError::Http(404)));
    let error = fetch(&model, tmp.path(), &mut transport).unwrap_err();
    assert!(error.0.contains("404"), "{error}");
    assert_eq!(transport.calls.len(), 1);
}

#[test]
fn test_download_refuses_oversized_responses() {
    let tmp = tempfile::tempdir().unwrap();
    let payload = vec![b'x'; 64];
    let mut transport = FakeTransport::serving(vec![payload.clone()]);
    let options = DownloadOptions {
        max_bytes: 10,
        ..DownloadOptions::default()
    };
    let error = download(
        &plain_model(&payload, "model.bin"),
        tmp.path(),
        &mut transport,
        options,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert!(error.0.contains("larger than"), "{error}");
    assert!(names(tmp.path()).is_empty());
}

#[test]
fn test_main_downloads_only_what_is_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let present = plain_model(b"a", "a.bin");
    let missing = plain_model(b"b", "b.bin");
    let pair = [present, missing.clone()];
    fs::write(tmp.path().join("a.bin"), b"a").unwrap();
    let mut transport = FakeTransport::serving(vec![b"b".to_vec()]);
    let (code, _, _) = run_main(&args(false, false, tmp.path()), &pair, &mut transport);
    assert_eq!(code, 0);
    assert_eq!(transport.calls, vec![missing.url.clone()]);
    assert_eq!(fs::read(tmp.path().join("b.bin")).unwrap(), b"b");

    // --force downloads everything again.
    transport.outcomes.push_back(Ok(b"a".to_vec()));
    transport.outcomes.push_back(Ok(b"b".to_vec()));
    let (code, _, _) = run_main(&args(false, true, tmp.path()), &pair, &mut transport);
    assert_eq!(code, 0);
    assert_eq!(transport.calls.len(), 3);
}
