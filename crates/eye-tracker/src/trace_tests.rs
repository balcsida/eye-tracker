// Ported from tests/test_trace.py. Not ported: the `TraceWriter.number("x")` and
// `TraceWriter.vector(5)` assertions of test_helpers_handle_bad_values; the Rust signatures only
// accept numbers and slices, so those inputs cannot be written.

use serde_json::{Value, json};

use super::*;

fn lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

#[test]
fn test_writes_json_lines_and_rounds() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sub").join("trace.jsonl");
    let writer = TraceWriter::new(&path);
    writer.write(&json!({
        "t": TraceWriter::number(Some(1.23456789), 3),
        "f": TraceWriter::vector(Some(&[0.1, f64::NAN]), DEFAULT_DIGITS),
    }));
    writer.write(&json!({"t": 2.0, "f": null}));
    writer.close();
    let parsed: Vec<Value> = lines(&path)
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        parsed,
        vec![
            json!({"t": 1.235, "f": [0.1, null]}),
            json!({"t": 2.0, "f": null})
        ]
    );
}

#[test]
fn test_line_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("t.jsonl");
    let writer = TraceWriter::with_max_lines(&path, 3);
    for i in 0..10 {
        writer.write(&json!({"i": i}));
    }
    writer.close();
    assert_eq!(lines(&path).len(), 3);
}

#[test]
fn test_unwritable_path_disables_quietly() {
    let tmp = tempfile::tempdir().unwrap();
    let blocker = tmp.path().join("file");
    fs::write(&blocker, "x").unwrap();
    let writer = TraceWriter::new(blocker.join("trace.jsonl")); // parent is a file
    writer.write(&json!({"a": 1}));
    writer.write(&json!({"a": 2}));
    writer.close(); // no panic
    assert!(!blocker.join("trace.jsonl").exists());
}

#[test]
fn test_helpers_handle_bad_values() {
    assert_eq!(TraceWriter::number(None, DEFAULT_DIGITS), None);
    assert_eq!(
        TraceWriter::number(Some(f64::INFINITY), DEFAULT_DIGITS),
        None
    );
    assert_eq!(TraceWriter::vector(None, DEFAULT_DIGITS), None);
}
