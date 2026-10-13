# Vendored crates

## tract-tflite 0.23.8

Upstream: [`tract-tflite` 0.23.8 on crates.io](https://crates.io/crates/tract-tflite/0.23.8)
([sonos/tract](https://github.com/sonos/tract), `MIT OR Apache-2.0`), unpacked unchanged except
for `src/ops/nn.rs`. The workspace uses it through `[patch.crates-io]` in the root `Cargo.toml`.
It is a path dependency, not a workspace member, so `cargo fmt`, `cargo clippy` and
`cargo test` in CI do not touch it.

### Why

The MediaPipe face landmark model (`crates/eye-tracker/models`) is a TFLite file that stock
tract-tflite cannot load. Two operators are missing:

* `DEQUANTIZE`: the model stores its weights as float16 and converts them to float32 at load
  time. Only the float flavour is implemented (a constant is cast at load time, anything else
  gets a `Cast` node); quantised integer inputs are rejected.
* `PRELU`: wired as `max(x, 0) + alpha * min(x, 0)` from tract's own math ops.

Both are registered in `src/ops/nn.rs` and marked with `// patch:` comments. `vendor/tract-tflite.patch`
is the whole difference to the published crate.

### Regenerate `tract-tflite.patch`

Run from the repository root, after changing the vendored sources:

```sh
tmp=$(mktemp -d)
mkdir "$tmp/crates.io"
curl -sL https://crates.io/api/v1/crates/tract-tflite/0.23.8/download | tar xz -C "$tmp/crates.io"
ln -s "$PWD/vendor" "$tmp/vendor"
(cd "$tmp" && diff -ru crates.io/tract-tflite-0.23.8/src vendor/tract-tflite/src) > vendor/tract-tflite.patch
rm -r "$tmp"
```

`diff` exits with 1 when the trees differ; that is expected.

### When to drop it

When a tract release includes TFLite `DEQUANTIZE` (float) and `PRELU` deserialisation: bump
`tract-tflite` and `tract-onnx` together in `[workspace.dependencies]`, delete `vendor/` and the
`[patch.crates-io]` section, and check that the face landmark model still loads (`cargo test`).
