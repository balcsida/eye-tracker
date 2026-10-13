//! `cargo xtask <command>`: developer tooling (ports of `scripts/*.py`).

mod build_site;
mod check_privacy;
mod fetch_models;
mod gen_config_docs;
mod make_icons;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// The repository root (xtask lives in `crates/xtask`).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Parser)]
#[command(name = "xtask", about = "Eye Tracker developer tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fail if the app crate contains network code or writes frames to disk.
    CheckPrivacy(check_privacy::Args),
    /// Download and verify the bundled face models.
    FetchModels(fetch_models::Args),
    /// Generate docs/configuration.md from the settings metadata.
    GenConfigDocs(gen_config_docs::Args),
    /// Render the application icons.
    MakeIcons(make_icons::Args),
    /// Build the website.
    BuildSite(build_site::Args),
    /// Start a built bundle and check that it runs.
    SmokeTestBundle,
}

fn main() {
    let code = match Cli::parse().command {
        Command::FetchModels(args) => fetch_models::main(&args),
        Command::GenConfigDocs(args) => gen_config_docs::main(&args),
        Command::MakeIcons(args) => make_icons::main(&args),
        Command::BuildSite(args) => build_site::main(&args),
        Command::CheckPrivacy(args) => check_privacy::main(&args),
        Command::SmokeTestBundle => {
            // TODO(port): scripts/smoke_test_bundle.py
            eprintln!("not ported yet");
            1
        }
    };
    std::process::exit(code);
}
