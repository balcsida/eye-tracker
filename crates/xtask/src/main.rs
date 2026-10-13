//! `cargo xtask <command>`: developer tooling (ports of `scripts/*.py`).

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "xtask", about = "Eye Tracker developer tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Fail if the app crate contains network code or writes frames to disk.
    CheckPrivacy,
    /// Download and verify the bundled face models.
    FetchModels,
    /// Generate docs/configuration.md from the settings metadata.
    GenConfigDocs,
    /// Render the application icons.
    MakeIcons,
    /// Build the website.
    BuildSite,
    /// Start a built bundle and check that it runs.
    SmokeTestBundle,
}

fn main() {
    // TODO(port): scripts/*.py
    let _command = Cli::parse().command;
    eprintln!("not ported yet");
    std::process::exit(1);
}
