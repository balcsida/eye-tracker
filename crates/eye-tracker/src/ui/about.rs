//! About dialog: version, links, licence, privacy statement and third-party notices.
//!
//! [`system_info`] and the texts are plain functions and constants; [`AboutDialog`] draws them
//! as an egui viewport. No controller is needed.

use egui::{Context, TextureHandle, TextureOptions, ViewportId};

use super::icons;
use super::util;
use crate::{APP_NAME, REPO_URL, VERSION};

pub const TAGLINE: &str = "Look at a monitor. Your cursor and keyboard focus follow.";
pub const COPYRIGHT: &str = "© 2026 Buğra Şıkel and contributors · MIT License";

/// Kept in line with docs/privacy.md: only the app's own code is free of networking code; some
/// libraries it ships contain networking code it never uses.
pub const PRIVACY_STATEMENT: &str = "Everything runs on this computer. Camera frames are analysed in memory and are \
never saved, uploaded or shared. The app uses the network only for the update \
check, which is off unless you turn it on (Windows): one request to GitHub, at most \
once a day, that carries nothing about you. Apart from that, Eye Tracker's own code \
contains no network code; some libraries it ships contain networking code it does \
not use, and every release build is checked for it (see the privacy \
documentation). Only numbers are stored: your settings and the calibration \
(head-pose and eye measurements with the positions of the calibration dots). The \
camera is released completely in privacy mode, which stays on until you turn it \
off, also across restarts; while paused; and, unless turned off in the settings, \
while the screen is locked.";

/// The words of [`PRIVACY_STATEMENT`] that link to the privacy documentation.
const PRIVACY_DOCS: &str = "the privacy documentation";

/// (component, licence, what it is used for, homepage)
pub const THIRD_PARTY: [(&str, &str, &str, &str); 14] = [
    (
        "egui and eframe",
        "MIT OR Apache-2.0",
        "User interface",
        "https://github.com/emilk/egui",
    ),
    (
        "tract",
        "MIT OR Apache-2.0",
        "Runs the face models",
        "https://github.com/sonos/tract",
    ),
    (
        "MediaPipe Face Landmarker model",
        "Apache-2.0",
        "Face landmarks, head pose and iris position: the face mesh model and its \
         face geometry, run by tract (the MediaPipe runtime is not included)",
        "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
    ),
    (
        "YuNet face detector model (OpenCV Zoo)",
        "MIT",
        "Face detection",
        "https://github.com/opencv/opencv_zoo",
    ),
    (
        "nokhwa",
        "Apache-2.0",
        "Camera capture",
        "https://github.com/l1npengtul/nokhwa",
    ),
    (
        "image",
        "MIT OR Apache-2.0",
        "Image processing",
        "https://github.com/image-rs/image",
    ),
    (
        "imageproc",
        "MIT",
        "Image processing",
        "https://github.com/image-rs/imageproc",
    ),
    (
        "tray-icon (Windows, macOS)",
        "MIT OR Apache-2.0",
        "Tray icon and menu",
        "https://github.com/tauri-apps/tray-icon",
    ),
    (
        "ksni (Linux)",
        "Unlicense",
        "Tray icon and menu",
        "https://github.com/iovxw/ksni",
    ),
    (
        "notify-rust",
        "MIT OR Apache-2.0",
        "Notifications",
        "https://github.com/hoodie/notify-rust",
    ),
    (
        "windows (Windows)",
        "MIT OR Apache-2.0",
        "Windows system integration",
        "https://github.com/microsoft/windows-rs",
    ),
    (
        "objc2 (macOS)",
        "MIT",
        "macOS system integration",
        "https://github.com/madsmtm/objc2",
    ),
    (
        "x11rb (Linux)",
        "MIT OR Apache-2.0",
        "X11 window focus and hotkeys",
        "https://github.com/psychon/x11rb",
    ),
    (
        "zbus (Linux)",
        "MIT",
        "D-Bus system integration",
        "https://github.com/dbus2/zbus",
    ),
];

/// The paragraph under the third-party table.
pub const NOTICES_NOTE: &str = "Each component is used under its own licence; the licence texts ship with \
the packages. The trained models are redistributed unmodified (their origin and checksums are \
listed in the models' NOTICE file).";

/// One line with the versions that matter for bug reports.
pub fn system_info() -> String {
    let system = match std::env::consts::OS {
        "macos" => "Darwin",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    };
    let release = if cfg!(windows) {
        sysinfo::System::os_version()
            .and_then(|version| version.split_whitespace().next().map(str::to_string))
    } else {
        sysinfo::System::kernel_version()
    }
    .unwrap_or_default();
    [
        format!("{APP_NAME} {VERSION}"),
        "Rust".to_string(),
        "egui".to_string(),
        format!("{system} {release} ({})", std::env::consts::ARCH),
    ]
    .join(" · ")
}

/// The privacy statement split around the words that link to the privacy documentation:
/// (text before, link text, text after, link target).
pub fn privacy_parts() -> (&'static str, &'static str, &'static str, String) {
    let url = format!("{REPO_URL}/blob/main/docs/privacy.md");
    match PRIVACY_STATEMENT.split_once(PRIVACY_DOCS) {
        Some((before, after)) => (before, PRIVACY_DOCS, after, url),
        None => (PRIVACY_STATEMENT, "", "", url),
    }
}

/// The third-party table and its note as plain text (one row per line).
pub fn notices_text() -> String {
    let mut text = String::from("Component\tLicence\tUsed for\n");
    for (name, licence, used_for, _url) in THIRD_PARTY {
        text.push_str(&format!("{name}\t{licence}\t{used_for}\n"));
    }
    text.push_str(NOTICES_NOTE);
    text
}

/// "About Eye Tracker" window (no controller needed).
#[derive(Default)]
pub struct AboutDialog {
    open: bool,
    copied: bool,
    icon: Option<TextureHandle>,
}

impl AboutDialog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&mut self) {
        self.open = true;
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn is_open(&self) -> bool {
        self.open
    }

    /// The label of the copy button (it confirms once the text was copied).
    pub fn copy_button_text(&self) -> &'static str {
        if self.copied {
            "Copied ✓"
        } else {
            "Copy system info"
        }
    }

    /// Put [`system_info`] on the clipboard.
    pub fn copy_system_info(&mut self, ctx: &Context) {
        ctx.copy_text(system_info());
        self.copied = true;
    }

    /// The dialog's window.
    pub fn viewport_id() -> ViewportId {
        ViewportId::from_hash_of("about")
    }

    /// The window: titled, opaque, with the app icon.
    pub fn viewport_builder() -> egui::ViewportBuilder {
        let builder = egui::ViewportBuilder::default()
            .with_title(format!("About {APP_NAME}"))
            .with_inner_size([560.0, 600.0])
            // Under glow a child viewport left unset inherits the transparent root's flag.
            .with_transparent(false);
        match icons::app_icon().best(256) {
            Some(image) => builder.with_icon(image.to_icon_data()),
            None => builder,
        }
    }

    /// Draw the window while it is open.
    pub fn show(&mut self, ctx: &Context) {
        if !self.open {
            return;
        }
        if self.icon.is_none() {
            self.icon = icons::app_icon().best(96).map(|image| {
                ctx.load_texture("about-icon", image.to_color_image(), TextureOptions::LINEAR)
            });
        }
        let icon = self.icon.clone();
        let mut copy = false;
        let mut close = false;
        let copy_text = self.copy_button_text();
        ctx.show_viewport_immediate(
            Self::viewport_id(),
            Self::viewport_builder(),
            |ui, _class| {
                close = ui.input(|i| i.viewport().close_requested());
                egui::Panel::bottom("about-buttons").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                        let button = ui
                            .button(copy_text)
                            .on_hover_text("Copy version details for a bug report");
                        copy |= button.clicked();
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| draw_contents(ui, icon.as_ref()));
                });
            },
        );
        if copy {
            self.copy_system_info(ctx);
        }
        if close {
            self.open = false;
        }
    }
}

fn draw_contents(ui: &mut egui::Ui, icon: Option<&TextureHandle>) {
    ui.horizontal_top(|ui| {
        if let Some(icon) = icon {
            ui.add(egui::Image::new((icon.id(), egui::vec2(72.0, 72.0))));
        }
        ui.vertical(|ui| {
            let size = egui::TextStyle::Body.resolve(ui.style()).size * 1.9;
            ui.label(egui::RichText::new(APP_NAME).size(size).strong());
            ui.add(
                egui::Label::new(
                    egui::RichText::new(format!("Version {VERSION}"))
                        .color(util::muted_color(ui, 0.62)),
                )
                .selectable(true),
            );
            ui.label(TAGLINE);
            ui.horizontal(|ui| {
                ui.hyperlink_to("GitHub", REPO_URL);
                ui.label("·");
                ui.hyperlink_to("Report an issue", format!("{REPO_URL}/issues"));
                ui.label("·");
                ui.hyperlink_to("MIT License", format!("{REPO_URL}/blob/main/LICENSE"));
            });
        });
    });
    ui.add_space(14.0);

    ui.label(egui::RichText::new("Privacy").strong());
    let (before, link, after, url) = privacy_parts();
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label(before);
        ui.hyperlink_to(link, url);
        ui.label(after);
    });
    ui.add_space(14.0);

    ui.label(egui::RichText::new("Third-party software").strong());
    egui::Grid::new("about-third-party")
        .num_columns(3)
        .spacing([12.0, 6.0])
        .striped(true)
        .show(ui, |ui| {
            for heading in ["Component", "Licence", "Used for"] {
                ui.label(egui::RichText::new(heading).strong());
            }
            ui.end_row();
            for (name, licence, used_for, url) in THIRD_PARTY {
                ui.hyperlink_to(name, url);
                ui.label(licence);
                ui.label(used_for);
                ui.end_row();
            }
        });
    ui.label(NOTICES_NOTE);
    ui.add_space(14.0);

    let muted = util::muted_color(ui, 0.62);
    ui.add(
        egui::Label::new(
            egui::RichText::new(format!("{COPYRIGHT}\n{}", system_info())).color(muted),
        )
        .selectable(true),
    );
}

#[cfg(test)]
#[path = "about_tests.rs"]
mod tests;
