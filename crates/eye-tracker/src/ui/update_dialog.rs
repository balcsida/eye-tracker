//! The update window: checking, what is new, downloading, installing.
//!
//! It only shows the state of the [`UpdateService`](crate::update::service::UpdateService)
//! and sends the user's choices back to it as [`UpdateAction`]s; closing it never stops a
//! running check (a download has its own Cancel button).

use super::icons;
use crate::update::release::releases_page;
use crate::update::service::{Phase, UpdateState};
use crate::{APP_NAME, VERSION};

/// What the window asks the update service to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateAction {
    /// `UpdateService::check_now`.
    CheckNow,
    /// `UpdateService::install`.
    Install,
    /// `UpdateService::cancel`.
    Cancel,
}

/// What a button does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Close,
    Install,
    Cancel,
    CheckNow,
    /// Open the release page (or the releases page) in the browser.
    OpenDownloadPage,
}

/// The progress bar.
#[derive(Debug, Clone, PartialEq)]
pub enum Bar {
    Hidden,
    /// Busy indicator (no known end).
    Busy,
    Progress {
        done: u64,
        total: u64,
        text: String,
    },
}

/// Everything the window shows for one service state.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateView {
    pub title: String,
    /// Hidden when empty.
    pub detail: String,
    /// "What is new in X" and the release page, below the detail.
    pub link: Option<(String, String)>,
    pub bar: Bar,
    /// The default button; `None`: hidden.
    pub primary: Option<(&'static str, Choice)>,
    pub secondary: Option<(&'static str, Choice)>,
}

/// What the window shows for `state`; `can_install` is `UpdateService::can_install`.
pub fn update_view(state: &UpdateState, can_install: bool) -> UpdateView {
    let release = state.release.as_ref();
    let mut view = UpdateView {
        title: String::new(),
        detail: String::new(),
        link: None,
        bar: Bar::Hidden,
        primary: Some(("Close", Choice::Close)),
        secondary: None,
    };
    let mut text = |title: String, detail: String| {
        view.title = title;
        view.detail = detail;
    };
    match (state.phase, release) {
        (Phase::Checking, _) => {
            text(
                "Checking for updates…".into(),
                "Asking GitHub for the latest release.".into(),
            );
            view.bar = Bar::Busy;
        }
        (Phase::UpToDate, _) => text(
            "You have the latest version".into(),
            format!("{APP_NAME} {VERSION} is up to date."),
        ),
        (Phase::Available, Some(release)) => {
            let title = format!("Version {} is available", release.version);
            if can_install {
                text(
                    title,
                    format!(
                        "You have {VERSION}. {APP_NAME} will download the new setup, check it \
                         and restart itself."
                    ),
                );
                view.primary = Some(("Install and restart", Choice::Install));
                view.secondary = Some(("Later", Choice::Close));
            } else {
                text(
                    title,
                    format!(
                        "You have {VERSION}. This copy cannot update itself; download the new \
                         version from the releases page."
                    ),
                );
                view.primary = Some(("Open the download page", Choice::OpenDownloadPage));
                view.secondary = Some(("Close", Choice::Close));
            }
            view.link = Some((
                format!("What is new in {}", release.version),
                release.page_url.clone(),
            ));
        }
        (Phase::Downloading, _) => {
            text(
                format!(
                    "Downloading {}…",
                    release.map_or("", |r| r.version.as_str())
                ),
                String::new(),
            );
            view.bar = if state.total > 0 {
                Bar::Progress {
                    done: state.done.min(state.total),
                    total: state.total,
                    text: format!(
                        "{:.1} / {:.1} MB",
                        state.done as f64 / 1e6,
                        state.total as f64 / 1e6
                    ),
                }
            } else {
                Bar::Busy
            };
            view.primary = Some(("Cancel", Choice::Cancel));
        }
        (Phase::Installing, _) => {
            text(
                "Installing…".into(),
                format!("{APP_NAME} will close and start again in a moment."),
            );
            view.bar = Bar::Busy;
            view.primary = None;
        }
        (Phase::Failed, _) => {
            text("The update did not work".into(), state.message.clone());
            let retry = release.is_some() && can_install;
            let again = if retry {
                Choice::Install
            } else {
                Choice::CheckNow
            };
            view.primary = Some(("Try again", again));
            view.secondary = Some(("Close", Choice::Close));
        }
        (Phase::Idle | Phase::Available, _) => text(String::new(), String::new()),
    }
    view
}

/// What the window reads from the update service.
pub struct UpdateInputs<'a> {
    pub state: &'a UpdateState,
    /// `UpdateService::can_install`.
    pub can_install: bool,
}

/// One window for every phase of an update (see the module docs).
#[derive(Debug, Default)]
pub struct UpdateDialog {
    visible: bool,
    /// Bring the window to the front with the next frame.
    raise: bool,
}

impl UpdateDialog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Show the window; look for an update unless one is known or being fetched.
    pub fn present(&mut self, state: &UpdateState) -> Vec<UpdateAction> {
        let known = state.phase == Phase::Failed && state.release.is_some();
        let mut actions = Vec::new();
        if matches!(state.phase, Phase::Idle | Phase::UpToDate | Phase::Failed) && !known {
            actions.push(UpdateAction::CheckNow);
        }
        self.visible = true;
        self.raise = true;
        actions
    }

    /// Hide the window (a running check or download goes on).
    pub fn close(&mut self) {
        self.visible = false;
    }

    /// A button was pressed. Opening the download page is done by [`Self::show`].
    pub fn choose(&mut self, choice: Choice) -> Option<UpdateAction> {
        match choice {
            Choice::Close => {
                self.close();
                None
            }
            Choice::Install => Some(UpdateAction::Install),
            Choice::Cancel => Some(UpdateAction::Cancel),
            Choice::CheckNow => Some(UpdateAction::CheckNow),
            Choice::OpenDownloadPage => None,
        }
    }

    /// The window: titled, opaque, with the app icon.
    pub fn viewport_builder() -> egui::ViewportBuilder {
        icons::with_app_icon(
            egui::ViewportBuilder::default()
                .with_title(format!("{APP_NAME} updates"))
                .with_inner_size([460.0, 200.0])
                .with_min_inner_size([440.0, 160.0])
                // Under glow a child viewport left unset inherits the transparent root's flag.
                .with_transparent(false),
        )
    }

    /// Draw the window while it is shown; returns what it asks the service to do.
    pub fn show(&mut self, ctx: &egui::Context, inputs: &UpdateInputs<'_>) -> Vec<UpdateAction> {
        if !self.visible {
            return Vec::new();
        }
        let view = update_view(inputs.state, inputs.can_install);
        let mut chosen = None;
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("eye-tracker-updates"),
            Self::viewport_builder(),
            |ui, _class| {
                if std::mem::take(&mut self.raise) {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                if ui.ctx().input(|i| i.viewport().close_requested()) {
                    chosen = Some(Choice::Close);
                }
                chosen = chosen.or(draw(ui, &view));
            },
        );
        let Some(choice) = chosen else {
            if matches!(view.bar, Bar::Busy | Bar::Progress { .. }) {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
            return Vec::new();
        };
        if choice == Choice::OpenDownloadPage {
            let url = inputs
                .state
                .release
                .as_ref()
                .map_or_else(releases_page, |r| r.page_url.clone());
            ctx.open_url(egui::OpenUrl::new_tab(url));
        }
        self.choose(choice).into_iter().collect()
    }
}

/// The window's content; returns the button pressed.
fn draw(ui: &mut egui::Ui, view: &UpdateView) -> Option<Choice> {
    let mut chosen = None;
    egui::CentralPanel::default().show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 12.0;
        ui.add(egui::Label::new(egui::RichText::new(&view.title).strong().size(16.0)).wrap());
        if !view.detail.is_empty() || view.link.is_some() {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                if !view.detail.is_empty() {
                    ui.add(egui::Label::new(&view.detail).wrap().selectable(true));
                }
                if let Some((text, url)) = &view.link {
                    ui.hyperlink_to(text, url);
                }
            });
        }
        match &view.bar {
            Bar::Hidden => {}
            Bar::Busy => {
                ui.add(egui::ProgressBar::new(0.0).animate(true));
            }
            Bar::Progress { done, total, text } => {
                let fraction = *done as f32 / (*total).max(1) as f32;
                ui.add(egui::ProgressBar::new(fraction).text(text.as_str()));
            }
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
            for (text, choice) in [view.primary, view.secondary].into_iter().flatten() {
                if ui.button(text).clicked() {
                    chosen = Some(choice);
                }
            }
        });
        if let Some((_, choice)) = view.primary
            && ui.ctx().input(|i| i.key_pressed(egui::Key::Enter))
        {
            chosen = Some(choice);
        }
    });
    chosen
}

#[cfg(test)]
#[path = "update_dialog_tests.rs"]
mod tests;
