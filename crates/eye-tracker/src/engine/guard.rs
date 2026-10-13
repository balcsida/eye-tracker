//! Shoulder-surfer guard: react when a second face keeps looking at the screen.
//!
//! A second face must be visible for `delay_s` before the guard triggers, so
//! people walking past are ignored. A shoulder surfer is far away and partly
//! hidden behind the user, so the detector misses their face in some frames:
//! dropouts shorter than `gap_s` (or than a couple of frame intervals at slow
//! frame rates) do not restart the delay. Once triggered, fewer than two faces
//! must be seen continuously for `clear_s` before the guard clears, so a face
//! that briefly drops out of detection does not flicker the privacy curtain.
//!
//! **The guard never decides whether the user is present.** It cannot recognise
//! faces; it can only compare face boxes, and any rule that turned "this face is
//! not the user's" into "nobody is at the computer" would sooner or later lock
//! out a user who merely moved in their chair while a colleague was in view.
//! Walk-away detection therefore counts every detected face as the user. The
//! guard only decides whether its privacy reaction (curtain, notification, lock)
//! is still needed.
//!
//! Whose face is whose is judged by continuity, not by size:
//!
//! * While the guard is idle and exactly one face is in view (and no second face
//!   was seen moments ago), that face is the user's (the *owner*); its box is
//!   remembered and followed.
//! * While two or more faces are in view only the primary (largest) face box is
//!   known. It updates the owner's box only when it overlaps it, so a colleague
//!   who leans in closer than the user never becomes the owner. Only when no
//!   owner is known yet (the guard was switched on with two faces in view) is
//!   the largest face taken as the owner.
//! * Once triggered, a lone face that does not match the owner (much smaller and
//!   not overlapping) means the user left and the onlooker stayed. The guard then
//!   stays active ([`ShoulderGuard::owner_missing`]), also through frames
//!   without any face, so an onlooker the detector misses for a moment does not
//!   get the curtain lifted in front of them.
//! * It clears once the owner is seen alone again, or once someone uses the
//!   keyboard or mouse while the owner is judged missing (`last_input`):
//!   whoever works at the computer with a single face in view is the user from
//!   then on. That also heals a wrong judgement - the guard never stays stuck
//!   on a user sitting at their own desk.
//! * While the owner is judged missing, two or more faces in view for
//!   `delay_s` trigger the guard again. The judgement may be wrong (the user
//!   moved to another spot while a larger face hid their box), and then a new
//!   onlooker looks over the shoulder of a user reading without touching
//!   anything: the reaction (curtain, notification, lock) must not stay off
//!   until the next keystroke.

use crate::config::PrivacySettings;

/// Normalised face box `(x, y, w, h)` as in `Observation::face_box`.
pub type FaceBox = (f64, f64, f64, f64);

/// At slow frame rates a single missed frame already spans `2 x interval`;
/// a run survives gaps of up to this many frame intervals.
pub const GAP_FRAMES: f64 = 2.5;

/// ... but never gaps longer than this, however slowly frames arrive.
pub const MAX_GAP_S: f64 = 3.0;

/// The remaining face is the user's when its area is at least this fraction of
/// the user's last known face area (an onlooker behind the user is much smaller)...
pub const OWNER_MIN_AREA_RATIO: f64 = 0.5;

/// ...or when it overlaps the user's last known face box at least this much (IoU).
/// In frames with several faces, overlap alone decides whether the primary face
/// is still the user's: the area test would accept a colleague leaning in.
pub const OWNER_MIN_IOU: f64 = 0.3;

/// Keyboard or mouse input counts as "someone works at the computer" only when
/// it happened at least this long after the owner was judged missing. The user
/// who gets up and leaves often touches the mouse once more after their face has
/// left the picture; that must not make the onlooker the owner.
pub const INPUT_ADOPT_DELAY_S: f64 = 2.0;

/// What [`ShoulderGuard::update`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardEvent {
    Trigger,
    Clear,
}

/// Tuning of [`ShoulderGuard`] (seconds).
#[derive(Debug, Clone, PartialEq)]
pub struct GuardConfig {
    pub enabled: bool,
    /// A second face must be visible this long before the guard triggers.
    pub delay_s: f64,
    /// Once triggered, fewer than two faces must be seen this long to clear.
    pub clear_s: f64,
    /// Missed detections of the second face shorter than this do not restart `delay_s`.
    pub gap_s: f64,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            delay_s: 2.0,
            clear_s: 1.5,
            gap_s: 1.0,
        }
    }
}

impl GuardConfig {
    /// Build from user settings (`s.guard_action` is handled by the controller).
    pub fn from_settings(s: &PrivacySettings) -> Self {
        Self {
            enabled: s.shoulder_guard,
            delay_s: s.guard_delay_s,
            ..Self::default()
        }
    }
}

/// Debounced detector for "more than one face in front of the camera".
#[derive(Debug)]
pub struct ShoulderGuard {
    config: GuardConfig,
    active: bool,
    multi_since: Option<f64>,
    multi_last: f64,
    single_since: Option<f64>,
    last_update: Option<f64>,
    owner_box: Option<FaceBox>,
    /// The user left while the guard was active (a lone stranger's face was
    /// seen) and has not been seen since; set at `owner_left_at`.
    owner_left: bool,
    owner_left_at: f64,
}

impl Default for ShoulderGuard {
    fn default() -> Self {
        Self::new(GuardConfig::default())
    }
}

impl ShoulderGuard {
    pub fn new(config: GuardConfig) -> Self {
        Self {
            config,
            active: false,
            multi_since: None,
            multi_last: 0.0,
            single_since: None,
            last_update: None,
            owner_box: None,
            owner_left: false,
            owner_left_at: f64::NEG_INFINITY,
        }
    }

    /// `true` between a `Trigger` and the following `Clear`.
    pub fn active(&self) -> bool {
        self.active
    }

    /// The guard is active and the user has not been seen since only another
    /// person's face was left in view.
    ///
    /// The user has most likely walked away while the onlooker stayed, so the
    /// guard stays active (the curtain stays up) until the user is back. This is
    /// a judgement from face boxes, not a recognition: it must never be used to
    /// decide that the user is absent (see the module docs).
    pub fn owner_missing(&self) -> bool {
        self.owner_left
    }

    pub fn config(&self) -> &GuardConfig {
        &self.config
    }

    /// Change the tuning. Disabling an active guard clears it on the next update.
    pub fn set_config(&mut self, config: GuardConfig) {
        self.config = config;
    }

    /// Forget everything (inactive, no pending run, no owner). Emits nothing.
    pub fn reset(&mut self) {
        self.active = false;
        self.multi_since = None;
        self.single_since = None;
        self.last_update = None;
        self.owner_box = None;
        self.owner_left = false;
        self.owner_left_at = f64::NEG_INFINITY;
    }

    /// Feed one observation; return `Trigger`, `Clear` or `None`.
    ///
    /// `face_count` is `None` when the camera cannot tell (camera off): the
    /// guard keeps its state, but an interrupted observation is not
    /// "continuous", so any run in progress restarts. `face_box` is the
    /// primary (largest) face of the observation (`Observation::face_box`);
    /// without it the guard cannot tell whose face remains and clears as soon
    /// as fewer than two faces are seen for `clear_s`. `last_input` is the
    /// time of the latest keyboard or mouse input (same clock as `now`;
    /// `-inf` if unknown): input while the owner is judged missing means
    /// someone works at the computer, see the module docs.
    pub fn update(
        &mut self,
        now: f64,
        face_count: Option<u32>,
        face_box: Option<FaceBox>,
        last_input: f64,
    ) -> Option<GuardEvent> {
        let cfg = self.config.clone();
        if !cfg.enabled {
            let was_active = self.active;
            self.reset();
            if was_active {
                log::info!("Shoulder guard disabled while active; clearing");
                return Some(GuardEvent::Clear);
            }
            return None;
        }

        let Some(face_count) = face_count else {
            // Nothing is known about this moment: runs restart, but whether the
            // owner left is kept (a camera hiccup does not bring them back).
            self.multi_since = None;
            self.single_since = None;
            self.last_update = None;
            return None;
        };

        let interval = self.last_update.map_or(0.0, |t| (now - t).max(0.0));
        self.last_update = Some(now);
        let tolerance = cfg.gap_s.max((GAP_FRAMES * interval).min(MAX_GAP_S));

        if face_count >= 2 {
            self.single_since = None;
            self.follow_owner_among_faces(face_box);
            if self.active && !self.owner_left {
                return None; // already reacted to these faces
            }
            let since = match self.multi_since {
                Some(s) if now - self.multi_last <= tolerance => s,
                _ => now,
            };
            self.multi_since = Some(since);
            self.multi_last = now;
            if now - since >= cfg.delay_s {
                self.multi_since = None;
                if self.active {
                    // Several faces while the user was judged gone: whoever they
                    // are, someone may be looking over a shoulder again (see the
                    // module docs). React once more; the guard stays active.
                    self.owner_left = false;
                    log::info!("Shoulder guard triggered again ({face_count} faces)");
                    return Some(GuardEvent::Trigger);
                }
                self.active = true;
                log::info!("Shoulder guard triggered ({face_count} faces)");
                return Some(GuardEvent::Trigger);
            }
            return None;
        }

        if self.multi_since.is_some() && now - self.multi_last > tolerance {
            self.multi_since = None;
        }
        if !self.active {
            if face_count == 1
                && self.multi_since.is_none()
                && let Some(face_box) = face_box
            {
                // Alone at the desk: the user. Not while a second face was seen
                // moments ago - the lone face of such a dropout may be either one.
                self.owner_box = Some(face_box);
            }
            return None;
        }

        let someone_working =
            self.owner_left && last_input >= self.owner_left_at + INPUT_ADOPT_DELAY_S;
        if let (1, Some(face_box), Some(_)) = (face_count, face_box, self.owner_box) {
            if someone_working || self.is_owner(face_box) {
                if self.owner_left {
                    log::info!(
                        "Shoulder guard: the user is back{}",
                        if someone_working {
                            " (keyboard or mouse input)"
                        } else {
                            ""
                        }
                    );
                }
                self.owner_left = false;
                self.owner_box = Some(face_box); // follow the user's movements
            } else {
                // The user left and the onlooker stayed: keep the curtain up.
                if !self.owner_left {
                    log::info!("Shoulder guard: only the onlooker's face is left");
                    self.owner_left = true;
                    self.owner_left_at = now;
                }
                self.single_since = None;
                return None;
            }
        } else if self.owner_left {
            // No face (or no box) after the user left: most likely the onlooker
            // was missed for a moment, so the curtain stays up - unless someone
            // is using the computer.
            if !someone_working {
                self.single_since = None;
                return None;
            }
            log::info!("Shoulder guard: keyboard or mouse input while no face is in view");
            self.owner_left = false;
        }
        let single_since = *self.single_since.get_or_insert(now);
        if now - single_since >= cfg.clear_s {
            self.active = false;
            self.single_since = None;
            log::info!("Shoulder guard cleared");
            return Some(GuardEvent::Clear);
        }
        None
    }

    /// Update the owner from the primary face of a frame with several faces.
    ///
    /// The primary face is the largest one, which is the user's only if nobody
    /// comes closer to the camera than they are. So it is taken as the user's
    /// only where it overlaps their last known box (continuity); the largest
    /// face is merely the fallback while no owner is known at all.
    fn follow_owner_among_faces(&mut self, face_box: Option<FaceBox>) {
        let Some(face_box) = face_box else {
            return;
        };
        if self
            .owner_box
            .is_none_or(|owner| iou(face_box, owner) >= OWNER_MIN_IOU)
        {
            self.owner_box = Some(face_box);
            if self.owner_left {
                log::info!("Shoulder guard: the user is back next to the onlooker");
            }
            self.owner_left = false;
        }
    }

    fn is_owner(&self, face_box: FaceBox) -> bool {
        let Some(owner) = self.owner_box else {
            return true;
        };
        let owner_area = area(owner);
        if owner_area <= 0.0 {
            return true;
        }
        area(face_box) >= OWNER_MIN_AREA_RATIO * owner_area || iou(face_box, owner) >= OWNER_MIN_IOU
    }
}

fn area(b: FaceBox) -> f64 {
    b.2.max(0.0) * b.3.max(0.0)
}

/// Intersection over union of two `(x, y, w, h)` boxes.
fn iou(a: FaceBox, b: FaceBox) -> f64 {
    let ix = ((a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0)).max(0.0);
    let iy = ((a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1)).max(0.0);
    let inter = ix * iy;
    let union = area(a) + area(b) - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

#[cfg(test)]
#[path = "guard_tests.rs"]
mod tests;
