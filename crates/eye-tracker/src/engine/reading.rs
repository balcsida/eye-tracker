//! Recognise what the user *reads* while typing somewhere else.
//!
//! Copying from a document into an editor means looking at the source while the
//! keystrokes go elsewhere. A decider that moves the keyboard focus to where the
//! user looks must not take those looks for a wish to work there.
//! [`ReadingTracker`] finds such a *reading target*: another target (a
//! monitor, a split pane) the user kept typing while, or right after, looking at.
//! Evidence is a look at the target (two or more consecutive updates with the gaze
//! clearly on it) followed by typing that continues at least
//! [`READING_EVIDENCE_S`] after the look began and resumes no later than
//! [`READING_RETURN_S`] after it ended. A single noisy gaze sample never counts,
//! and neither does the usual "last keystroke while the eyes already move on" of a
//! user who is about to work on the other target.
//!
//! The reading target is forgotten when typing has paused for the reading grace
//! or when the gaze has not been on it for [`READING_FORGET_S`]. The owner
//! forgets it ([`ReadingTracker::clear`]) when the keyboard focus moves, since
//! what the user read beside the old focus says nothing about the new one.
//!
//! Targets are identified by any comparable key; the tracker is pure logic driven by
//! injected times, like the deciders that use it.

use std::fmt::Debug;

/// Typing must go on at least this long after a look at another target began
/// for that target to become a reading target. Shorter overlaps are a user
/// finishing a word while the eyes already move to where they want to work.
pub const READING_EVIDENCE_S: f64 = 1.0;

/// Typing that resumes within this time after a look at another target ended
/// also makes it a reading target: the user glanced over, did not want to
/// switch and carried on typing where they were.
pub const READING_RETURN_S: f64 = 2.0;

/// A reading target the gaze has not rested on for this long is forgotten.
pub const READING_FORGET_S: f64 = 60.0;

/// Updates further apart than this end a look. Longer than a decider's dwell gap
/// on purpose: while the user types the camera runs at 1-2 fps, and reading must
/// be recognised at exactly that rate.
pub const READING_MAX_GAP_S: f64 = 1.5;

/// Absorbs float rounding in time differences.
const EPS: f64 = 1e-6;

/// A run of consecutive updates with the gaze clearly on one other target.
#[derive(Debug, Clone)]
struct Look<K> {
    key: K,
    start: f64,
    last: f64,
}

impl<K> Look<K> {
    /// Seen on more than one update (a single noisy sample is no look).
    fn repeated(&self) -> bool {
        self.last > self.start
    }
}

/// The reading target beside the current keyboard focus (see the module docs).
///
/// Per update the owner reports a clear look at another target with
/// [`observe`](Self::observe) or the absence of one with [`end_look`](Self::end_look), then
/// calls [`update`](Self::update). A long gap between updates ends the look
/// ([`end_look`](Self::end_look)); a clock going backwards or a focus change clears everything
/// ([`clear`](Self::clear)).
#[derive(Debug, Clone)]
pub struct ReadingTracker<K> {
    look: Option<Look<K>>,
    prev_look: Option<Look<K>>,
    reading: Option<K>,
    reading_seen: f64,
}

impl<K> Default for ReadingTracker<K> {
    fn default() -> Self {
        Self {
            look: None,
            prev_look: None,
            reading: None,
            reading_seen: f64::NEG_INFINITY,
        }
    }
}

impl<K: Clone + PartialEq + Debug> ReadingTracker<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The current reading target, if any.
    pub fn reading(&self) -> Option<&K> {
        self.reading.as_ref()
    }

    /// Forget the looks and the reading target.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// The gaze is clearly on `key` (not the focused target) at `now`.
    pub fn observe(&mut self, key: K, now: f64) {
        match &mut self.look {
            Some(look) if look.key == key => look.last = now,
            _ => {
                self.end_look();
                self.look = Some(Look {
                    key: key.clone(),
                    start: now,
                    last: now,
                });
            }
        }
        if self.reading.as_ref() == Some(&key) {
            self.reading_seen = now;
        }
    }

    /// The gaze is not clearly on another target (or the updates had a gap).
    pub fn end_look(&mut self) {
        // A single-sample "look" is kept out so that one noisy gaze sample cannot
        // replace the evidence of a real look just before it.
        if let Some(look) = self.look.take()
            && look.repeated()
        {
            self.prev_look = Some(look);
        }
    }

    /// Recognise and expire the reading target.
    ///
    /// `last_key` is the time of the last keystroke (`-inf` for never).
    /// A `reading_grace_s` not above `typing_grace_s` turns the rule off.
    /// `active` is false while there is no focused target to read beside, so
    /// no new reading target is recognised; `context` (the focused target)
    /// only appears in the debug log.
    pub fn update(
        &mut self,
        now: f64,
        last_key: f64,
        typing_grace_s: f64,
        reading_grace_s: f64,
        context: &dyn Debug,
        active: bool,
    ) {
        if reading_grace_s <= typing_grace_s {
            self.reading = None;
            return;
        }
        let typing_recent = now - last_key < reading_grace_s - EPS;
        if typing_recent && active {
            for look in [&self.look, &self.prev_look].into_iter().flatten() {
                if look.repeated()
                    && look.start + READING_EVIDENCE_S <= last_key + EPS
                    && last_key <= look.last + READING_RETURN_S + EPS
                {
                    if self.reading.as_ref() != Some(&look.key) {
                        log::debug!("Typing at {context:?} while reading {:?}", look.key);
                        self.reading = Some(look.key.clone());
                    }
                    self.reading_seen = self.reading_seen.max(look.last);
                    break;
                }
            }
        }
        if self.reading.is_some() && (!typing_recent || now - self.reading_seen > READING_FORGET_S)
        {
            self.reading = None;
        }
    }
}

#[cfg(test)]
#[path = "reading_tests.rs"]
mod tests;
