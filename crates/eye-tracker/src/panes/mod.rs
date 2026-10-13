//! Split-pane focus (experimental): keyboard focus follows the gaze between the split panes of the focused terminal window.

// TODO(port): src/eye_tracker/panes/__init__.py

pub mod decider;
pub mod providers;
pub mod registry;
pub mod types;
pub mod worker;
