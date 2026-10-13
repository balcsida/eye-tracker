//! The opt-in update check and installer (the only networking code of the app).

// TODO(port): src/eye_tracker/update/__init__.py

pub mod fetch;
pub mod installer;
pub mod release;
pub mod service;
pub mod winhttp;
