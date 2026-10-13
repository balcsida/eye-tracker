//! Decision engine: turns observations and input activity into actions.

// TODO(port): src/eye_tracker/engine/__init__.py

pub mod camera_yield;
pub mod controller;
pub mod decision;
pub mod guard;
pub mod input_state;
pub mod presence;
pub mod reading;
pub mod scheduler;
pub mod window_memory;
