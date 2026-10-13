//! Thread-pool policy for the vision pipeline.
//!
//! The Python app capped OpenCV's process-wide worker pool, which parallelises most calls
//! (colour conversion, resizing, DNN layers) over one thread per core. For the small images this
//! app handles the extra threads barely shorten a call but spin while they wait for work, which
//! multiplies the CPU time. Measured on a 16-thread desktop (OpenCV 5.0, single frames paced
//! like the worker), the face landmark network took 12.5 ms wall / 10.9 ms CPU with one thread
//! and 9.6 ms wall / 84 ms CPU with 16; YuNet detection at 320 px took 5.8 ms / 3.9 ms against
//! 3.3 ms / 24.5 ms.
//!
//! So the pool is capped at a single thread: a few milliseconds of latency buy an up to eightfold
//! CPU saving, which is what matters for an app that runs all day.
//!
//! The Rust port has no such pool: tract runs a plan on the calling thread (its default
//! executor) and the image helpers in [`super`] are sequential. The policy is therefore
//! satisfied by construction, and [`limit_opencv_threads`] only reports the effective size.

/// Upper bound for the worker pool while the vision pipeline runs.
pub const MAX_OPENCV_THREADS: usize = 1;

/// The size of the worker pool after capping it at `max_threads`; always
/// [`MAX_OPENCV_THREADS`] because the pipeline never starts a pool (see the module docs).
///
/// Never raises the limit and is cheap enough to call whenever a vision component starts.
pub fn limit_opencv_threads(max_threads: usize) -> usize {
    max_threads.clamp(1, MAX_OPENCV_THREADS)
}

#[cfg(test)]
#[path = "threads_tests.rs"]
mod tests;
