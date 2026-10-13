// Ported from tests/test_vision_threads.py. Not ported, there is no OpenCV pool to fake:
// test_caps_a_large_pool (the pool calls), test_never_raises_the_limit (pool size 2),
// test_builds_without_a_pool_are_tolerated (cv2.error) and test_backends_cap_the_pool (backends
// call the function on construction; covered by the backend tests that port with them).

use super::*;

#[test]
fn test_caps_a_large_pool() {
    assert_eq!(limit_opencv_threads(MAX_OPENCV_THREADS), 1);
    assert_eq!(MAX_OPENCV_THREADS, 1);
    assert_eq!(limit_opencv_threads(16), 1);
}

#[test]
fn test_invalid_limit_means_one() {
    assert_eq!(limit_opencv_threads(0), 1);
}
