//! Tests for update/fetch.rs: what the update check may fetch, without a network.
//!
//! The fakes are shared with the installer and service tests (as test_update_fetch.py was
//! imported by the other test files); this module is `pub(crate)` for that.
//!
//! Not ported: `test_without_a_transport_other_platforms_say_so` monkeypatched
//! `fetch.supported`; it calls `transport_for(false)` instead, which is what
//! `default_transport` runs. Nothing else is left out.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::*;

/// What a fake answer remembers; shared so a test can look at it after the fetcher used it.
#[derive(Clone)]
pub(crate) struct ResponseState {
    pub status: u32,
    pub body: Vec<u8>,
    pub location: Option<String>,
    /// `None`: no length announced. `Some(None)`-style "the real length" is `announced_real`.
    pub announced: Option<u64>,
    /// The announced length is the real length of the body.
    pub announced_real: bool,
    pub closed: bool,
    pub pos: usize,
    /// Pause before each read (a body that arrives slowly).
    pub read_delay: std::time::Duration,
}

#[derive(Clone)]
pub(crate) struct FakeResponse(pub Arc<Mutex<ResponseState>>);

impl FakeResponse {
    pub fn new(status: u32, body: &[u8]) -> Self {
        Self(Arc::new(Mutex::new(ResponseState {
            status,
            body: body.to_vec(),
            location: None,
            announced: None,
            announced_real: true,
            closed: false,
            pos: 0,
            read_delay: std::time::Duration::ZERO,
        })))
    }

    pub fn ok(body: &[u8]) -> Self {
        Self::new(200, body)
    }

    pub fn status(status: u32) -> Self {
        Self::new(status, b"")
    }

    /// Announce `length` (`None`: no length at all) instead of the real one.
    pub fn announcing(self, length: Option<u64>) -> Self {
        {
            let mut state = self.0.lock();
            state.announced = length;
            state.announced_real = false;
        }
        self
    }

    pub fn slow(self) -> Self {
        self.0.lock().read_delay = std::time::Duration::from_millis(10);
        self
    }

    pub fn closed(&self) -> bool {
        self.0.lock().closed
    }

    pub fn pos(&self) -> usize {
        self.0.lock().pos
    }

    /// An unread copy of this answer.
    fn fresh(&self) -> Self {
        let mut state = self.0.lock().clone();
        state.pos = 0;
        state.closed = false;
        Self(Arc::new(Mutex::new(state)))
    }
}

pub(crate) fn redirect(location: &str, status: u32) -> FakeResponse {
    let response = FakeResponse::status(status);
    response.0.lock().location = Some(location.to_string());
    response
}

impl Response for FakeResponse {
    fn status(&self) -> u32 {
        self.0.lock().status
    }

    fn location(&self) -> Option<String> {
        self.0.lock().location.clone()
    }

    fn content_length(&self) -> Option<u64> {
        let state = self.0.lock();
        if state.announced_real {
            Some(state.body.len() as u64)
        } else {
            state.announced
        }
    }

    fn read(&mut self, size: usize) -> Result<Vec<u8>, NetworkError> {
        let delay = self.0.lock().read_delay;
        std::thread::sleep(delay);
        let mut state = self.0.lock();
        let end = (state.pos + size).min(state.body.len());
        let chunk = state.body[state.pos..end].to_vec();
        state.pos += chunk.len();
        Ok(chunk)
    }

    fn close(&mut self) {
        self.0.lock().closed = true;
    }
}

#[derive(Default)]
struct TransportState {
    answers: HashMap<(String, String), FakeResponse>,
    asked: Vec<(String, String, Option<String>)>,
    closed: bool,
    fresh: bool,
}

/// Answers by (host, path); records what it was asked. Clones share their state. With `fresh`,
/// every request gets an unread copy of the answer (a check can run twice).
#[derive(Clone, Default)]
pub(crate) struct FakeTransport(Arc<Mutex<TransportState>>);

impl FakeTransport {
    pub fn new(answers: Vec<((&str, &str), FakeResponse)>) -> Self {
        let transport = Self::default();
        for (key, response) in answers {
            transport.insert(key.0, key.1, response);
        }
        transport
    }

    pub fn fresh(self) -> Self {
        self.0.lock().fresh = true;
        self
    }

    pub fn insert(&self, host: &str, path: &str, response: FakeResponse) {
        self.0
            .lock()
            .answers
            .insert((host.to_string(), path.to_string()), response);
    }

    pub fn asked(&self) -> Vec<(String, String, Option<String>)> {
        self.0.lock().asked.clone()
    }

    pub fn closed(&self) -> bool {
        self.0.lock().closed
    }
}

impl Transport for FakeTransport {
    fn open(
        &mut self,
        host: &str,
        path: &str,
        accept: Option<&str>,
    ) -> Result<Box<dyn Response>, NetworkError> {
        let mut state = self.0.lock();
        state.asked.push((
            host.to_string(),
            path.to_string(),
            accept.map(str::to_string),
        ));
        let template = state
            .answers
            .get(&(host.to_string(), path.to_string()))
            .expect("an answer for the request")
            .clone();
        Ok(Box::new(if state.fresh {
            template.fresh()
        } else {
            template
        }))
    }

    fn close(&mut self) {
        self.0.lock().closed = true;
    }
}

fn fetcher(transport: &FakeTransport) -> Fetcher {
    Fetcher::with_transport(Box::new(transport.clone()))
}

fn error_of<T>(result: Result<T, impl std::fmt::Display>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

// ------------------------------------------------------------------------- addresses
#[test]
fn test_split_url_accepts_github_hosts() {
    let cases = [
        (
            "https://api.github.com/repos/a/b/releases/latest",
            ("api.github.com", "/repos/a/b/releases/latest"),
        ),
        ("https://GitHub.com/a/b?x=1", ("github.com", "/a/b?x=1")),
        ("https://github.com", ("github.com", "/")),
        (
            "https://release-assets.githubusercontent.com/x/y?sig=abc%3D",
            ("release-assets.githubusercontent.com", "/x/y?sig=abc%3D"),
        ),
    ];
    for (url, (host, path)) in cases {
        assert_eq!(
            split_url(url).unwrap(),
            (host.to_string(), path.to_string())
        );
    }
}

#[test]
fn test_split_url_refuses_everything_else() {
    let urls = [
        "http://github.com/a", // not https
        "https://evil.example/a",
        "https://github.com.evil.example/a",
        "https://evil.example/github.com",
        "https://user:pw@github.com/a", // credentials
        "https://github.com:8443/a",    // port
        "https://github.com@evil.example/a",
        "//github.com/a",
        "ftp://github.com/a",
        "https://github.com/a b",
        "https://github.com/a#frag",
        "",
        "github.com/a",
    ];
    for url in urls {
        assert!(split_url(url).is_err(), "{url:?}");
    }
}

#[test]
fn test_only_github_hosts_are_allowed() {
    let mut hosts = ALLOWED_HOSTS.to_vec();
    hosts.sort_unstable();
    assert_eq!(
        hosts,
        [
            "api.github.com",
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        ]
    );
}

#[test]
fn test_a_location_path_stays_on_the_host() {
    assert_eq!(
        resolve_location("github.com", "/a/b"),
        "https://github.com/a/b"
    );
    assert_eq!(
        resolve_location("github.com", "https://x.test/a"),
        "https://x.test/a"
    );
    // A protocol-relative address is not a path of this host.
    assert_eq!(
        resolve_location("github.com", "//evil.example/a"),
        "//evil.example/a"
    );
}

// ------------------------------------------------------------------------------- get
#[test]
fn test_get_returns_the_body_and_sends_only_the_accept_header() {
    let transport =
        FakeTransport::new(vec![(("api.github.com", "/x"), FakeResponse::ok(b"hello"))]);
    {
        let mut fetcher = fetcher(&transport);
        assert_eq!(
            fetcher
                .get("https://api.github.com/x", 100, Some("a/b"))
                .unwrap(),
            b"hello"
        );
    }
    assert_eq!(
        transport.asked(),
        [("api.github.com".into(), "/x".into(), Some("a/b".into()))]
    );
    assert!(!transport.closed()); // a transport that was passed in is not ours to close
}

#[test]
fn test_get_follows_redirects_through_allowed_hosts() {
    let last = FakeResponse::ok(b"file");
    let transport = FakeTransport::new(vec![
        (
            ("github.com", "/a"),
            redirect("https://release-assets.githubusercontent.com/b?s=1", 302),
        ),
        (
            ("release-assets.githubusercontent.com", "/b?s=1"),
            last.clone(),
        ),
    ]);
    assert_eq!(
        fetcher(&transport)
            .get("https://github.com/a", 10, None)
            .unwrap(),
        b"file"
    );
    assert!(last.closed());
}

#[test]
fn test_a_relative_redirect_stays_on_the_host() {
    let transport = FakeTransport::new(vec![
        (("github.com", "/a"), redirect("/b", 302)),
        (("github.com", "/b"), FakeResponse::ok(b"x")),
    ]);
    assert_eq!(
        fetcher(&transport)
            .get("https://github.com/a", 10, None)
            .unwrap(),
        b"x"
    );
}

#[test]
fn test_a_redirect_to_another_host_is_refused_without_asking_it() {
    let first = redirect("https://evil.example/file", 302);
    let transport = FakeTransport::new(vec![(("github.com", "/a"), first.clone())]);
    let message = error_of(fetcher(&transport).get("https://github.com/a", 10, None));
    assert!(message.contains("not a GitHub host"), "{message}");
    let hosts: Vec<_> = transport
        .asked()
        .into_iter()
        .map(|(host, _, _)| host)
        .collect();
    assert_eq!(hosts, ["github.com"]);
    assert!(first.closed());
}

#[test]
fn test_a_redirect_that_leaves_https_is_refused() {
    for location in ["http://github.com/b", "ftp://github.com/b", "//evil/b"] {
        let transport = FakeTransport::new(vec![(("github.com", "/a"), redirect(location, 302))]);
        assert!(
            fetcher(&transport)
                .get("https://github.com/a", 10, None)
                .is_err(),
            "{location}"
        );
    }
}

#[test]
fn test_redirect_loops_end() {
    let transport = FakeTransport::new(vec![(
        ("github.com", "/a"),
        redirect("https://github.com/a", 302),
    )])
    .fresh();
    let message = error_of(fetcher(&transport).get("https://github.com/a", 10, None));
    assert!(message.contains("Too many redirects"), "{message}");
    assert_eq!(transport.asked().len(), MAX_REDIRECTS + 1);
}

#[test]
fn test_other_statuses_are_errors() {
    for status in [403, 404, 500, 301, 304] {
        let transport =
            FakeTransport::new(vec![(("github.com", "/a"), FakeResponse::status(status))]);
        let message = error_of(fetcher(&transport).get("https://github.com/a", 10, None));
        assert!(message.contains(&format!("HTTP {status}")), "{message}");
    }
}

#[test]
fn test_an_announced_length_over_the_limit_is_refused_before_reading() {
    let response = FakeResponse::ok(b"x").announcing(Some(11));
    let transport = FakeTransport::new(vec![(("github.com", "/a"), response.clone())]);
    let message = error_of(fetcher(&transport).get("https://github.com/a", 10, None));
    assert!(message.contains("larger"), "{message}");
    assert!(response.closed());
    assert_eq!(response.pos(), 0);
}

#[test]
fn test_a_body_over_the_limit_is_refused_even_if_it_lied_about_its_length() {
    let response = FakeResponse::ok(&[b'x'; 50]).announcing(None);
    let transport = FakeTransport::new(vec![(("github.com", "/a"), response.clone())]);
    let message = error_of(fetcher(&transport).get("https://github.com/a", 10, None));
    assert!(message.contains("larger"), "{message}");
    assert!(response.closed());
}

#[test]
fn test_get_reads_a_body_in_several_chunks() {
    let transport = FakeTransport::new(vec![(
        ("github.com", "/a"),
        FakeResponse::ok(b"0123456789"),
    )]);
    let mut fetcher = fetcher(&transport).with_chunk_bytes(4);
    assert_eq!(
        fetcher.get("https://github.com/a", 10, None).unwrap(),
        b"0123456789"
    );
}

#[test]
fn test_a_fetcher_closes_the_transport_it_made() {
    let made = FakeTransport::new(vec![(("github.com", "/a"), FakeResponse::ok(b"x"))]);
    let handle = made.clone();
    {
        let mut fetcher = Fetcher::with_factory(Box::new(move || Ok(Box::new(handle.clone()))));
        fetcher.get("https://github.com/a", 10, None).unwrap();
    }
    assert!(made.closed());
}

#[test]
fn test_without_a_transport_other_platforms_say_so() {
    let message = error_of(transport_for(false));
    assert!(message.contains("not available"), "{message}");
}

// -------------------------------------------------------------------------- download
#[test]
fn test_download_writes_the_file_and_reports_progress() {
    let dir = tempfile::tempdir().unwrap();
    let transport =
        FakeTransport::new(vec![(("github.com", "/f"), FakeResponse::ok(&[b'x'; 100]))]);
    let mut seen: Vec<(u64, u64)> = Vec::new();
    let out = dir.path().join("f.part");
    let size = fetcher(&transport)
        .download(
            "https://github.com/f",
            &out,
            100,
            Some(&mut |done, total| seen.push((done, total))),
            None,
        )
        .unwrap();
    assert_eq!(size, 100);
    assert_eq!(std::fs::read(&out).unwrap(), vec![b'x'; 100]);
    assert_eq!(seen.last(), Some(&(100, 100)));
}

#[test]
fn test_download_stops_at_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let response = FakeResponse::ok(&[b'x'; 100]).announcing(None);
    let transport = FakeTransport::new(vec![(("github.com", "/f"), response.clone())]);
    let message = error_of(fetcher(&transport).download(
        "https://github.com/f",
        &dir.path().join("f"),
        50,
        None,
        None,
    ));
    assert!(message.contains("larger"), "{message}");
    assert!(response.closed());
}

#[test]
fn test_download_can_be_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let response = FakeResponse::ok(&[b'x'; 100]);
    let transport = FakeTransport::new(vec![(("github.com", "/f"), response.clone())]);
    let polls = std::cell::Cell::new(0);
    let cancelled = || {
        polls.set(polls.get() + 1);
        polls.get() == 3
    };
    let result = fetcher(&transport).with_chunk_bytes(10).download(
        "https://github.com/f",
        &dir.path().join("f"),
        100,
        None,
        Some(&cancelled),
    );
    assert!(matches!(result, Err(DownloadError::Cancelled)));
    assert!(response.closed());
    assert_eq!(response.pos(), 20);
}
