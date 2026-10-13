//! The rules for what the update check may fetch, on top of a bare transport.
//!
//! The transport ([`super::winhttp`]) only moves bytes. Everything that decides *what*
//! may be fetched lives here, in plain code that is tested without a network:
//!
//! * HTTPS only, port 443, no credentials in the address, and only GitHub's own
//!   hosts ([`ALLOWED_HOSTS`]): the API, the release pages and the hosts that
//!   serve release files;
//! * redirects are followed here, one at a time, and every hop is checked against
//!   the same rules (the transport never follows one by itself);
//! * a size limit per request, checked against the announced length and while the
//!   body arrives;
//! * nothing is sent but the address and an `Accept` header: no cookies, no
//!   identifier, no system information.

use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// The hosts a request, or a redirect of one, may go to.
pub const ALLOWED_HOSTS: [&str; 4] = [
    "api.github.com",
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
];
pub const MAX_REDIRECTS: usize = 5;
pub const CHUNK_BYTES: usize = 64 * 1024;
const REDIRECTS: [u32; 5] = [301, 302, 303, 307, 308];

/// A request that failed or was refused (the message is meant for the user).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct NetworkError(pub String);

impl NetworkError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Why a download did not finish. The user cancelling it is not a failure.
#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error(transparent)]
    Network(#[from] NetworkError),
    /// The user cancelled a download.
    #[error("The download was cancelled.")]
    Cancelled,
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// An open HTTP response (see [`super::winhttp::WinHttpTransport`]).
pub trait Response {
    fn status(&self) -> u32;
    fn location(&self) -> Option<String>;
    fn content_length(&self) -> Option<u64>;
    /// Up to `size` bytes of the body; empty at its end.
    fn read(&mut self, size: usize) -> Result<Vec<u8>, NetworkError>;
    fn close(&mut self);
}

pub trait Transport {
    /// Send one GET to `https://host/path` without following redirects.
    fn open(
        &mut self,
        host: &str,
        path: &str,
        accept: Option<&str>,
    ) -> Result<Box<dyn Response>, NetworkError>;
    fn close(&mut self);
}

/// Whether this platform has a transport (Windows, for now).
pub fn supported() -> bool {
    cfg!(windows)
}

pub fn default_transport() -> Result<Box<dyn Transport>, NetworkError> {
    transport_for(supported())
}

fn transport_for(supported: bool) -> Result<Box<dyn Transport>, NetworkError> {
    if !supported {
        return Err(NetworkError::new(
            "Update checks are not available on this platform yet.",
        ));
    }
    #[cfg(windows)]
    {
        Ok(Box::new(super::winhttp::WinHttpTransport::new()?))
    }
    #[cfg(not(windows))]
    unreachable!("supported() is false without WinHTTP")
}

/// `(host, path)` of an allowed address; [`NetworkError`] for any other.
///
/// The address must match `https://<host>[/<path>]` with a host of letters, digits, dots and
/// dashes only (so no credentials, no port) and a path without white space or fragment.
pub fn split_url(url: &str) -> Result<(String, String), NetworkError> {
    let parsed = url.strip_prefix("https://").and_then(|rest| {
        let host_len = rest
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'.' || *b == b'-')
            .count();
        let (host, path) = rest.split_at(host_len);
        let path_ok = path.is_empty()
            || (path.starts_with('/') && !path.chars().any(|c| c.is_whitespace() || c == '#'));
        (host_len > 0 && path_ok).then_some((host, path))
    });
    let Some((host, path)) = parsed else {
        return Err(NetworkError::new(
            "Refused an address that is not a plain https address.",
        ));
    };
    let host = host.to_ascii_lowercase();
    if !ALLOWED_HOSTS.contains(&host.as_str()) {
        return Err(NetworkError::new(format!(
            "Refused an address on {host}: not a GitHub host."
        )));
    }
    let path = if path.is_empty() { "/" } else { path };
    Ok((host, path.to_string()))
}

/// The absolute address a `Location` header names (a path stays on the host).
pub fn resolve_location(base_host: &str, location: &str) -> String {
    if location.starts_with('/') && !location.starts_with("//") {
        return format!("https://{base_host}{location}");
    }
    location.to_string()
}

type TransportFactory = Box<dyn Fn() -> Result<Box<dyn Transport>, NetworkError>>;

/// Fetches allowed addresses through a transport, with the limits above.
///
/// A transport passed in is not closed by the fetcher; one it made itself is, on
/// [`Fetcher::close`] or when the fetcher is dropped.
pub struct Fetcher {
    transport: Option<Box<dyn Transport>>,
    owned: bool,
    factory: TransportFactory,
    chunk_bytes: usize,
}

impl Default for Fetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl Fetcher {
    /// A fetcher that makes the platform's transport on first use.
    pub fn new() -> Self {
        Self::with_factory(Box::new(default_transport))
    }

    /// A fetcher over a transport that stays the caller's to close.
    pub fn with_transport(transport: Box<dyn Transport>) -> Self {
        let mut fetcher = Self::new();
        fetcher.transport = Some(transport);
        fetcher.owned = false;
        fetcher
    }

    /// A fetcher that makes its transport with `factory` on first use.
    pub fn with_factory(factory: TransportFactory) -> Self {
        Self {
            transport: None,
            owned: true,
            factory,
            chunk_bytes: CHUNK_BYTES,
        }
    }

    /// Read bodies in chunks of this size instead of [`CHUNK_BYTES`].
    pub fn with_chunk_bytes(mut self, chunk_bytes: usize) -> Self {
        self.chunk_bytes = chunk_bytes;
        self
    }

    pub fn close(&mut self) {
        if self.owned
            && let Some(mut transport) = self.transport.take()
        {
            transport.close();
        }
    }

    // ------------------------------------------------------------------ public
    /// The body of `url` (at most `max_bytes`).
    pub fn get(
        &mut self,
        url: &str,
        max_bytes: u64,
        accept: Option<&str>,
    ) -> Result<Vec<u8>, NetworkError> {
        let mut response = self.open(url, accept, max_bytes)?;
        let result = (|| {
            let mut body = Vec::new();
            loop {
                let chunk = response.read(self.chunk_bytes)?;
                if chunk.is_empty() {
                    return Ok(body);
                }
                if (body.len() + chunk.len()) as u64 > max_bytes {
                    return Err(NetworkError::new("The answer is larger than expected."));
                }
                body.extend_from_slice(&chunk);
            }
        })();
        response.close();
        result
    }

    /// Write `url` to `dest` and return the size. `progress(done, total)` is
    /// called as data arrives (`total` is 0 when the length is unknown);
    /// `cancelled()` is polled between chunks ([`DownloadError::Cancelled`]).
    pub fn download(
        &mut self,
        url: &str,
        dest: &Path,
        max_bytes: u64,
        mut progress: Option<&mut dyn FnMut(u64, u64)>,
        cancelled: Option<&dyn Fn() -> bool>,
    ) -> Result<u64, DownloadError> {
        let mut response = self.open(url, None, max_bytes)?;
        let total = response.content_length().unwrap_or(0);
        let result = (|| {
            let mut out = File::create(dest)?;
            let mut done = 0u64;
            loop {
                if cancelled.is_some_and(|is_cancelled| is_cancelled()) {
                    return Err(DownloadError::Cancelled);
                }
                let chunk = response.read(self.chunk_bytes)?;
                if chunk.is_empty() {
                    return Ok(done);
                }
                done += chunk.len() as u64;
                if done > max_bytes {
                    return Err(NetworkError::new("The download is larger than expected.").into());
                }
                out.write_all(&chunk)?;
                if let Some(report) = progress.as_mut() {
                    report(done, total);
                }
            }
        })();
        response.close();
        result
    }

    // --------------------------------------------------------------- internals
    fn open(
        &mut self,
        url: &str,
        accept: Option<&str>,
        max_bytes: u64,
    ) -> Result<Box<dyn Response>, NetworkError> {
        if self.transport.is_none() {
            self.transport = Some((self.factory)()?);
        }
        let Some(transport) = self.transport.as_mut() else {
            return Err(NetworkError::new("No connection."));
        };
        let mut url = url.to_string();
        for _hop in 0..=MAX_REDIRECTS {
            let (host, path) = split_url(&url)?;
            let mut response = transport.open(&host, &path, accept)?;
            if response.status() == 200 {
                if response
                    .content_length()
                    .is_some_and(|length| length > max_bytes)
                {
                    response.close();
                    return Err(NetworkError::new("The file is larger than expected."));
                }
                return Ok(response);
            }
            let (status, location) = (response.status(), response.location());
            response.close();
            match location {
                Some(location) if REDIRECTS.contains(&status) && !location.is_empty() => {
                    url = resolve_location(&host, &location);
                }
                _ => {
                    return Err(NetworkError::new(format!(
                        "GitHub answered with HTTP {status}."
                    )));
                }
            }
        }
        Err(NetworkError::new("Too many redirects."))
    }
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        self.close();
    }
}

#[cfg(test)]
#[path = "fetch_tests.rs"]
pub(crate) mod tests;
