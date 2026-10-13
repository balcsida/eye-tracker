//! The update check's connection on Windows: WinHTTP.
//!
//! This is the only module of Eye Tracker that touches the network, and the only
//! exception in `cargo xtask check-privacy`. It is used for
//! one thing, the opt-in update check (see `docs/privacy.md`), and it is as small
//! as it can be:
//!
//! * WinHTTP is part of Windows. Using it adds no library to the app (no OpenSSL,
//!   no TLS crate), so the bundle scan stays as strict as before.
//! * Certificates are checked by Windows against its own root store, and proxy
//!   settings are the system's (automatic detection). Nothing here turns a check
//!   off.
//! * Redirects are switched off in WinHTTP: [`Fetcher`](super::fetch::Fetcher)
//!   follows them one hop at a time and checks every host.
//! * Only `GET` is sent, with a `User-Agent` that names the app and its
//!   version, and an `Accept` header. No cookie, no identifier, no body.
//!
//! Not used on other platforms: [`supported`](super::fetch::supported) is false
//! there and this module is empty.
#![cfg(windows)]

use std::ffi::c_void;
use std::ptr;

use windows::Win32::Networking::WinHttp::{
    WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_FLAG_SECURE, WINHTTP_OPTION_REDIRECT_POLICY,
    WINHTTP_OPTION_REDIRECT_POLICY_NEVER, WINHTTP_QUERY_CONTENT_LENGTH, WINHTTP_QUERY_FLAG_NUMBER,
    WINHTTP_QUERY_LOCATION, WINHTTP_QUERY_STATUS_CODE, WinHttpCloseHandle, WinHttpConnect,
    WinHttpOpen, WinHttpOpenRequest, WinHttpQueryHeaders, WinHttpReadData, WinHttpReceiveResponse,
    WinHttpSendRequest, WinHttpSetOption, WinHttpSetTimeouts,
};
use windows::core::{Error, HSTRING, PCWSTR, w};

use super::fetch::{NetworkError, Response, Transport};
use crate::{REPO_URL, VERSION};

fn user_agent() -> String {
    format!("EyeTracker/{VERSION} (+{REPO_URL})")
}

const PORT: u16 = 443;
/// Milliseconds: resolve, connect, send, receive.
const TIMEOUTS: (i32, i32, i32, i32) = (10_000, 10_000, 15_000, 30_000);

type Handle = *mut c_void;

/// The Windows error number of the last failed call, in a message for the user.
fn fail(what: &str) -> NetworkError {
    // from_thread() wraps GetLastError() as an HRESULT; the low 16 bits are the error number.
    let code = Error::from_thread().code().0 as u32 & 0xFFFF;
    NetworkError::new(format!("{what} (Windows error {code})."))
}

fn close(handle: Handle) {
    if !handle.is_null() {
        // SAFETY: the handle came from WinHTTP and is closed once (callers null their copy).
        let _ = unsafe { WinHttpCloseHandle(handle) };
    }
}

/// An answered request; [`Response::read`] hands out the body.
pub struct WinHttpResponse {
    request: Handle,
    connection: Handle,
    status: u32,
    content_length: Option<u64>,
    location: Option<String>,
}

impl WinHttpResponse {
    fn new(request: Handle, connection: Handle) -> Self {
        let mut response = Self {
            request,
            connection,
            status: 0,
            content_length: None,
            location: None,
        };
        response.status = response.number(WINHTTP_QUERY_STATUS_CODE).unwrap_or(0);
        response.content_length = response.number(WINHTTP_QUERY_CONTENT_LENGTH).map(u64::from);
        response.location = response.text(WINHTTP_QUERY_LOCATION);
        response
    }

    fn number(&self, level: u32) -> Option<u32> {
        let mut value = 0u32;
        let mut size = size_of::<u32>() as u32;
        // SAFETY: the buffer is `value` and `size` is its size in bytes.
        unsafe {
            WinHttpQueryHeaders(
                self.request,
                level | WINHTTP_QUERY_FLAG_NUMBER,
                PCWSTR::null(),
                Some(ptr::from_mut(&mut value).cast()),
                &mut size,
                ptr::null_mut(),
            )
        }
        .ok()
        .map(|()| value)
    }

    fn text(&self, level: u32) -> Option<String> {
        let mut buffer = [0u16; 2048];
        let mut size = size_of_val(&buffer) as u32;
        // SAFETY: the buffer is `buffer` and `size` is its size in bytes.
        unsafe {
            WinHttpQueryHeaders(
                self.request,
                level,
                PCWSTR::null(),
                Some(buffer.as_mut_ptr().cast()),
                &mut size,
                ptr::null_mut(),
            )
        }
        .ok()?;
        let units = (size as usize / 2).min(buffer.len());
        let text = String::from_utf16_lossy(&buffer[..units]);
        let text = text.trim_end_matches('\0');
        (!text.is_empty()).then(|| text.to_string())
    }
}

impl Response for WinHttpResponse {
    fn status(&self) -> u32 {
        self.status
    }

    fn location(&self) -> Option<String> {
        self.location.clone()
    }

    fn content_length(&self) -> Option<u64> {
        self.content_length
    }

    fn read(&mut self, size: usize) -> Result<Vec<u8>, NetworkError> {
        if self.request.is_null() {
            return Ok(Vec::new());
        }
        let mut buffer = vec![0u8; size];
        let mut read = 0u32;
        // SAFETY: the buffer holds `size` bytes (at most u32::MAX for the chunk sizes used).
        let result = unsafe {
            WinHttpReadData(
                self.request,
                buffer.as_mut_ptr().cast(),
                u32::try_from(size).unwrap_or(u32::MAX),
                &mut read,
            )
        };
        if result.is_err() {
            return Err(fail("The connection was interrupted"));
        }
        buffer.truncate(read as usize);
        Ok(buffer)
    }

    fn close(&mut self) {
        close(std::mem::replace(&mut self.request, ptr::null_mut()));
        close(std::mem::replace(&mut self.connection, ptr::null_mut()));
    }
}

impl Drop for WinHttpResponse {
    fn drop(&mut self) {
        Response::close(self);
    }
}

/// One WinHTTP session; every [`Transport::open`] is a single GET without redirects.
pub struct WinHttpTransport {
    session: Handle,
}

impl WinHttpTransport {
    pub fn new() -> Result<Self, NetworkError> {
        let agent = HSTRING::from(user_agent());
        // SAFETY: `agent` outlives the call; no proxy name and no bypass list are passed.
        let session = unsafe {
            WinHttpOpen(
                &agent,
                WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY,
                PCWSTR::null(),
                PCWSTR::null(),
                0,
            )
        };
        if session.is_null() {
            return Err(fail("Could not start a connection"));
        }
        // SAFETY: `session` is the open session handle.
        let _ =
            unsafe { WinHttpSetTimeouts(session, TIMEOUTS.0, TIMEOUTS.1, TIMEOUTS.2, TIMEOUTS.3) };
        Ok(Self { session })
    }

    fn send(&self, request: Handle, host: &str, accept: Option<&str>) -> Result<(), NetworkError> {
        let policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER.to_ne_bytes();
        // SAFETY: `request` is an open request handle; the option value is a DWORD.
        let _ = unsafe {
            WinHttpSetOption(
                Some(request.cast_const()),
                WINHTTP_OPTION_REDIRECT_POLICY,
                Some(&policy),
            )
        };
        let headers: Option<Vec<u16>> =
            accept.map(|accept| format!("Accept: {accept}\r\n").encode_utf16().collect());
        // SAFETY: `request` is open; `headers` lives through the call; no body is sent.
        let sent = unsafe { WinHttpSendRequest(request, headers.as_deref(), None, 0, 0, 0) };
        // SAFETY: `request` is open and its request was sent.
        if sent.is_err() || unsafe { WinHttpReceiveResponse(request, ptr::null_mut()) }.is_err() {
            return Err(fail(&format!("Could not reach {host}")));
        }
        Ok(())
    }
}

impl Transport for WinHttpTransport {
    fn open(
        &mut self,
        host: &str,
        path: &str,
        accept: Option<&str>,
    ) -> Result<Box<dyn Response>, NetworkError> {
        let server = HSTRING::from(host);
        // SAFETY: `self.session` is open and `server` outlives the call.
        let connection = unsafe { WinHttpConnect(self.session, &server, PORT, 0) };
        if connection.is_null() {
            return Err(fail(&format!("Could not reach {host}")));
        }
        let object = HSTRING::from(path);
        // SAFETY: `connection` is open; the strings outlive the call; the verb is GET over TLS.
        let request = unsafe {
            WinHttpOpenRequest(
                connection,
                w!("GET"),
                &object,
                PCWSTR::null(),
                PCWSTR::null(),
                ptr::null(),
                WINHTTP_FLAG_SECURE,
            )
        };
        if request.is_null() {
            let error = fail("Could not start a request");
            close(connection);
            return Err(error);
        }
        match self.send(request, host, accept) {
            Ok(()) => Ok(Box::new(WinHttpResponse::new(request, connection))),
            Err(error) => {
                close(request);
                close(connection);
                Err(error)
            }
        }
    }

    fn close(&mut self) {
        close(std::mem::replace(&mut self.session, ptr::null_mut()));
    }
}

impl Drop for WinHttpTransport {
    fn drop(&mut self) {
        Transport::close(self);
    }
}
