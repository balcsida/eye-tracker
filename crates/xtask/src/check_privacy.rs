//! Privacy gate for Eye Tracker: the source tree, its dependencies and the built bundle.
//!
//! Eye Tracker promises that it makes no network connections and never writes
//! camera frames to disk. This command enforces that promise mechanically, in
//! three places.
//!
//! Source tree (default)
//! =====================
//!
//! It parses (never compiles) every Rust file under `crates/eye-tracker/src` with
//! `syn` and reports:
//!
//! `network-import`
//!     A `use`, `extern crate` or path of a networking crate or module (`std::net`,
//!     `tokio::net`, `socket2`, `reqwest`, `ureq`, `hyper`, `curl`, `native_tls`,
//!     `rustls`, `openssl`, a telemetry SDK, a URL opener such as `webbrowser` or
//!     `open`...).
//! `network-api`
//!     A name that opens a connection or a server or resolves a host however it was
//!     reached (`TcpStream`, `TcpListener`, `UdpSocket`, `ToSocketAddrs`,
//!     `to_socket_addrs`, `lookup_host`), e.g. through a glob import.
//! `frame-write`
//!     Any use of an image encoder that could put a frame on disk or into a byte
//!     buffer (the `image` crate's `save_buffer`, `save_with_format`, `write_to`,
//!     `write_with_encoder` and its `*Encoder` types), a `save` call in a file that
//!     uses the `image` crate, and a `std::fs::write`/`write`/`write_all` of
//!     something named like a frame.
//! `native-network`
//!     Windows' networking APIs (`windows::Win32::Networking`, `NetworkManagement`),
//!     a native networking library loaded at run time (`LoadLibraryW`, `dlopen`,
//!     `libloading`) or linked (`#[link(name = "ws2_32")]`), native resolver and
//!     connection functions (`getaddrinfo`, `WSAStartup`, `WinHttpOpen`...), and raw
//!     socket calls through `libc` (`libc::socket`, `libc::connect`...).
//! `network-command`
//!     A network command-line tool started as a subprocess (`curl`, `wget`,
//!     `ssh`...), also behind a shell or wrapper (`sh -c "curl ..."`,
//!     `powershell -c "iwr ..."`, `sudo wget`).
//! `macos-network`
//!     Apple's networking APIs as objc2 exposes them: the URL-loading, host-name and
//!     Bonjour classes (`NSURLSession` and its tasks, `NSURLConnection`,
//!     `NSURLDownload`, `NSURLRequest`, `NSHost`, `NSNetService`...) and Core
//!     Foundation's host, socket and HTTP stream functions
//!     (`CFStreamCreatePairWithSocketToHost`...), however they are reached (path,
//!     `use`, `class!`, `AnyClass::get`, `NSClassFromString`); a call of a
//!     `...WithContentsOfURL` selector (`NSData::dataWithContentsOfURL`,
//!     `NSString::stringWithContentsOfURL_encoding_error`, `initWithContentsOfURL`,
//!     `NSImage::initByReferencingURL`), which reads a web address as readily as a
//!     file, unless its URL is a file URL made with `NSURL::fileURLWithPath` (in
//!     place, or in a name that only ever holds one); and `NSBundle` loading
//!     `CFNetwork`, `Network` or `WebKit`.
//! `syntax-error`
//!     A file that cannot be parsed, and therefore cannot be verified.
//!
//! Test code is not part of the app: `*_tests.rs` files, the `testing/` helpers and
//! `#[cfg(test)]` items are skipped, as the Python gate never scanned `tests/`.
//!
//! The one reviewed exception is the opt-in update check: `update/winhttp.rs` may
//! use WinHTTP (`native-network` only; see [`SOURCE_ALLOWLIST`]). Every other rule
//! still applies to that file, and the rule applies to no other file. It is
//! reported as a notice on every run.
//!
//! Dependencies (default run)
//! ==========================
//!
//! The source check cannot see what third-party code does, so the default run also
//! reads the dependency tree of the app crate (`cargo metadata`, normal
//! dependencies on every platform) and fails on:
//!
//! `network-crate`
//!     A crate that depends on a networking crate ([`NETWORK_CRATES`]) and is not on
//!     the allow-list ([`DEPENDENCY_ALLOWLIST`], per crate and per networking crate,
//!     with the reason).
//!
//! Built bundle (`--bundle DIR`)
//! =============================
//!
//! The release workflow also scans the built output. Every native binary (PE, ELF
//! and Mach-O, found by their magic bytes, not by file name) is read with the
//! `object` crate, and the bundle fails on:
//!
//! `telemetry-endpoint`
//!     Any file containing a known telemetry/usage-logging marker
//!     (`play.googleapis.com`, Clearcut, Firebase logging, Sentry, Segment...).
//!     Never allow-listed.
//! `network-library`
//!     A binary that links a networking library (`WS2_32`, `WININET`, `WINHTTP`,
//!     `DNSAPI`... on Windows; `libcurl`, `libssl`, `libresolv`, `libkrb5`...
//!     elsewhere; the `CFNetwork`, `Network`, `GSS`, `Kerberos`, `LDAP` and `WebKit`
//!     frameworks on macOS) and is not on the allow-list.
//! `network-symbol`
//!     An ELF or Mach-O binary that imports host-name resolution or remote
//!     connection functions (`getaddrinfo`, `gethostbyname`...; on Linux and macOS
//!     sockets live in libc, so the imported *symbols* are what matter), or
//!     Foundation's URL-loading classes (`_OBJC_CLASS_$_NSURLSession`,
//!     `NSURLConnection`, `NSURLDownload`) and is not on the allow-list. Plain
//!     `socket`/`connect` are not flagged: D-Bus, X11 and the app's own IPC use them
//!     for local (`AF_UNIX`) connections.
//! `unreadable-binary`
//!     A native binary whose headers cannot be parsed, so it cannot be verified.
//!
//! Binaries that legitimately link networking code are listed in
//! [`BUNDLE_ALLOWLIST`], each with the exact indicators it may have and the reason;
//! anything else fails. Allow-listed findings are printed as notices so every
//! release log shows them.
//!
//! Usage:
//!
//! ```text
//! cargo xtask check-privacy                    # scan crates/eye-tracker/src and its dependencies
//! cargo xtask check-privacy PATH ...           # scan other files or directories
//! cargo xtask check-privacy --bundle DIST_DIR  # scan a built bundle
//! ```
//!
//! Exit status: 0 when clean, 1 when violations were found, 2 on a usage error.
//!
//! The source check is deliberately conservative: it has no allow-list comments.
//! If it flags something legitimate, change the rule here so the exception is
//! reviewed. It cannot see through values computed at run time (a library name
//! built from variables, `save` on a frame in a file that hides the `image` crate
//! behind a re-export); code review covers those.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use object::read::elf::{Dyn as _, FileHeader, SectionHeader as _, Sym as _};
use object::read::macho::{FatArch, MachHeader, MachOFatFile32, MachOFatFile64, Nlist as _};
use object::read::pe::{ImageNtHeaders, ImageOptionalHeader as _, PeFile32, PeFile64};
use object::{LittleEndian as LE, elf, macho};
use proc_macro2::{Span, TokenStream, TokenTree};
use regex::Regex;
use regex::bytes::RegexSet;
use serde_json::Value;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{Attribute, Expr, Item, Lit, LitStr, UseTree};

/// The package whose sources and dependencies the default run checks.
pub const APP_PACKAGE: &str = "eye-tracker";

/// The repository root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// The app's sources, scanned when no path is given.
pub fn default_target() -> PathBuf {
    repo_root().join("crates").join("eye-tracker").join("src")
}

/// A source file that may break specific rules, and why.
///
/// `path` is relative to `crates/eye-tracker/src` (POSIX separators) and must match
/// exactly. Only the listed `rules` are allowed there; every other rule is still
/// enforced for the file, and no other file gets the exception.
#[derive(Debug, PartialEq, Eq)]
pub struct SourceAllowRule {
    pub path: &'static str,
    pub rules: &'static [&'static str],
    pub reason: &'static str,
}

/// The reviewed exceptions of the source check. Keep this list as short as it is:
/// each entry is a place where the app talks to the network, which its privacy
/// promise (README, docs/privacy.md) has to name.
pub const SOURCE_ALLOWLIST: &[SourceAllowRule] = &[SourceAllowRule {
    path: "update/winhttp.rs",
    rules: &["native-network"],
    reason: "the opt-in update check (off by default): one HTTPS GET a day to GitHub's release \
             address through Windows' own WinHTTP, with the host and the size of every request \
             checked by update/fetch.rs; sends no identifier",
}];

/// Crates that exist to talk to the network (or only wrap it), with `_` for `-` as
/// in Rust paths. Their use in the app's sources is a `network-import`, and a
/// dependency on one is a `network-crate` unless [`DEPENDENCY_ALLOWLIST`] allows it.
pub const NETWORK_CRATES: &[&str] = &[
    // Sockets, event loops for sockets, resolvers.
    "async_net",
    "hickory_proto",
    "hickory_resolver",
    "mio",
    "socket2",
    "trust_dns_proto",
    "trust_dns_resolver",
    // HTTP, TLS and other protocol clients and servers.
    "actix_web",
    "async_tungstenite",
    "attohttpc",
    "axum",
    "curl",
    "curl_sys",
    "grpcio",
    "h2",
    "h3",
    "hyper",
    "hyper_util",
    "isahc",
    "lettre",
    "libp2p",
    "mdns_sd",
    "native_tls",
    "openssl",
    "openssl_sys",
    "paho_mqtt",
    "quinn",
    "reqwest",
    "rumqttc",
    "rustls",
    "ssh2",
    "surf",
    "tokio_native_tls",
    "tokio_rustls",
    "tokio_tungstenite",
    "tonic",
    "tungstenite",
    "ureq",
    "warp",
    "websocket",
    "zeroconf",
    "zmq",
    // objc2's wrappers of Apple's networking frameworks.
    "objc2_cf_network",
    "objc2_network",
    "objc2_web_kit",
    // URL openers: a web address handed to the browser. Only egui's `open_url` may
    // do that (see DEPENDENCY_ALLOWLIST).
    "open",
    "opener",
    "webbrowser",
    // Telemetry, analytics and crash-reporting SDKs.
    "amplitude",
    "bugsnag",
    "datadog",
    "mixpanel",
    "newrelic",
    "opentelemetry",
    "opentelemetry_otlp",
    "posthog_rs",
    "rollbar",
    "segment",
    "sentry",
];

/// Modules whose parent crate is harmless but which exist to connect over sockets.
pub const NETWORK_SUBMODULES: &[&str] = &[
    "async_io::net",
    "async_std::net",
    "core::net",
    "nix::sys::socket",
    "rustix::net",
    "smol::net",
    "std::net",
    "tokio::net",
];

/// The `use` roots of paths inside the app crate (like Python's relative imports).
const RELATIVE_ROOTS: &[&str] = &["crate", "self", "super", "Self"];

/// Names that open connections or servers or resolve host names, matched wherever
/// they appear (a glob import of `std::net` reaches them without the module name).
pub const NETWORK_APIS: &[&str] = &[
    "TcpListener",
    "TcpStream",
    "ToSocketAddrs",
    "UdpSocket",
    "lookup_host",
    "to_socket_addrs",
];

/// Encoders that turn a frame into a file or a byte buffer (the `image` crate's).
pub const FRAME_WRITE_NAMES: &[&str] = &[
    "AvifEncoder",
    "BmpEncoder",
    "FarbfeldEncoder",
    "GifEncoder",
    "HdrEncoder",
    "IcoEncoder",
    "ImageEncoder",
    "JpegEncoder",
    "OpenExrEncoder",
    "PngEncoder",
    "PnmEncoder",
    "QoiEncoder",
    "TgaEncoder",
    "TiffEncoder",
    "WebPEncoder",
    "save_buffer",
    "save_buffer_with_format",
    "save_with_format",
    "write_buffer_with_format",
    "write_to",
    "write_with_encoder",
];
/// Crates whose images have a `save` method that writes a file.
const IMAGE_CRATES: &[&str] = &["image", "imageproc"];

/// Windows DLLs that provide networking (import names, lower case, no ".dll").
pub const WINDOWS_NETWORK_DLLS: &[&str] = &[
    "cryptnet",
    "dnsapi",
    "httpapi",
    "icmp",
    "iphlpapi",
    "mpr",
    "mswsock",
    "netapi32",
    "rasapi32",
    "urlmon",
    "webio",
    "websocket",
    "winhttp",
    "wininet",
    "wldap32",
    "ws2_32",
    "wsock32",
];
/// The `windows` crate's namespaces of the DLLs above (`Win32::Networking::WinSock`...).
const WINDOWS_NETWORK_NAMESPACES: &[&str] = &["Networking", "NetworkManagement"];

/// POSIX shared libraries (soname prefixes) for networking.
pub const POSIX_NETWORK_LIBRARIES: &[&str] = &[
    "libcurl",
    "libgnutls",
    "libgssapi",
    "libkrb5",
    "libldap",
    "libnghttp2",
    "libresolv",
    "libsoup",
    "libssl",
    "libwebsockets",
    "libzmq",
];
/// Apple's networking frameworks, the counterparts of libgssapi, libkrb5 and libldap,
/// and WebKit (a web browser engine).
pub const MACOS_NETWORK_FRAMEWORKS: &[&str] =
    &["CFNetwork", "GSS", "Kerberos", "LDAP", "Network", "WebKit"];

/// Library names the source check refuses to load: [`WINDOWS_NETWORK_DLLS`] and these.
const NATIVE_NETWORK_LIBS_EXTRA: &[&str] =
    &["cfnetwork", "curl", "libcurl", "libresolv", "libsoup"];

/// Native resolver/connection functions whose names only exist for networking;
/// flagged on any path (`ws2::WSAStartup`, `libc::getaddrinfo`...).
pub const NATIVE_NETWORK_FUNCTIONS: &[&str] = &[
    "DnsQuery_A",
    "DnsQuery_W",
    "GetAddrInfoW",
    "HttpOpenRequestW",
    "HttpSendRequestW",
    "IcmpSendEcho",
    "IcmpSendEcho2",
    "InternetConnectW",
    "InternetOpenUrlW",
    "InternetOpenW",
    "URLDownloadToFileW",
    "URLOpenStreamW",
    "getaddrinfo",
    "gethostbyaddr",
    "gethostbyname",
    "gethostbyname2",
    "gethostbyname_r",
    "getnameinfo",
];
/// Prefixes of native networking function families (`WSASocketW`, `WinHttpOpen`...).
const NATIVE_NETWORK_FUNCTION_PREFIXES: &[&str] = &[
    "WSA",
    "WinHttp",
    "curl_easy_",
    "curl_global_",
    "curl_multi_",
];
/// libc socket calls: harmless names elsewhere (`sender.connect`), so they are
/// flagged only as `libc::<call>`.
pub const NATIVE_SOCKET_CALLS: &[&str] = &[
    "accept", "accept4", "bind", "connect", "listen", "recvfrom", "recvmsg", "sendmsg", "sendto",
    "socket",
];
/// Calls that load a native library by name.
const LIBRARY_LOADERS: &[&str] = &[
    "GetModuleHandleA",
    "GetModuleHandleW",
    "LoadLibraryA",
    "LoadLibraryExA",
    "LoadLibraryExW",
    "LoadLibraryW",
    "dlopen",
];
/// Calls that look a function up by name in a loaded library.
const SYMBOL_LOOKUPS: &[&str] = &["GetProcAddress", "dlsym"];

/// Command-line tools that transfer data over the network.
pub const NETWORK_COMMANDS: &[&str] = &[
    "aria2c",
    "bitsadmin",
    "certutil",
    "curl",
    "dig",
    "fetch",
    "ftp",
    "http",
    "https",
    "invoke-restmethod",
    "invoke-webrequest",
    "irm",
    "iwr",
    "lynx",
    "nc",
    "ncat",
    "netcat",
    "nslookup",
    "ping",
    "rsync",
    "scp",
    "sftp",
    "socat",
    "ssh",
    "telnet",
    "tftp",
    "w3m",
    "wget",
    "xh",
];

/// Commands that run another command given as an argument (`sh -c "curl ..."`).
const COMMAND_WRAPPERS: &[&str] = &[
    "bash",
    "busybox",
    "cmd",
    "dash",
    "doas",
    "env",
    "flatpak-spawn",
    "nice",
    "nohup",
    "pkexec",
    "powershell",
    "pwsh",
    "sh",
    "start",
    "sudo",
    "timeout",
    "xargs",
    "zsh",
];

/// Calls whose string arguments are a command line (besides `Command::new`).
const COMMAND_CALLS: &[&str] = &[
    "CreateProcessA",
    "CreateProcessW",
    "ShellExecuteA",
    "ShellExecuteExW",
    "ShellExecuteW",
    "WinExec",
    "execl",
    "execle",
    "execlp",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "system",
    "which",
];

/// Macros that wrap a string literal (`w!("x")`, `ns_string!("x")`...).
const STRING_MACROS: &[&str] = &["c_str", "cstr", "h", "ns_string", "s", "w"];

/// Apple's networking classes and functions as objc2 exposes them (Foundation and
/// Core Foundation, which the app's macOS code loads): loading URLs, resolving host
/// names, Bonjour, sockets to hosts. The names are unique to Apple's frameworks, so
/// they are flagged wherever they appear, however they were reached.
pub const MACOS_NETWORK_NAMES: &[&str] = &[
    "NSHost",
    "NSMutableURLRequest",
    "NSNetService",
    "NSNetServiceBrowser",
    "NSURLConnection",
    "NSURLDownload",
    "NSURLRequest",
    "NSURLSession",
    "CFHostCreateWithName",
    "CFHostStartInfoResolution",
    "CFReadStreamCreateForHTTPRequest",
    "CFReadStreamCreateForStreamedHTTPRequest",
    "CFSocketConnectToAddress",
    "CFStreamCreatePairWithSocketToHost",
    "CFURLCreateDataAndPropertiesFromResource",
];
/// Families of the classes above (`NSURLSessionConfiguration`, `NSURLSessionDataTask`,
/// `NSURLConnectionDelegate`...).
const MACOS_NETWORK_PREFIXES: &[&str] = &["NSURLSession", "NSURLConnection"];
/// Selectors that read whatever a URL names, a web address as readily as a file:
/// `NSData::dataWithContentsOfURL`, `NSString::stringWithContentsOfURL_encoding_error`,
/// `initWithContentsOfURL`, `NSImage::initByReferencingURL`... They are allowed with a
/// file URL made in place.
static MACOS_URL_READ: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z](?:WithContentsOfURL|ByReferencingURL)").unwrap());
/// `NSURL` constructors that can only make a file URL (`fileURLWithPath`...).
const FILE_URL_PREFIX: &str = "fileURLWith";
/// `NSURL` constructors from a string: a file URL when the literal says `file:`.
const URL_FROM_STRING: &[&str] = &[
    "URLWithString",
    "URLWithString_relativeToURL",
    "initWithString",
];
/// How objc2 code reaches a class by name (`AnyClass::get` too).
const OBJC_CLASS_LOOKUPS: &[&str] = &[
    "NSClassFromString",
    "lookUpClass",
    "objc_getClass",
    "objc_lookUpClass",
];
const OBJC_BUNDLE_LOADERS: &[&str] = &["bundleWithIdentifier", "bundleWithPath", "bundleWithURL"];
/// Frameworks that `NSBundle` must not load (by name, path or identifier).
pub const MACOS_NETWORK_BUNDLES: &[&str] = &["CFNetwork", "Network", "WebKit"];

const SKIP_DIRS: &[&str] = &[".git", "node_modules", "target", "testing"];
/// shortcut: test files are recognised by the naming convention in docs/architecture.md
/// (`*_tests.rs`, `src/testing/`), not by following `#[cfg(test)] mod` declarations; upgrade
/// if the convention ever changes.
const TEST_FILE_SUFFIX: &str = "_tests.rs";
static SHELL_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"[^\s;&|()`'"<>$]+"#).unwrap());

/// One privacy rule broken at a specific source location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Violation {
    pub path: PathBuf,
    pub line: usize,
    pub col: usize,
    pub rule: &'static str,
    pub message: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}:{}: {}: {}",
            display_path(&self.path),
            self.line,
            self.col,
            self.rule,
            self.message
        )
    }
}

/// Outcome of scanning a set of files.
#[derive(Debug, Default)]
pub struct ScanResult {
    pub files: Vec<PathBuf>,
    pub violations: Vec<Violation>,
    /// Findings that a [`SOURCE_ALLOWLIST`] entry allows (always reported).
    pub notices: Vec<(Violation, &'static SourceAllowRule)>,
}

impl ScanResult {
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }
}

// --------------------------------------------------------------------------- helpers
/// Path relative to the working directory when possible (clickable in CI logs).
pub fn display_path(path: &Path) -> String {
    let relative = std::env::current_dir()
        .and_then(|cwd| Ok((path.canonicalize()?, cwd.canonicalize()?)))
        .ok()
        .and_then(|(path, cwd)| path.strip_prefix(cwd).ok().map(Path::to_path_buf));
    posix(relative.as_deref().unwrap_or(path))
}

fn posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn is_network_module(module: &str) -> bool {
    let first = module.split("::").next().unwrap_or_default();
    NETWORK_CRATES.contains(&first)
        || NETWORK_SUBMODULES
            .iter()
            .any(|sub| module == *sub || module.starts_with(&format!("{sub}::")))
}

/// Normalise a command token: `C:\tools\curl.exe` → `curl`.
fn command_word(text: &str) -> String {
    let text = text
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .replace('\\', "/");
    let mut word = text.rsplit('/').next().unwrap_or_default().to_lowercase();
    for suffix in [".exe", ".cmd", ".bat", ".ps1"] {
        if let Some(stem) = word.strip_suffix(suffix) {
            word = stem.to_string();
        }
    }
    word
}

/// The first network tool anywhere in a shell command line, or `None`.
///
/// Every word is checked, not only the first, so `sudo wget ...`,
/// `sh -c "curl ..."` and `a && curl ...` are all caught.
fn network_tool_in(command_line: &str) -> Option<String> {
    SHELL_WORD
        .find_iter(command_line)
        .map(|word| word.as_str())
        .find(|word| NETWORK_COMMANDS.contains(&command_word(word).as_str()))
        .map(str::to_string)
}

fn native_lib_stem(text: &str) -> String {
    let text = text.replace('\\', "/");
    let mut stem = text.rsplit('/').next().unwrap_or_default().to_lowercase();
    for suffix in [".dll", ".dylib"] {
        if let Some(rest) = stem.strip_suffix(suffix) {
            stem = rest.to_string();
        }
    }
    stem.split(".so").next().unwrap_or_default().to_string()
}

fn is_native_network_lib(text: &str) -> bool {
    let stem = native_lib_stem(text);
    if WINDOWS_NETWORK_DLLS.contains(&stem.as_str())
        || NATIVE_NETWORK_LIBS_EXTRA.contains(&stem.as_str())
        || POSIX_NETWORK_LIBRARIES
            .iter()
            .any(|prefix| stem.starts_with(prefix))
    {
        return true;
    }
    // macOS frameworks: ".../Network.framework/Network", ".../CFNetwork.framework/...",
    // ".../WebKit.framework/WebKit"
    let path = text.replace('\\', "/").to_lowercase();
    path.contains("network.framework") || path.contains("/webkit.framework")
}

fn is_native_network_function(name: &str) -> bool {
    NATIVE_NETWORK_FUNCTIONS.contains(&name)
        || NATIVE_NETWORK_FUNCTION_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

fn is_macos_network_name(name: &str) -> bool {
    MACOS_NETWORK_NAMES.contains(&name)
        || MACOS_NETWORK_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// `"CFNetwork"`, `".../WebKit.framework"` or `"com.apple.Network"`.
fn is_macos_network_bundle(text: &str) -> bool {
    let lowered = text.to_lowercase();
    MACOS_NETWORK_BUNDLES.iter().any(|name| {
        let name = name.to_lowercase();
        lowered == name
            || lowered == format!("com.apple.{name}")
            || lowered.contains(&format!("/{name}.framework"))
    })
}

fn is_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident("cfg") && attr.parse_args::<syn::Ident>().is_ok_and(|i| i == "test")
    })
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(i) => &i.attrs,
        Item::Enum(i) => &i.attrs,
        Item::ExternCrate(i) => &i.attrs,
        Item::Fn(i) => &i.attrs,
        Item::ForeignMod(i) => &i.attrs,
        Item::Impl(i) => &i.attrs,
        Item::Macro(i) => &i.attrs,
        Item::Mod(i) => &i.attrs,
        Item::Static(i) => &i.attrs,
        Item::Struct(i) => &i.attrs,
        Item::Trait(i) => &i.attrs,
        Item::TraitAlias(i) => &i.attrs,
        Item::Type(i) => &i.attrs,
        Item::Union(i) => &i.attrs,
        Item::Use(i) => &i.attrs,
        _ => &[],
    }
}

fn segments(path: &syn::Path) -> Vec<String> {
    path.segments.iter().map(|s| s.ident.to_string()).collect()
}

/// Every leaf of a `use` tree as its full path (`std::{fs, net::*}` → `std::fs`, `std::net::*`).
fn use_leaves(tree: &UseTree, prefix: &mut Vec<String>, out: &mut Vec<Vec<String>>) {
    match tree {
        UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            use_leaves(&p.tree, prefix, out);
            prefix.pop();
        }
        UseTree::Name(n) if n.ident == "self" => out.push(prefix.clone()),
        UseTree::Name(n) => out.push([prefix.as_slice(), &[n.ident.to_string()]].concat()),
        UseTree::Rename(r) => out.push([prefix.as_slice(), &[r.ident.to_string()]].concat()),
        UseTree::Glob(_) => out.push([prefix.as_slice(), &["*".to_string()]].concat()),
        UseTree::Group(g) => g
            .items
            .iter()
            .for_each(|item| use_leaves(item, prefix, out)),
    }
}

fn lit_text(lit: &Lit) -> Option<String> {
    match lit {
        Lit::Str(s) => Some(s.value()),
        Lit::ByteStr(s) => Some(
            String::from_utf8_lossy(&s.value())
                .trim_end_matches('\0')
                .into(),
        ),
        Lit::CStr(s) => Some(s.value().to_string_lossy().into_owned()),
        _ => None,
    }
}

/// The string a literal expression holds: `"x"`, `&"x"`, `"x".to_string()`,
/// `c"x".as_ptr()`, `w!("x")`, `String::from("x")`...
fn string_lit(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(e) => lit_text(&e.lit),
        Expr::Reference(e) => string_lit(&e.expr),
        Expr::Paren(e) => string_lit(&e.expr),
        Expr::Group(e) => string_lit(&e.expr),
        Expr::MethodCall(e) if e.args.is_empty() => string_lit(&e.receiver),
        Expr::Call(e) if e.args.len() == 1 => match &*e.func {
            Expr::Path(p)
                if p.path.segments.last().is_some_and(|s| {
                    ["from", "from_str", "new"].contains(&s.ident.to_string().as_str())
                }) =>
            {
                string_lit(&e.args[0])
            }
            _ => None,
        },
        Expr::Macro(e) => {
            let name = e.mac.path.segments.last()?.ident.to_string();
            if !STRING_MACROS.contains(&name.as_str()) {
                return None;
            }
            e.mac.parse_body::<Lit>().ok().as_ref().and_then(lit_text)
        }
        _ => None,
    }
}

/// `a.b.c` for a variable or a field chain, `None` for anything else.
fn dotted_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(p) if p.qself.is_none() && p.path.segments.len() == 1 => {
            Some(p.path.segments[0].ident.to_string())
        }
        Expr::Field(f) => {
            let member = match &f.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(index) => index.index.to_string(),
            };
            Some(format!("{}.{member}", dotted_name(&f.base)?))
        }
        _ => None,
    }
}

/// Whether an expression names a frame (an identifier containing "frame", or
/// `to_rgb_image`, which turns a `Frame` into an image).
fn mentions_frame(expr: &Expr) -> bool {
    struct Finder(bool);
    impl Visit<'_> for Finder {
        fn visit_ident(&mut self, ident: &syn::Ident) {
            let name = ident.to_string().to_lowercase();
            self.0 |= name.contains("frame") || name == "to_rgb_image";
        }
    }
    let mut finder = Finder(false);
    finder.visit_expr(expr);
    finder.0
}

fn location(span: Span) -> (usize, usize) {
    let start = span.start();
    (start.line, start.column + 1)
}

// --------------------------------------------------------------------------- scanner
/// What the scanner needs to know about the whole file before checking it.
#[derive(Default)]
struct Collector {
    /// Modules declared in this file (`mod winhttp;`): paths into them are the app's own.
    local_modules: HashSet<String>,
    /// Whether the file uses the `image` crate, so a `save` call writes an image.
    uses_image: bool,
    /// Every value assigned to a name or field chain (for the file-URL analysis).
    assignments: HashMap<String, Vec<Expr>>,
}

impl Visit<'_> for Collector {
    fn visit_item(&mut self, item: &Item) {
        if !is_cfg_test(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_item_mod(&mut self, item: &syn::ItemMod) {
        self.local_modules.insert(item.ident.to_string());
        visit::visit_item_mod(self, item);
    }

    fn visit_item_use(&mut self, item: &syn::ItemUse) {
        let mut leaves = Vec::new();
        use_leaves(&item.tree, &mut Vec::new(), &mut leaves);
        self.uses_image |= leaves.iter().any(|leaf| {
            leaf.first()
                .is_some_and(|f| IMAGE_CRATES.contains(&f.as_str()))
        });
    }

    fn visit_path(&mut self, path: &syn::Path) {
        let segs = segments(path);
        self.uses_image |= segs.len() > 1 && IMAGE_CRATES.contains(&segs[0].as_str());
        visit::visit_path(self, path);
    }

    fn visit_local(&mut self, local: &syn::Local) {
        let mut pat = &local.pat;
        if let syn::Pat::Type(typed) = pat {
            pat = &typed.pat;
        }
        if let (syn::Pat::Ident(ident), Some(init)) = (pat, &local.init) {
            let values = self.assignments.entry(ident.ident.to_string()).or_default();
            values.push((*init.expr).clone());
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_assign(&mut self, assign: &syn::ExprAssign) {
        if let Some(name) = dotted_name(&assign.left) {
            self.assignments
                .entry(name)
                .or_default()
                .push((*assign.right).clone());
        }
        visit::visit_expr_assign(self, assign);
    }
}

/// Collects violations for one parsed file.
struct ModuleScanner<'p> {
    path: &'p Path,
    violations: Vec<Violation>,
    local_modules: HashSet<String>,
    uses_image: bool,
    /// Names and field chains that only ever hold a file URL
    /// (`let url = NSURL::fileURLWithPath(&path)`).
    file_urls: HashSet<String>,
}

impl<'p> ModuleScanner<'p> {
    fn new(path: &'p Path, file: &syn::File) -> Self {
        let mut collector = Collector::default();
        collector.visit_file(file);
        let mut scanner = ModuleScanner {
            path,
            violations: Vec::new(),
            local_modules: collector.local_modules,
            uses_image: collector.uses_image,
            file_urls: HashSet::new(),
        };
        // A name assigned from another file-URL name: a few rounds reach the fixpoint.
        for _ in 0..4 {
            let found: HashSet<String> = collector
                .assignments
                .iter()
                .filter(|(_, values)| values.iter().all(|v| scanner.is_file_url(Some(v))))
                .map(|(name, _)| name.clone())
                .collect();
            if found.len() == scanner.file_urls.len() {
                break;
            }
            scanner.file_urls = found;
        }
        scanner
    }

    fn report(&mut self, span: Span, rule: &'static str, message: String) {
        let (line, col) = location(span);
        self.violations.push(Violation {
            path: self.path.to_path_buf(),
            line,
            col,
            rule,
            message,
        });
    }

    fn run(mut self, file: &syn::File) -> Vec<Violation> {
        self.visit_file(file);
        let unique: BTreeSet<Violation> = self.violations.into_iter().collect();
        unique.into_iter().collect()
    }

    fn is_relative(&self, segs: &[String]) -> bool {
        segs.first()
            .is_some_and(|f| RELATIVE_ROOTS.contains(&f.as_str()) || self.local_modules.contains(f))
    }

    // ------------------------------------------------------------------ names
    /// Names flagged wherever they appear (path segment, method, macro token).
    fn check_name(&mut self, name: &str, span: Span) {
        if FRAME_WRITE_NAMES.contains(&name) {
            self.report(
                span,
                "frame-write",
                format!("use of '{name}' (frames must never be encoded)"),
            );
        }
        if NETWORK_APIS.contains(&name) {
            self.report(
                span,
                "network-api",
                format!("use of '{name}' (opens a network connection)"),
            );
        }
        if is_macos_network_name(name) {
            self.report(
                span,
                "macos-network",
                format!("use of '{name}' (macOS networking)"),
            );
        }
    }

    /// A `use` leaf or a path outside the app crate (`how` is "import" or "use").
    fn check_path(&mut self, segs: &[String], span: Span, how: &str) {
        if self.is_relative(segs) {
            return;
        }
        let joined = segs.join("::");
        if (segs.len() > 1 || how == "import") && is_network_module(&joined) {
            self.report(
                span,
                "network-import",
                format!("{how} of networking module '{joined}'"),
            );
        }
        // A native function is flagged however it was reached (a bare call after a glob import too).
        if let Some(function) = segs.iter().find(|s| is_native_network_function(s)) {
            self.report(
                span,
                "native-network",
                format!("native networking function '{function}'"),
            );
        } else if segs.len() > 1 {
            let native = if segs.iter().any(|s| s == "Win32")
                && segs
                    .iter()
                    .any(|s| WINDOWS_NETWORK_NAMESPACES.contains(&s.as_str()))
            {
                Some(format!("{how} of native networking API '{joined}'"))
            } else if let Some(lib) = segs.iter().find(|s| {
                let lower = s.to_lowercase();
                WINDOWS_NETWORK_DLLS.contains(&lower.as_str())
                    || NATIVE_NETWORK_LIBS_EXTRA.contains(&lower.as_str())
            }) {
                Some(format!("native networking library '{lib}'"))
            } else {
                let (prev, last) = (&segs[segs.len() - 2], &segs[segs.len() - 1]);
                (prev == "libc" && NATIVE_SOCKET_CALLS.contains(&last.as_str()))
                    .then(|| format!("raw socket call '{last}' through libc"))
            };
            if let Some(message) = native {
                self.report(span, "native-network", message);
            }
        }
        for seg in segs {
            self.check_name(seg, span);
        }
    }

    // --------------------------------------------------------------- file URLs
    /// Whether `expr` is certainly a file URL: `NSURL::fileURLWithPath(...)`,
    /// `NSURL::URLWithString(ns_string!("file:..."))`, or a name that only ever holds one.
    fn is_file_url(&self, expr: Option<&Expr>) -> bool {
        let Some(expr) = expr else { return false };
        match expr {
            Expr::Reference(e) => self.is_file_url(Some(&e.expr)),
            Expr::Paren(e) => self.is_file_url(Some(&e.expr)),
            Expr::Unary(e) => self.is_file_url(Some(&e.expr)),
            Expr::MethodCall(e)
                if ["clone", "expect", "unwrap"].contains(&e.method.to_string().as_str()) =>
            {
                self.is_file_url(Some(&e.receiver))
            }
            Expr::Unsafe(e) => match e.block.stmts.as_slice() {
                [syn::Stmt::Expr(inner, None)] => self.is_file_url(Some(inner)),
                _ => false,
            },
            Expr::Call(call) => {
                let Expr::Path(func) = &*call.func else {
                    return false;
                };
                let name = func
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if name.starts_with(FILE_URL_PREFIX) {
                    return true;
                }
                let literal = call.args.first().and_then(string_lit).unwrap_or_default();
                URL_FROM_STRING.contains(&name.as_str())
                    && literal.to_lowercase().starts_with("file:")
            }
            _ => dotted_name(expr).is_some_and(|name| self.file_urls.contains(&name)),
        }
    }

    /// `NSData::dataWithContentsOfURL(url)` and its kind, unless `url` is a file URL.
    fn check_url_read(&mut self, url: Option<&Expr>, name: &str, span: Span) {
        if !self.is_file_url(url) {
            self.report(
                span,
                "macos-network",
                format!(
                    "'{name}' reads whatever its URL names, a web address too \
                     (pass a file URL made with NSURL::fileURLWithPath)"
                ),
            );
        }
    }

    // -------------------------------------------------------------------- calls
    fn check_call(
        &mut self,
        path: &syn::Path,
        args: &Punctuated<Expr, syn::Token![,]>,
        span: Span,
    ) {
        let segs = segments(path);
        let name = segs.last().map(String::as_str).unwrap_or_default();
        let prev = if segs.len() > 1 {
            segs[segs.len() - 2].as_str()
        } else {
            ""
        };
        let literals: Vec<String> = args.iter().filter_map(string_lit).collect();

        if (LIBRARY_LOADERS.contains(&name) || (prev == "Library" && name == "new"))
            && let Some(lib) = literals.iter().find(|l| is_native_network_lib(l))
        {
            self.report(
                span,
                "native-network",
                format!("loads native networking library '{lib}'"),
            );
        }
        if SYMBOL_LOOKUPS.contains(&name)
            && let Some(function) = literals.iter().find(|l| is_native_network_function(l))
        {
            self.report(
                span,
                "native-network",
                format!("{name}(..., '{function}') through a library handle"),
            );
        }
        if ((prev == "Command" && name == "new") || COMMAND_CALLS.contains(&name))
            && let Some(tool) = literals.iter().find_map(|l| network_tool_in(l))
        {
            self.report(
                span,
                "network-command",
                format!("runs network tool '{tool}'"),
            );
        }
        if MACOS_URL_READ.is_match(name) {
            // objc2's init methods take the allocated object first.
            let url = args.iter().nth(usize::from(name.starts_with("init")));
            self.check_url_read(url, name, span);
        }
        if (OBJC_CLASS_LOOKUPS.contains(&name) || (prev == "AnyClass" && name == "get"))
            && let Some(literal) = args.first().and_then(string_lit)
            && is_macos_network_name(&literal)
        {
            self.report(
                span,
                "macos-network",
                format!("{name}('{literal}') (macOS networking)"),
            );
        }
        if OBJC_BUNDLE_LOADERS.contains(&name)
            && let Some(text) = literals.iter().find(|l| is_macos_network_bundle(l))
        {
            self.report(span, "macos-network", format!("{name} loads '{text}'"));
        }
        if prev == "fs" && name == "write" && args.iter().any(mentions_frame) {
            self.report(
                span,
                "frame-write",
                "fs::write of a frame (frames must never be written to disk)".into(),
            );
        }
    }

    /// `["curl", url]` is flagged wherever it is built, so a command stored in a
    /// variable before being run is caught too. Behind a wrapper (`["sh", "-c",
    /// "curl ..."]`, `["sudo", "wget", ...]`) every argument is a command line of its own.
    fn check_command_sequence<'e>(
        &mut self,
        mut elements: impl Iterator<Item = &'e Expr>,
        span: Span,
    ) {
        let Some(head) = elements.next().and_then(string_lit) else {
            return;
        };
        let word = command_word(&head);
        if NETWORK_COMMANDS.contains(&word.as_str()) {
            self.report(
                span,
                "network-command",
                format!("command line runs network tool '{head}'"),
            );
            return;
        }
        if !COMMAND_WRAPPERS.contains(&word.as_str()) {
            return;
        }
        if let Some(tool) = elements
            .filter_map(string_lit)
            .find_map(|text| network_tool_in(&text))
        {
            self.report(
                span,
                "network-command",
                format!("command line runs network tool '{tool}' via '{head}'"),
            );
        }
    }

    /// `Command::new("sh").arg("-c").args([...])`: the program the chain starts.
    fn command_head(expr: &Expr) -> Option<String> {
        match expr {
            Expr::MethodCall(m) => Self::command_head(&m.receiver),
            Expr::Reference(e) => Self::command_head(&e.expr),
            Expr::Paren(e) => Self::command_head(&e.expr),
            Expr::Call(call) => match &*call.func {
                Expr::Path(p) if segments(&p.path).ends_with(&["Command".into(), "new".into()]) => {
                    call.args.first().and_then(string_lit)
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn check_wrapped_arguments(&mut self, call: &syn::ExprMethodCall) {
        let Some(head) = Self::command_head(&call.receiver) else {
            return;
        };
        if !COMMAND_WRAPPERS.contains(&command_word(&head).as_str()) {
            return;
        }
        let mut texts = Vec::new();
        for arg in &call.args {
            let mut arg = arg;
            while let Expr::Reference(e) = arg {
                arg = &e.expr;
            }
            match arg {
                Expr::Array(array) => texts.extend(array.elems.iter().filter_map(string_lit)),
                Expr::Macro(m) if m.mac.path.is_ident("vec") => {
                    if let Ok(items) = m
                        .mac
                        .parse_body_with(Punctuated::<Expr, syn::Token![,]>::parse_terminated)
                    {
                        texts.extend(items.iter().filter_map(string_lit));
                    }
                }
                other => texts.extend(string_lit(other)),
            }
        }
        if let Some(tool) = texts.iter().find_map(|text| network_tool_in(text)) {
            self.report(
                call.method.span(),
                "network-command",
                format!("command line runs network tool '{tool}' via '{head}'"),
            );
        }
    }

    /// Identifiers in a macro body that does not parse as expressions (`msg_send!`...).
    fn scan_tokens(&mut self, tokens: TokenStream) {
        for tree in tokens {
            match tree {
                TokenTree::Ident(ident) => {
                    let name = ident.to_string();
                    self.check_name(&name, ident.span());
                    if is_native_network_function(&name) {
                        self.report(
                            ident.span(),
                            "native-network",
                            format!("native networking function '{name}'"),
                        );
                    }
                    if MACOS_URL_READ.is_match(&name) {
                        self.report(
                            ident.span(),
                            "macos-network",
                            format!("'{name}' reads whatever its URL names"),
                        );
                    }
                }
                TokenTree::Group(group) => self.scan_tokens(group.stream()),
                _ => {}
            }
        }
    }
}

impl Visit<'_> for ModuleScanner<'_> {
    fn visit_item(&mut self, item: &Item) {
        // Test code is not part of the app (see the module documentation).
        if !is_cfg_test(item_attrs(item)) {
            visit::visit_item(self, item);
        }
    }

    fn visit_item_use(&mut self, item: &syn::ItemUse) {
        let span = match &item.vis {
            syn::Visibility::Inherited => item.use_token.span,
            vis => vis.span(),
        };
        let mut leaves = Vec::new();
        use_leaves(&item.tree, &mut Vec::new(), &mut leaves);
        for leaf in leaves {
            self.check_path(&leaf, span, "import");
        }
    }

    fn visit_item_extern_crate(&mut self, item: &syn::ItemExternCrate) {
        self.check_path(&[item.ident.to_string()], item.extern_token.span, "import");
    }

    fn visit_item_foreign_mod(&mut self, item: &syn::ItemForeignMod) {
        for attr in item.attrs.iter().filter(|a| a.path().is_ident("link")) {
            let mut names = Vec::new();
            let _ = attr.parse_nested_meta(|meta| {
                let value = meta.value()?;
                if meta.path.is_ident("name") {
                    names.push(value.parse::<LitStr>()?.value());
                } else {
                    value.parse::<Expr>()?;
                }
                Ok(())
            });
            for name in names.into_iter().filter(|n| is_native_network_lib(n)) {
                self.report(
                    attr.span(),
                    "native-network",
                    format!("links native networking library '{name}'"),
                );
            }
        }
        visit::visit_item_foreign_mod(self, item);
    }

    fn visit_foreign_item_fn(&mut self, item: &syn::ForeignItemFn) {
        let name = item.sig.ident.to_string();
        if is_native_network_function(&name) {
            self.report(
                item.sig.ident.span(),
                "native-network",
                format!("native networking function '{name}'"),
            );
        }
        self.check_name(&name, item.sig.ident.span());
        visit::visit_foreign_item_fn(self, item);
    }

    fn visit_path(&mut self, path: &syn::Path) {
        self.check_path(&segments(path), path.span(), "use");
        visit::visit_path(self, path);
    }

    fn visit_expr_path(&mut self, expr: &syn::ExprPath) {
        // Not called here (see visit_expr_call), so the URL it will be given is unknown.
        if let Some(last) = expr.path.segments.last() {
            let name = last.ident.to_string();
            if MACOS_URL_READ.is_match(&name) {
                self.report(
                    expr.span(),
                    "macos-network",
                    format!("'{name}' reads whatever its URL names"),
                );
            }
        }
        visit::visit_expr_path(self, expr);
    }

    fn visit_expr_call(&mut self, call: &syn::ExprCall) {
        match &*call.func {
            Expr::Path(func) => {
                self.visit_path(&func.path);
                self.check_call(&func.path, &call.args, func.span());
            }
            other => self.visit_expr(other),
        }
        for arg in &call.args {
            self.visit_expr(arg);
        }
    }

    fn visit_expr_method_call(&mut self, call: &syn::ExprMethodCall) {
        let name = call.method.to_string();
        let span = call.method.span();
        self.check_name(&name, span);
        if is_native_network_function(&name) {
            self.report(
                span,
                "native-network",
                format!("native networking function '{name}'"),
            );
        }
        if name == "save" && self.uses_image {
            self.report(
                span,
                "frame-write",
                "use of 'save' in a file that uses the image crate (frames must never be encoded)"
                    .into(),
            );
        } else if name == "save" && mentions_frame(&call.receiver) {
            self.report(
                span,
                "frame-write",
                "'save' of a frame (frames must never be written to disk)".into(),
            );
        }
        if (name == "write" || name == "write_all") && call.args.iter().any(mentions_frame) {
            self.report(
                span,
                "frame-write",
                format!("'{name}' of a frame (frames must never be written to disk)"),
            );
        }
        if MACOS_URL_READ.is_match(&name) {
            self.check_url_read(call.args.first(), &name, span);
        }
        if name == "arg" || name == "args" {
            self.check_wrapped_arguments(call);
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_array(&mut self, array: &syn::ExprArray) {
        self.check_command_sequence(array.elems.iter(), array.span());
        visit::visit_expr_array(self, array);
    }

    fn visit_expr_tuple(&mut self, tuple: &syn::ExprTuple) {
        self.check_command_sequence(tuple.elems.iter(), tuple.span());
        visit::visit_expr_tuple(self, tuple);
    }

    fn visit_macro(&mut self, mac: &syn::Macro) {
        match mac.parse_body_with(Punctuated::<Expr, syn::Token![,]>::parse_terminated) {
            Ok(exprs) => {
                if mac.path.is_ident("vec") {
                    self.check_command_sequence(exprs.iter(), mac.span());
                }
                for expr in &exprs {
                    self.visit_expr(expr);
                }
            }
            Err(_) => self.scan_tokens(mac.tokens.clone()),
        }
        visit::visit_macro(self, mac);
    }
}

// ------------------------------------------------------------------ public API (source)
/// Scan Rust source code; `path` is only used for reporting.
pub fn scan_source(source: &str, path: &Path) -> Vec<Violation> {
    match syn::parse_file(source) {
        Ok(file) => ModuleScanner::new(path, &file).run(&file),
        Err(error) => {
            let line = error.span().start().line;
            let message = format!("cannot be parsed, so it was not verified: {error}");
            vec![Violation {
                path: path.to_path_buf(),
                line,
                col: 1,
                rule: "syntax-error",
                message,
            }]
        }
    }
}

/// Scan one file (Rust sources are UTF-8; anything else cannot be verified).
pub fn scan_file(path: &Path) -> Vec<Violation> {
    let source = std::fs::read(path)
        .map_err(|e| e.to_string())
        .and_then(|bytes| String::from_utf8(bytes).map_err(|e| e.to_string()));
    match source {
        Ok(source) => scan_source(&source, path),
        Err(error) => vec![Violation {
            path: path.to_path_buf(),
            line: 0,
            col: 1,
            rule: "syntax-error",
            message: format!("cannot be parsed, so it was not verified: {error}"),
        }],
    }
}

/// Rust files under the given files/directories, in a stable order; test files
/// found in directories are left out (see the module documentation).
pub fn iter_rust_files(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for root in paths {
        if root.is_file() {
            files.push(root.clone());
            continue;
        }
        let walker = walkdir::WalkDir::new(root)
            .sort_by_file_name()
            .into_iter()
            .filter_entry(|entry| {
                entry.depth() == 0
                    || !entry.file_type().is_dir()
                    || !SKIP_DIRS.contains(&entry.file_name().to_string_lossy().as_ref())
            });
        for entry in walker.filter_map(Result::ok) {
            let name = entry.file_name().to_string_lossy();
            if entry.file_type().is_file()
                && name.ends_with(".rs")
                && !name.ends_with(TEST_FILE_SUFFIX)
            {
                files.push(entry.into_path());
            }
        }
    }
    files
}

/// The [`SOURCE_ALLOWLIST`] entry for exactly this file of the app, if any.
fn source_allowance(path: &Path, app_root: &Path) -> Option<&'static SourceAllowRule> {
    let (path, root) = (path.canonicalize().ok()?, app_root.canonicalize().ok()?);
    let relative = posix(path.strip_prefix(root).ok()?);
    SOURCE_ALLOWLIST.iter().find(|rule| rule.path == relative)
}

/// Scan every Rust file under `paths`; findings the allowlist covers for files of
/// the app at `app_root` are kept apart as notices.
pub fn scan_paths(paths: &[PathBuf], app_root: &Path) -> ScanResult {
    let mut result = ScanResult::default();
    for path in iter_rust_files(paths) {
        let allowance = source_allowance(&path, app_root);
        for violation in scan_file(&path) {
            match allowance {
                Some(rule) if rule.rules.contains(&violation.rule) => {
                    result.notices.push((violation, rule))
                }
                _ => result.violations.push(violation),
            }
        }
        result.files.push(path);
    }
    result
}

// ====================================================================== allow rules
/// Networking indicators a bundled binary (or a crate) may have, and why.
///
/// `pattern` is matched case-insensitively against the binary's path relative to
/// the bundle, from the right like Python's `PurePath.match` (`"eye-tracker*.exe"`
/// matches `bin/eye-tracker-gui.exe`), or against a crate's name. `indicators` are
/// glob patterns for the normalised indicator keys ([`network_indicators`]:
/// `"ws2_32"`, `"getaddrinfo"`...; for crates the networking crate's name with `_`
/// for `-`); nothing else is allowed for that binary or crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowRule {
    pub pattern: &'static str,
    pub indicators: &'static [&'static str],
    pub reason: &'static str,
}

impl AllowRule {
    pub fn matches(&self, relative: &str) -> bool {
        let (relative, pattern) = (relative.to_lowercase(), self.pattern.to_lowercase());
        let path: Vec<&str> = relative.split('/').filter(|p| !p.is_empty()).collect();
        let pattern: Vec<&str> = pattern.split('/').filter(|p| !p.is_empty()).collect();
        pattern.len() <= path.len()
            && path
                .iter()
                .rev()
                .zip(pattern.iter().rev())
                .all(|(name, pat)| fnmatch(name, pat))
    }

    pub fn allows(&self, key: &str) -> bool {
        self.indicators
            .iter()
            .any(|indicator| fnmatch(key, indicator))
    }
}

/// Shell-style match of one name against a pattern with `*` and `?` (case-sensitive).
fn fnmatch(name: &str, pattern: &str) -> bool {
    let (name, pattern): (Vec<char>, Vec<char>) =
        (name.chars().collect(), pattern.chars().collect());
    let (mut n, mut p) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while n < name.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == name[n]) {
            n += 1;
            p += 1;
        } else if p < pattern.len() && pattern[p] == '*' {
            star = Some((p, n));
            p += 1;
        } else if let Some((star_p, star_n)) = star {
            p = star_p + 1;
            n = star_n + 1;
            star = Some((star_p, star_n + 1));
        } else {
            return false;
        }
    }
    pattern[p..].iter().all(|&c| c == '*')
}

// ====================================================================== dependencies
const WHY_TOKIO: &str = "tokio is the async runtime of ksni (the Linux tray, its default 'tokio' \
     feature) and of zbus under it; their only connection is the D-Bus session bus, a local Unix \
     socket named by $DBUS_SESSION_BUS_ADDRESS. tokio's 'net' feature (socket2, mio) is how that \
     socket is polled; the source check keeps tokio::net out of the app's own code";
const WHY_WEBBROWSER: &str = "egui's open_url, the one helper the UI uses to open a link: it hands \
     the address to the user's default browser when the user clicks a link; the app itself \
     connects to nothing";

/// Every crate of the app's dependency tree that may depend on a networking crate,
/// and why that is acceptable. Entries are per crate and per networking crate, so a
/// new networking dependency still fails the check until someone reviews it here.
pub const DEPENDENCY_ALLOWLIST: &[AllowRule] = &[
    AllowRule {
        pattern: "tokio",
        indicators: &["mio", "socket2"],
        reason: WHY_TOKIO,
    },
    AllowRule {
        pattern: "egui-winit",
        indicators: &["webbrowser"],
        reason: WHY_WEBBROWSER,
    },
];

/// One crate of the app's dependency tree and the indexes of its normal dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrateNode {
    pub name: String,
    pub version: String,
    pub dependencies: Vec<usize>,
}

/// Outcome of scanning the dependency tree.
#[derive(Debug, Default)]
pub struct DependencyResult {
    /// Crates in the tree, the app included.
    pub crates: usize,
    pub violations: Vec<BundleFinding>,
    pub allowed: Vec<BundleFinding>,
}

impl DependencyResult {
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }
}

/// The output of `cargo metadata` for the workspace (every platform).
pub fn load_metadata(workspace: &Path) -> Result<Value, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(workspace.join("Cargo.toml"))
        .output()
        .map_err(|e| format!("cannot run cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| format!("cargo metadata: {e}"))
}

/// The app package and every crate it reaches through normal (not build or dev)
/// dependencies, on every platform; the app comes first.
pub fn app_dependencies(metadata: &Value) -> Result<Vec<CrateNode>, String> {
    let members: HashSet<&str> = metadata["workspace_members"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect();
    let mut packages = HashMap::new();
    for package in metadata["packages"].as_array().into_iter().flatten() {
        let (Some(id), Some(name), Some(version)) = (
            package["id"].as_str(),
            package["name"].as_str(),
            package["version"].as_str(),
        ) else {
            continue;
        };
        packages.insert(id, (name, version));
    }
    let root = packages
        .iter()
        .find(|(id, (name, _))| *name == APP_PACKAGE && members.contains(*id))
        .map(|(id, _)| *id)
        .ok_or_else(|| format!("no package '{APP_PACKAGE}' in the workspace"))?;
    let resolve: HashMap<&str, &Value> = metadata["resolve"]["nodes"]
        .as_array()
        .ok_or("cargo metadata has no resolved dependency graph")?
        .iter()
        .filter_map(|node| Some((node["id"].as_str()?, node)))
        .collect();

    let mut index: HashMap<&str, usize> = HashMap::from([(root, 0)]);
    let mut order = vec![root];
    let mut edges: Vec<Vec<usize>> = Vec::new();
    let mut next = 0;
    while next < order.len() {
        let id = order[next];
        let node = resolve
            .get(id)
            .ok_or_else(|| format!("{id} is not in the resolved graph"))?;
        let mut targets = Vec::new();
        for dep in node["deps"].as_array().into_iter().flatten() {
            let normal = dep["dep_kinds"]
                .as_array()
                .into_iter()
                .flatten()
                .any(|k| k["kind"].is_null());
            let Some(target) = dep["pkg"].as_str().filter(|_| normal) else {
                continue;
            };
            let position = *index.entry(target).or_insert_with(|| {
                order.push(target);
                order.len() - 1
            });
            if !targets.contains(&position) {
                targets.push(position);
            }
        }
        edges.push(targets);
        next += 1;
    }
    order
        .iter()
        .zip(edges)
        .map(|(id, dependencies)| {
            let (name, version) = packages
                .get(id)
                .ok_or_else(|| format!("unknown package {id}"))?;
            Ok(CrateNode {
                name: name.to_string(),
                version: version.to_string(),
                dependencies,
            })
        })
        .collect()
}

fn crate_key(name: &str) -> String {
    name.replace('-', "_").to_lowercase()
}

/// Every dependency on a networking crate, each allowed or not by `allowlist`.
pub fn scan_dependencies(crates: &[CrateNode], allowlist: &[AllowRule]) -> DependencyResult {
    let mut result = DependencyResult {
        crates: crates.len(),
        ..Default::default()
    };
    for package in crates {
        for dependency in package.dependencies.iter().map(|&i| &crates[i]) {
            let key = crate_key(&dependency.name);
            if !NETWORK_CRATES.contains(&key.as_str()) {
                continue;
            }
            let allowed_by = allowlist
                .iter()
                .find(|r| r.matches(&package.name) && r.allows(&key))
                .copied();
            let finding = BundleFinding {
                relative: format!("{} {}", package.name, package.version),
                rule: "network-crate",
                detail: format!("needs {} {}", dependency.name, dependency.version),
                allowed_by,
            };
            if allowed_by.is_some() {
                &mut result.allowed
            } else {
                &mut result.violations
            }
            .push(finding);
        }
    }
    result
}

// ====================================================================== built bundles
/// Byte strings that identify usage logging / telemetry clients. Matched
/// case-insensitively, as ASCII and as UTF-16LE, in every file of the bundle.
pub const TELEMETRY_MARKERS: &[&str] = &[
    "play.googleapis.com", // Google Clearcut log endpoint (MediaPipe's usage logger)
    "clearcut",
    "firebaselogging",
    "crashlyticsreports",
    "app-measurement.com",
    "google-analytics.com",
    "googletagmanager.com",
    "ingest.sentry.io",
    "api.segment.io",
    "api.mixpanel.com",
    "api2.amplitude.com",
    "i.posthog.com",
    "app.posthog.com",
    "sessions.bugsnag.com",
    "api.rollbar.com",
    "dc.services.visualstudio.com",
];

/// Imported functions (ELF / Mach-O) that resolve host names or open remote
/// connections. Plain `socket`/`connect` are omitted on purpose: they are also how
/// local (`AF_UNIX`) IPC works. On macOS, Foundation does HTTP(S) without CFNetwork
/// being linked: a binary that uses NSURLSession (or the older NSURLConnection and
/// NSURLDownload) imports its class symbol, stored here, like every symbol, without
/// the leading underscore.
pub const NETWORK_SYMBOLS: &[&str] = &[
    "CFHostStartInfoResolution",
    "CFReadStreamCreateForHTTPRequest",
    "CFReadStreamCreateForStreamedHTTPRequest",
    "CFSocketConnectToAddress",
    "CFStreamCreatePairWithSocketToHost",
    "OBJC_CLASS_$_NSURLConnection",
    "OBJC_CLASS_$_NSURLDownload",
    "OBJC_CLASS_$_NSURLSession",
    "OBJC_CLASS_$_NSURLSessionConfiguration",
    "DNSServiceGetAddrInfo",
    "DNSServiceQueryRecord",
    "gethostbyaddr",
    "gethostbyaddr_r",
    "gethostbyname",
    "gethostbyname2",
    "gethostbyname2_r",
    "gethostbyname_r",
    "getaddrinfo",
    "getaddrinfo_a",
    "getnameinfo",
    "nw_connection_create",
    "res_nquery",
    "res_nsearch",
    "res_query",
    "res_search",
    "__res_query",
    "__res_search",
];

const RESOLVER: &[&str] = &["getaddrinfo", "getnameinfo", "gethostby*"];
const WHY_UPDATE_CHECK: &str = "WinHTTP is the opt-in update check (update/winhttp.rs, the source \
     check's one exception): one HTTPS GET a day to GitHub's release address, off by default";
const WHY_RUST_STD: &str = "Rust's standard library links Winsock for std::net and calls \
     WSACleanup at exit; the app's own code opens no socket (enforced by the source check)";
const WHY_RESOLVER: &str = "std::net's resolver, reached through x11rb (a TCP connection only when \
     $DISPLAY names another host) and zbus (the D-Bus session bus, a local Unix socket named by \
     $DBUS_SESSION_BUS_ADDRESS); the app's own code resolves no host (enforced by the source check)";

/// Every bundled binary that may link networking code, and why that is acceptable.
/// Entries are per file and per indicator, so a new networking dependency of an
/// allowed binary still fails the check until someone reviews it here.
pub const BUNDLE_ALLOWLIST: &[AllowRule] = &[
    AllowRule {
        pattern: "eye-tracker*.exe",
        indicators: &["winhttp"],
        reason: WHY_UPDATE_CHECK,
    },
    AllowRule {
        pattern: "eye-tracker*.exe",
        indicators: &["ws2_32"],
        reason: WHY_RUST_STD,
    },
    AllowRule {
        pattern: "eye-tracker*",
        indicators: RESOLVER,
        reason: WHY_RESOLVER,
    },
];

/// What a native binary imports: libraries and (ELF/Mach-O) undefined symbols.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeBinary {
    /// `"pe"`, `"elf"` or `"macho"`.
    pub format: &'static str,
    pub libraries: Vec<String>,
    pub symbols: BTreeSet<String>,
}

/// A file looked like a native binary but its headers could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryFormatError(pub String);

impl fmt::Display for BinaryFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A networking library or symbol that a native binary imports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Indicator {
    /// `"network-library"` or `"network-symbol"`.
    pub rule: &'static str,
    /// As written in the binary (`"WS2_32.dll"`, `"getaddrinfo"`).
    pub name: String,
    /// Normalised for the allow-list (`"ws2_32"`, `"getaddrinfo"`).
    pub key: String,
}

/// A privacy-relevant finding in one bundled file (or one crate of the tree).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleFinding {
    pub relative: String,
    pub rule: &'static str,
    pub detail: String,
    pub allowed_by: Option<AllowRule>,
}

impl fmt::Display for BundleFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}: {}", self.relative, self.rule, self.detail)
    }
}

/// Outcome of scanning a built bundle.
#[derive(Debug, Default)]
pub struct BundleResult {
    pub files: usize,
    pub binaries: usize,
    pub violations: Vec<BundleFinding>,
    pub allowed: Vec<BundleFinding>,
}

impl BundleResult {
    pub fn ok(&self) -> bool {
        self.violations.is_empty()
    }
}

// ----------------------------------------------------------------- binary readers
/// `"pe"`, `"elf"`, `"macho"` from the first bytes of a file, else `None`.
pub fn binary_format(head: &[u8]) -> Option<&'static str> {
    let magic = head.get(..4);
    if magic == Some(b"\x7fELF") {
        return Some("elf");
    }
    if head.starts_with(b"MZ") {
        return Some("pe");
    }
    if matches!(
        magic,
        Some(b"\xfe\xed\xfa\xce" | b"\xce\xfa\xed\xfe" | b"\xfe\xed\xfa\xcf" | b"\xcf\xfa\xed\xfe")
    ) {
        return Some("macho");
    }
    // Universal binaries share CAFEBABE with Java class files; the architecture
    // count is small, whereas a class file has its (>= 45) version number there.
    if matches!(magic, Some(b"\xca\xfe\xba\xbe" | b"\xca\xfe\xba\xbf")) && head.len() >= 8 {
        let count = u32::from_be_bytes([head[4], head[5], head[6], head[7]]);
        if 0 < count && count < 20 {
            return Some("macho");
        }
    }
    None
}

fn format_error(kind: &str) -> impl Fn(object::read::Error) -> BinaryFormatError + '_ {
    move |error| BinaryFormatError(format!("unreadable {kind} file: {error}"))
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// DLL names imported by a PE file (normal and delay-load import tables).
pub fn parse_pe(data: &[u8]) -> Result<NativeBinary, BinaryFormatError> {
    let truncated = || BinaryFormatError("truncated PE file".into());
    let offset = data.get(0x3C..0x40).ok_or_else(truncated)?;
    let pe_offset = u32::from_le_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize;
    if data.get(pe_offset..pe_offset.saturating_add(4)) != Some(b"PE\0\0") {
        return Err(BinaryFormatError("MZ file without a PE header".into()));
    }
    let magic = data
        .get(pe_offset + 24..pe_offset + 26)
        .ok_or_else(truncated)?;
    match u16::from_le_bytes([magic[0], magic[1]]) {
        0x10B => pe_libraries(&PeFile32::parse(data).map_err(format_error("PE"))?),
        0x20B => pe_libraries(&PeFile64::parse(data).map_err(format_error("PE"))?),
        magic => Err(BinaryFormatError(format!(
            "unknown PE optional header magic {magic:#x}"
        ))),
    }
}

fn pe_libraries<Pe: ImageNtHeaders>(
    file: &object::read::pe::PeFile<'_, Pe>,
) -> Result<NativeBinary, BinaryFormatError> {
    let error = format_error("PE");
    let mut libraries = Vec::new();
    if let Some(table) = file.import_table().map_err(&error)? {
        let mut descriptors = table.descriptors().map_err(&error)?;
        while let Some(descriptor) = descriptors.next().map_err(&error)? {
            libraries.push(text(table.name(descriptor.name.get(LE)).map_err(&error)?));
        }
    }
    if let Some(table) = file.delay_load_import_table().map_err(&error)? {
        let image_base = file.nt_headers().optional_header().image_base();
        let mut descriptors = table.descriptors().map_err(&error)?;
        while let Some(descriptor) = descriptors.next().map_err(&error)? {
            let name = descriptor.dll_name_rva.get(LE);
            // Old (VC6) descriptors hold virtual addresses instead of RVAs.
            let rva = if descriptor.attributes.get(LE) & 1 != 0 {
                name
            } else {
                name.wrapping_sub(image_base as u32)
            };
            libraries.push(text(table.name(rva).map_err(&error)?));
        }
    }
    Ok(NativeBinary {
        format: "pe",
        libraries,
        symbols: BTreeSet::new(),
    })
}

/// `DT_NEEDED` libraries and undefined dynamic symbols of an ELF file.
pub fn parse_elf(data: &[u8]) -> Result<NativeBinary, BinaryFormatError> {
    match data.get(4) {
        Some(1) => elf_imports::<elf::FileHeader32<object::Endianness>>(data),
        Some(2) => elf_imports::<elf::FileHeader64<object::Endianness>>(data),
        _ => Err(BinaryFormatError("unknown ELF class or byte order".into())),
    }
}

fn elf_imports<Elf: FileHeader<Endian = object::Endianness>>(
    data: &[u8],
) -> Result<NativeBinary, BinaryFormatError> {
    let error = format_error("ELF");
    let header = Elf::parse(data).map_err(&error)?;
    let endian = header.endian().map_err(&error)?;
    // Section names are not needed, so a missing e_shstrndx is no reason to give up.
    let headers = header.section_headers(endian, data).map_err(&error)?;
    let sections = object::read::elf::SectionTable::<Elf, &[u8]>::new(headers, Default::default());
    let mut libraries = Vec::new();
    let mut symbols = BTreeSet::new();
    for (index, section) in sections.enumerate() {
        if let Some((entries, link)) = section.dynamic(endian, data).map_err(&error)? {
            let strings = sections.strings(endian, data, link).map_err(&error)?;
            for entry in entries {
                match entry.tag(endian) {
                    elf::DT_NULL => break,
                    elf::DT_NEEDED => {
                        libraries.push(text(entry.string(endian, strings).map_err(&error)?))
                    }
                    _ => {}
                }
            }
        } else if section.sh_type(endian) == elf::SHT_DYNSYM {
            let table = sections
                .symbol_table_by_index(endian, data, index)
                .map_err(&error)?;
            for symbol in table.symbols().iter().skip(1) {
                // Undefined GLOBAL or WEAK symbols: what the binary imports.
                if symbol.st_name(endian) != 0
                    && symbol.st_shndx(endian) == elf::SHN_UNDEF
                    && matches!(symbol.st_bind(), elf::STB_GLOBAL | elf::STB_WEAK)
                {
                    symbols.insert(text(symbol.name(endian, table.strings()).map_err(&error)?));
                }
            }
        }
    }
    Ok(NativeBinary {
        format: "elf",
        libraries,
        symbols,
    })
}

/// Linked dylibs/frameworks and undefined symbols of a (universal) Mach-O file.
pub fn parse_macho(data: &[u8]) -> Result<NativeBinary, BinaryFormatError> {
    let slices: Vec<&[u8]> = match data.get(..4) {
        Some(b"\xca\xfe\xba\xbe") => fat_slices(
            data,
            MachOFatFile32::parse(data)
                .map_err(format_error("Mach-O"))?
                .arches(),
        )?,
        Some(b"\xca\xfe\xba\xbf") => fat_slices(
            data,
            MachOFatFile64::parse(data)
                .map_err(format_error("Mach-O"))?
                .arches(),
        )?,
        _ => vec![data],
    };
    let mut binary = NativeBinary {
        format: "macho",
        libraries: Vec::new(),
        symbols: BTreeSet::new(),
    };
    for slice in slices {
        let thin = match slice.get(..4) {
            Some(b"\xcf\xfa\xed\xfe" | b"\xfe\xed\xfa\xcf") => {
                macho_imports::<macho::MachHeader64<object::Endianness>>(slice)?
            }
            Some(b"\xce\xfa\xed\xfe" | b"\xfe\xed\xfa\xce") => {
                macho_imports::<macho::MachHeader32<object::Endianness>>(slice)?
            }
            _ => return Err(BinaryFormatError("not a Mach-O slice".into())),
        };
        for library in thin.libraries {
            if !binary.libraries.contains(&library) {
                binary.libraries.push(library);
            }
        }
        binary.symbols.extend(thin.symbols);
    }
    Ok(binary)
}

fn fat_slices<'d, Arch: FatArch>(
    data: &'d [u8],
    arches: &[Arch],
) -> Result<Vec<&'d [u8]>, BinaryFormatError> {
    arches
        .iter()
        .map(|arch| arch.data(data).map_err(format_error("Mach-O")))
        .collect()
}

fn macho_imports<Mach: MachHeader<Endian = object::Endianness>>(
    data: &[u8],
) -> Result<NativeBinary, BinaryFormatError> {
    let error = format_error("Mach-O");
    let header = Mach::parse(data, 0).map_err(&error)?;
    let endian = header.endian().map_err(&error)?;
    let mut commands = header.load_commands(endian, data, 0).map_err(&error)?;
    let mut libraries = Vec::new();
    let mut symbols = BTreeSet::new();
    while let Some(command) = commands.next().map_err(&error)? {
        if let Some(dylib) = command.dylib().map_err(&error)? {
            libraries.push(text(
                command.string(endian, dylib.dylib.name).map_err(&error)?,
            ));
        } else if let Some(symtab) = command.symtab().map_err(&error)? {
            let table = symtab.symbols::<Mach, _>(endian, data).map_err(&error)?;
            for nlist in table.iter() {
                let kind = nlist.n_type();
                // Undefined external, not a debugging (stab) entry.
                if nlist.n_strx(endian) != 0
                    && !kind.is_stab()
                    && kind.typ() == macho::N_UNDF
                    && kind.is_ext()
                {
                    let name = text(nlist.name(endian, table.strings()).map_err(&error)?);
                    symbols.insert(name.strip_prefix('_').map(str::to_string).unwrap_or(name));
                }
            }
        }
    }
    Ok(NativeBinary {
        format: "macho",
        libraries,
        symbols,
    })
}

/// Parse a native binary; `Ok(None)` if `data` is not one.
pub fn parse_binary(data: &[u8]) -> Result<Option<NativeBinary>, BinaryFormatError> {
    match binary_format(&data[..data.len().min(8)]) {
        Some("pe") => parse_pe(data).map(Some),
        Some("elf") => parse_elf(data).map(Some),
        Some(_) => parse_macho(data).map(Some),
        None => Ok(None),
    }
}

// ------------------------------------------------------------------ bundle scanning
static MARKERS: LazyLock<RegexSet> = LazyLock::new(|| {
    let patterns = TELEMETRY_MARKERS.iter().map(|marker| {
        let wide: String = marker
            .chars()
            .map(|c| format!("{}\\x00", regex::escape(&c.to_string())))
            .collect();
        format!("(?i-u:{}|{wide})", regex::escape(marker))
    });
    RegexSet::new(patterns).unwrap()
});

/// Telemetry markers found in `data`, case-insensitively, as ASCII or UTF-16LE.
pub fn telemetry_markers(data: &[u8]) -> Vec<&'static str> {
    MARKERS
        .matches(data)
        .into_iter()
        .map(|index| TELEMETRY_MARKERS[index])
        .collect()
}

/// Normalise a linked library name for the allow-list.
///
/// `WS2_32.dll` → `ws2_32`, `libssl.so.3` → `libssl`,
/// `/usr/lib/libresolv.9.dylib` → `libresolv`, `.../CFNetwork` → `cfnetwork`.
pub fn library_key(library: &str) -> String {
    static VERSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\.\d+)+$").unwrap());
    let library = library.replace('\\', "/");
    let name = library
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .to_lowercase();
    let name = name.strip_suffix(".dll").unwrap_or(&name);
    let name = name.strip_suffix(".dylib").unwrap_or(name);
    let name = name.split(".so").next().unwrap_or_default();
    VERSION.replace(name, "").into_owned()
}

fn is_network_framework(library: &str) -> bool {
    let path = library.replace('\\', "/").to_lowercase();
    MACOS_NETWORK_FRAMEWORKS
        .iter()
        .any(|name| path.contains(&format!("/{}.framework/", name.to_lowercase())))
}

/// Every networking library or symbol a native binary imports.
pub fn network_indicators(binary: &NativeBinary) -> Vec<Indicator> {
    let mut found = Vec::new();
    for library in &binary.libraries {
        let key = library_key(library);
        let flagged = if binary.format == "pe" {
            WINDOWS_NETWORK_DLLS.contains(&key.as_str())
        } else {
            POSIX_NETWORK_LIBRARIES
                .iter()
                .any(|prefix| key.starts_with(prefix))
                || is_network_framework(library)
        };
        if flagged {
            found.push(Indicator {
                rule: "network-library",
                name: library.clone(),
                key,
            });
        }
    }
    found.extend(
        binary
            .symbols
            .iter()
            .filter(|s| NETWORK_SYMBOLS.contains(&s.as_str()))
            .map(|symbol| Indicator {
                rule: "network-symbol",
                name: symbol.clone(),
                key: symbol.to_lowercase(),
            }),
    );
    found
}

/// Findings for one bundled file, and whether it is a native binary.
pub fn scan_bundle_file(
    data: &[u8],
    relative: &str,
    allowlist: &[AllowRule],
) -> (Vec<BundleFinding>, bool) {
    let finding = |rule, detail, allowed_by| BundleFinding {
        relative: relative.to_string(),
        rule,
        detail,
        allowed_by,
    };
    let mut findings: Vec<BundleFinding> = telemetry_markers(data)
        .into_iter()
        .map(|marker| {
            finding(
                "telemetry-endpoint",
                format!("contains the telemetry marker '{marker}'"),
                None,
            )
        })
        .collect();
    let binary = match parse_binary(data) {
        Ok(Some(binary)) => binary,
        Ok(None) => return (findings, false),
        Err(error) => {
            findings.push(finding(
                "unreadable-binary",
                format!("cannot be verified: {error}"),
                None,
            ));
            return (findings, true);
        }
    };
    for indicator in network_indicators(&binary) {
        let rule = allowlist
            .iter()
            .find(|r| r.matches(relative) && r.allows(&indicator.key))
            .copied();
        let verb = if indicator.rule == "network-library" {
            "links"
        } else {
            "imports"
        };
        findings.push(finding(
            indicator.rule,
            format!("{verb} {}", indicator.name),
            rule,
        ));
    }
    (findings, true)
}

/// Scan a built bundle (a folder, a `.app`, or a single file).
pub fn scan_bundle(root: &Path, allowlist: &[AllowRule]) -> BundleResult {
    let mut result = BundleResult::default();
    let (base, files): (PathBuf, Vec<PathBuf>) = if root.is_file() {
        (
            root.parent().map(Path::to_path_buf).unwrap_or_default(),
            vec![root.to_path_buf()],
        )
    } else {
        // Symlinks (framework "Current" links, soname aliases) point at files that
        // are scanned themselves.
        let walker = walkdir::WalkDir::new(root)
            .sort_by_file_name()
            .into_iter()
            .filter_map(Result::ok);
        (
            root.to_path_buf(),
            walker
                .filter(|e| e.file_type().is_file())
                .map(|e| e.into_path())
                .collect(),
        )
    };
    for path in files {
        let relative = posix(path.strip_prefix(&base).unwrap_or(&path));
        result.files += 1;
        let data = match std::fs::read(&path) {
            Ok(data) => data,
            Err(error) => {
                let detail = format!("cannot be read: {error}");
                result.violations.push(BundleFinding {
                    relative,
                    rule: "unreadable-binary",
                    detail,
                    allowed_by: None,
                });
                continue;
            }
        };
        let (findings, is_binary) = scan_bundle_file(&data, &relative, allowlist);
        result.binaries += usize::from(is_binary);
        for finding in findings {
            if finding.allowed_by.is_some() {
                &mut result.allowed
            } else {
                &mut result.violations
            }
            .push(finding);
        }
    }
    result
}

// ------------------------------------------------------------------------ command line
/// Fail if the app's sources contain network or frame-writing code or its
/// dependencies pull in a networking crate, or (--bundle) if a built bundle contains
/// telemetry or unexpected networking libraries.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Files or directories to scan (default: crates/eye-tracker/src and its dependencies).
    pub paths: Vec<PathBuf>,
    /// Scan a built bundle (a folder, a .app, or a single file) instead of Rust sources.
    #[arg(long, value_name = "DIR", conflicts_with = "paths")]
    pub bundle: Option<PathBuf>,
    /// Print violations only.
    #[arg(short, long)]
    pub quiet: bool,
}

/// What a run prints, and its exit status.
#[derive(Debug, Default)]
pub struct Report {
    pub code: i32,
    pub out: String,
    pub err: String,
}

impl Report {
    fn out(&mut self, line: impl AsRef<str>) {
        self.out.push_str(line.as_ref());
        self.out.push('\n');
    }

    fn err(&mut self, line: impl AsRef<str>) {
        self.err.push_str(line.as_ref());
        self.err.push('\n');
    }

    fn done(mut self, code: i32) -> Self {
        self.code = code;
        self
    }
}

fn github_annotation(violation: &Violation) -> String {
    let message = format!("{}: {}", violation.rule, violation.message)
        .replace('%', "%25")
        .replace('\n', "%0A");
    format!(
        "::error file={},line={},col={},title=Privacy check::{message}",
        display_path(&violation.path),
        violation.line,
        violation.col
    )
}

fn escape(text: &str) -> String {
    text.replace('%', "%25")
        .replace('\r', "%0D")
        .replace('\n', "%0A")
}

/// One line per allow-listed file (or crate): what it links and why that is accepted.
fn allowed_summary(findings: &[BundleFinding]) -> Vec<String> {
    let mut grouped: Vec<((&str, &str), Vec<&str>)> = Vec::new();
    for finding in findings {
        let reason = finding
            .allowed_by
            .map(|rule| rule.reason)
            .unwrap_or_default();
        // "links WS2_32.dll" -> "WS2_32.dll"
        let what = finding
            .detail
            .split_once(' ')
            .map_or(finding.detail.as_str(), |(_, rest)| rest);
        let key = (finding.relative.as_str(), reason);
        match grouped.iter_mut().find(|(k, _)| *k == key) {
            Some((_, names)) => names.push(what),
            None => grouped.push((key, vec![what])),
        }
    }
    grouped
        .into_iter()
        .map(|((relative, reason), names)| {
            format!("{relative}: allowed {}: {reason}", names.join(", "))
        })
        .collect()
}

fn main_bundle(root: &Path, quiet: bool, annotate: bool) -> Report {
    let mut report = Report::default();
    let result = scan_bundle(root, BUNDLE_ALLOWLIST);
    if !quiet {
        for line in allowed_summary(&result.allowed) {
            report.out(if annotate {
                format!("::notice title=Privacy check (bundle)::{}", escape(&line))
            } else {
                line
            });
        }
    }
    for finding in &result.violations {
        let text = finding.to_string();
        report.out(if annotate {
            format!("::error title=Privacy check (bundle)::{}", escape(&text))
        } else {
            text
        });
    }
    let shown = display_path(root);
    if result.ok() {
        if !quiet {
            report.out(format!(
                "check_privacy: OK: {} native binaries in {} files of {shown}; \
                 no telemetry, and networking only where allow-listed.",
                result.binaries, result.files
            ));
        }
        return report.done(0);
    }
    report.err(format!(
        "check_privacy: FAILED: {} problem(s) in the bundle {shown}. Review each one; a library \
         that legitimately links networking code needs an entry with a reason in BUNDLE_ALLOWLIST \
         (crates/xtask/src/check_privacy.rs).",
        result.violations.len()
    ));
    report.done(1)
}

/// Run the gate; `github_actions` is the value of `GITHUB_ACTIONS` (annotations
/// replace plain lines when it is `"true"`).
pub fn run(args: &Args, github_actions: Option<&str>) -> Report {
    let annotate = github_actions == Some("true");
    if let Some(bundle) = &args.bundle {
        if !bundle.exists() {
            let mut report = Report::default();
            report.err(format!(
                "check_privacy: no such file or directory: {}",
                bundle.display()
            ));
            return report.done(2);
        }
        return main_bundle(bundle, args.quiet, annotate);
    }

    let mut report = Report::default();
    let app_root = default_target();
    let with_dependencies = args.paths.is_empty();
    let targets = if with_dependencies {
        vec![app_root.clone()]
    } else {
        args.paths.clone()
    };
    let missing: Vec<&PathBuf> = targets.iter().filter(|p| !p.exists()).collect();
    if !missing.is_empty() {
        for path in missing {
            report.err(format!(
                "check_privacy: no such file or directory: {}",
                path.display()
            ));
        }
        return report.done(2);
    }

    let result = scan_paths(&targets, &app_root);
    for rule in SOURCE_ALLOWLIST {
        let found: Vec<&Violation> = result
            .notices
            .iter()
            .filter(|(_, r)| r.path == rule.path)
            .map(|(v, _)| v)
            .collect();
        if !found.is_empty() {
            let kinds: BTreeSet<&str> = found.iter().map(|v| v.rule).collect();
            let kinds = kinds.into_iter().collect::<Vec<_>>().join(", ");
            let line = format!(
                "allowed: {}: {} {kinds} finding(s): {}",
                rule.path,
                found.len(),
                rule.reason
            );
            report.out(if annotate {
                format!("::notice title=Privacy check (source)::{}", escape(&line))
            } else {
                line
            });
        }
    }
    for violation in &result.violations {
        report.out(if annotate {
            github_annotation(violation)
        } else {
            violation.to_string()
        });
    }

    let dependencies = with_dependencies.then(|| {
        load_metadata(&repo_root())
            .and_then(|metadata| app_dependencies(&metadata))
            .map(|crates| scan_dependencies(&crates, DEPENDENCY_ALLOWLIST))
    });
    let mut failed = !result.ok();
    match &dependencies {
        Some(Ok(deps)) => {
            if !args.quiet {
                for line in allowed_summary(&deps.allowed) {
                    report.out(if annotate {
                        format!(
                            "::notice title=Privacy check (dependencies)::{}",
                            escape(&line)
                        )
                    } else {
                        line
                    });
                }
            }
            for finding in &deps.violations {
                let text = finding.to_string();
                report.out(if annotate {
                    format!(
                        "::error title=Privacy check (dependencies)::{}",
                        escape(&text)
                    )
                } else {
                    text
                });
            }
            failed |= !deps.ok();
        }
        Some(Err(error)) => {
            report.err(format!(
                "check_privacy: the dependency tree cannot be read, so it was not verified: {error}"
            ));
            failed = true;
        }
        None => {}
    }

    let shown = targets
        .iter()
        .map(|p| display_path(p))
        .collect::<Vec<_>>()
        .join(", ");
    if !failed {
        if !args.quiet {
            report.out(match &dependencies {
                Some(Ok(deps)) => format!(
                    "check_privacy: OK: {} files in {shown} and {} crates it depends on; no network or \
                     frame-writing code found, and networking crates only where allow-listed.",
                    result.files.len(),
                    deps.crates - 1
                ),
                _ => format!(
                    "check_privacy: OK: {} files in {shown}; no network or frame-writing code found.",
                    result.files.len()
                ),
            });
        }
        return report.done(0);
    }
    if !result.ok() {
        let bad_files: BTreeSet<&PathBuf> = result.violations.iter().map(|v| &v.path).collect();
        report.err(format!(
            "check_privacy: FAILED: {} violation(s) in {} of {} files. Eye Tracker must not use the \
             network or write frames.",
            result.violations.len(),
            bad_files.len(),
            result.files.len()
        ));
    }
    if let Some(Ok(deps)) = &dependencies
        && !deps.ok()
    {
        report.err(format!(
                "check_privacy: FAILED: {} networking crate dependency(ies) in the tree of {APP_PACKAGE}. \
                 Review each one; a crate that legitimately needs one needs an entry with a reason in \
                 DEPENDENCY_ALLOWLIST (crates/xtask/src/check_privacy.rs).",
                deps.violations.len()
            ));
    }
    report.done(1)
}

/// `cargo xtask check-privacy`: prints the report and returns the exit status.
pub fn main(args: &Args) -> i32 {
    let report = run(args, std::env::var("GITHUB_ACTIONS").ok().as_deref());
    print!("{}", report.out);
    eprint!("{}", report.err);
    report.code
}

#[cfg(test)]
#[path = "check_privacy_tests.rs"]
mod tests;
