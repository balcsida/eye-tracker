//! Tests for `cargo xtask check-privacy`, the "no network, no frames on disk" guard.
//!
//! It checks the app's sources, its dependency tree and (`--bundle`) the built
//! output; the bundle tests use small synthetic PE, ELF and Mach-O files built here.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::json;
use tempfile::TempDir;

use super::*;

fn rules(source: &str) -> Vec<&'static str> {
    scan_source(source, Path::new("snippet.rs"))
        .into_iter()
        .map(|v| v.rule)
        .collect()
}

fn source_args(paths: &[&Path]) -> Args {
    Args {
        paths: paths.iter().map(|p| p.to_path_buf()).collect(),
        bundle: None,
        quiet: false,
    }
}

fn bundle_args(root: &Path, quiet: bool) -> Args {
    Args {
        paths: Vec::new(),
        bundle: Some(root.to_path_buf()),
        quiet,
    }
}

// ---------------------------------------------------------------------------- the real tree
#[test]
fn test_source_tree_is_clean() {
    let result = scan_paths(&[default_target()], &default_target());
    assert!(
        result.files.len() > 10,
        "the scan should cover the whole package"
    );
    let found: Vec<String> = result.violations.iter().map(Violation::to_string).collect();
    assert!(result.ok(), "privacy violations:\n{}", found.join("\n"));
}

// ------------------------------------------------------------- the one reviewed exception
const WINHTTP_LOAD: &str = "use windows::Win32::Networking::WinHttp::WinHttpOpen;\n\
                            fn open() {\n    unsafe { WinHttpOpen(None, 0, None, None, 0) };\n}\n";

/// A stand-in for crates/eye-tracker/src, so the allowlist can be tried on any file.
fn app_tree(dir: &TempDir) -> PathBuf {
    let root = dir.path().join("src");
    fs::create_dir_all(root.join("update")).unwrap();
    root
}

#[test]
fn test_the_only_exception_is_the_update_connection() {
    let found: Vec<_> = SOURCE_ALLOWLIST
        .iter()
        .map(|rule| (rule.path, rule.rules))
        .collect();
    assert_eq!(found, [("update/winhttp.rs", &["native-network"][..])]);
}

#[test]
fn test_the_real_exception_is_reported_not_hidden() {
    let result = scan_paths(&[default_target()], &default_target());
    assert!(result.ok());
    let paths: BTreeSet<&str> = result.notices.iter().map(|(_, rule)| rule.path).collect();
    assert_eq!(paths, BTreeSet::from(["update/winhttp.rs"]));
    let kinds: BTreeSet<&str> = result.notices.iter().map(|(v, _)| v.rule).collect();
    assert_eq!(kinds, BTreeSet::from(["native-network"]));
}

#[test]
fn test_the_exception_is_printed_on_every_run() {
    let report = run(&source_args(&[]), None);
    assert_eq!(report.code, 0, "{}{}", report.out, report.err);
    assert!(report.out.contains("allowed: update/winhttp.rs"));
    assert!(report.out.contains("update check"));
}

#[test]
fn test_winhttp_is_allowed_in_that_file_only() {
    let dir = TempDir::new().unwrap();
    let root = app_tree(&dir);
    fs::write(root.join("update").join("winhttp.rs"), WINHTTP_LOAD).unwrap();
    fs::write(root.join("update").join("other.rs"), WINHTTP_LOAD).unwrap();
    fs::write(root.join("winhttp.rs"), WINHTTP_LOAD).unwrap();
    let result = scan_paths(std::slice::from_ref(&root), &root);
    let flagged: BTreeSet<String> = result
        .violations
        .iter()
        .map(|v| posix(v.path.strip_prefix(&root).unwrap()))
        .collect();
    assert_eq!(
        flagged,
        BTreeSet::from(["update/other.rs".into(), "winhttp.rs".into()])
    );
    let names: BTreeSet<String> = result
        .notices
        .iter()
        .map(|(v, _)| v.path.file_name().unwrap().to_string_lossy().into())
        .collect();
    assert_eq!(names, BTreeSet::from(["winhttp.rs".into()]));
}

#[test]
fn test_a_copy_outside_the_app_is_not_excused() {
    let dir = TempDir::new().unwrap();
    let root = app_tree(&dir);
    let elsewhere = dir.path().join("update");
    fs::create_dir(&elsewhere).unwrap();
    fs::write(elsewhere.join("winhttp.rs"), WINHTTP_LOAD).unwrap();
    let result = scan_paths(&[elsewhere], &root);
    assert!(!result.ok());
    assert!(result.notices.is_empty());
}

#[test]
fn test_the_excused_file_still_answers_to_every_other_rule() {
    let dir = TempDir::new().unwrap();
    let root = app_tree(&dir);
    let source = format!(
        "{WINHTTP_LOAD}use std::net;\nuse ureq;\n\
         fn keep(frame: &Frame) {{ std::fs::write(\"a.raw\", &frame.data).unwrap(); }}\n"
    );
    fs::write(root.join("update").join("winhttp.rs"), source).unwrap();
    let result = scan_paths(std::slice::from_ref(&root), &root);
    let kinds: BTreeSet<&str> = result.violations.iter().map(|v| v.rule).collect();
    assert_eq!(kinds, BTreeSet::from(["network-import", "frame-write"]));
    let notices: BTreeSet<&str> = result.notices.iter().map(|(v, _)| v.rule).collect();
    assert_eq!(notices, BTreeSet::from(["native-network"]));
}

#[test]
fn test_scan_source_itself_stays_strict() {
    // The allowance belongs to the file, never to a snippet or a different path.
    assert_eq!(rules(WINHTTP_LOAD), ["native-network", "native-network"]);
}

#[test]
fn test_only_the_fetch_rules_import_winhttp() {
    let pattern = Regex::new(r"\bwinhttp::").unwrap();
    let root = default_target();
    let mut importers: Vec<String> = walkdir::WalkDir::new(&root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .filter(|e| pattern.is_match(&fs::read_to_string(e.path()).unwrap()))
        .map(|e| posix(e.path().strip_prefix(&root).unwrap()))
        .collect();
    importers.sort();
    assert_eq!(importers, ["update/fetch.rs"]);
}

#[test]
fn test_default_target_is_the_package() {
    assert_eq!(
        default_target(),
        repo_root().join("crates").join("eye-tracker").join("src")
    );
    assert!(default_target().join("lib.rs").is_file());
}

// --------------------------------------------------------------------------------- violations
const VIOLATIONS: &[(&str, &str)] = &[
    ("use std::net::TcpStream;", "network-import"),
    ("use std::net;", "network-import"),
    ("use std::{fs, net::UdpSocket};", "network-import"),
    ("use core::net::SocketAddr;", "network-import"),
    (
        "fn f() { let s = std::net::TcpStream::connect(\"x:80\"); }",
        "network-import",
    ),
    ("use reqwest::blocking::get;", "network-import"),
    ("use ureq as u;", "network-import"),
    ("extern crate hyper;", "network-import"),
    ("use tokio::net::TcpListener;", "network-import"),
    ("use socket2::Socket;", "network-import"),
    ("use native_tls::TlsConnector;", "network-import"),
    ("use rustls::ClientConfig;", "network-import"),
    ("use openssl::ssl::SslConnector;", "network-import"),
    ("use curl::easy::Easy;", "network-import"),
    (
        "fn f() { let _ = isahc::get(\"https://x.org\"); }",
        "network-import",
    ),
    (
        "fn f() { sentry::capture_message(\"x\", sentry::Level::Info); }",
        "network-import",
    ),
    (
        "fn f() { webbrowser::open(\"https://x.org\").ok(); }",
        "network-import",
    ),
    (
        "fn f() { let _ = open::that(\"https://x.org\"); }",
        "network-import",
    ),
    (
        "use std::net::*;\nfn f() { TcpStream::connect(\"x:1\").ok(); }",
        "network-api",
    ),
    ("fn f(a: &str) { a.to_socket_addrs().ok(); }", "network-api"),
    (
        "async fn f() { tokio::net::lookup_host(\"x:1\").await.ok(); }",
        "network-api",
    ),
    (
        "use image::ImageFormat;\nfn f(i: &image::RgbImage) { i.save(\"a.png\").unwrap(); }",
        "frame-write",
    ),
    (
        "fn f(frame: &Frame) { frame.to_rgb_image().save(\"face.png\").unwrap(); }",
        "frame-write",
    ),
    (
        "fn f(i: &RgbImage, w: &mut Cursor<Vec<u8>>) { i.write_to(w, ImageFormat::Png).unwrap(); }",
        "frame-write",
    ),
    ("use image::codecs::png::PngEncoder;", "frame-write"),
    ("use image::save_buffer;", "frame-write"),
    (
        "fn f(frame: &Frame) { std::fs::write(\"f.raw\", &frame.data).unwrap(); }",
        "frame-write",
    ),
    (
        "fn f(file: &mut File, frame: &Frame) { file.write_all(&frame.data).unwrap(); }",
        "frame-write",
    ),
    (
        "use windows::Win32::Networking::WinSock::WSAStartup;",
        "native-network",
    ),
    (
        "fn f() { unsafe { LoadLibraryW(w!(\"ws2_32.dll\")) }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { libloading::Library::new(\"libcurl.so.4\") }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { libc::socket(2, 1, 0) }; }",
        "native-network",
    ),
    (
        "fn f() { Command::new(\"curl\").arg(\"-O\").status().ok(); }",
        "network-command",
    ),
    (
        "fn f() { Command::new(\"wget http://example.org/x\"); }",
        "network-command",
    ),
    (
        "const CMD: [&str; 2] = [\"C:\\\\tools\\\\curl.exe\", \"--help\"];",
        "network-command",
    ),
    ("fn f() { which(\"ssh\"); }", "network-command"),
    ("fn broken( {", "syntax-error"),
];

#[test]
fn test_detects_violation() {
    for (source, rule) in VIOLATIONS {
        assert!(
            rules(source).contains(rule),
            "{source:?} should break {rule}: {:?}",
            rules(source)
        );
    }
}

/// Ordinary APIs that could slip past the check.
const BYPASSES: &[(&str, &str)] = &[
    // Networking modules reached through other crates.
    ("use async_std::net::TcpStream;", "network-import"),
    ("use nix::sys::socket::connect;", "network-import"),
    (
        "fn f() { rustix::net::socket(a, b, None).ok(); }",
        "network-import",
    ),
    ("use mio::net::TcpStream;", "network-import"),
    (
        "use std::net as n;\nfn f() { n::UdpSocket::bind(\"0.0.0.0:0\").ok(); }",
        "network-api",
    ),
    // Raw sockets and the resolver through libc and FFI.
    (
        "fn f() { unsafe { libc::connect(fd, addr, 16) }; }",
        "native-network",
    ),
    (
        "fn f() { unsafe { libc::sendto(fd, b, 1, 0, a, 16) }; }",
        "native-network",
    ),
    (
        "fn f() { unsafe { libc::getaddrinfo(h, s, hints, res) }; }",
        "native-network",
    ),
    (
        "unsafe extern \"C\" {\n    fn getaddrinfo(n: *const i8) -> i32;\n}",
        "native-network",
    ),
    (
        "#[link(name = \"wininet\")]\nunsafe extern \"system\" {}",
        "native-network",
    ),
    (
        "#[link(name = \"ws2_32\", kind = \"raw-dylib\")]\nunsafe extern \"system\" {}",
        "native-network",
    ),
    (
        "fn f() { unsafe { dlsym(lib, c\"gethostbyname\".as_ptr()) }; }",
        "native-network",
    ),
    (
        "fn f() { unsafe { GetProcAddress(h, s!(\"WSAStartup\")) }; }",
        "native-network",
    ),
    (
        "fn f() { api.WinHttpOpen(None, 0, None, None, 0); }",
        "native-network",
    ),
    (
        "use windows::Win32::NetworkManagement::IpHelper::GetAdaptersInfo;",
        "native-network",
    ),
    (
        "use windows::Win32::Networking::WinInet::*;",
        "native-network",
    ),
    (
        "fn f() { unsafe { LoadLibraryW(w!(\"winhttp\")) }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { LoadLibraryA(s!(\"dnsapi\")) }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { Library::new(\"libcurl-gnutls.so.4\") }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { Library::new(\"/usr/lib/x86_64-linux-gnu/libssl.so.3\") }.ok(); }",
        "native-network",
    ),
    (
        "fn f() { unsafe { dlopen(c\"/System/Library/Frameworks/CFNetwork.framework/CFNetwork\".as_ptr(), 1) }; }",
        "native-network",
    ),
    // Network tools behind shells and wrappers.
    (
        "fn f() { Command::new(\"sh\").args([\"-c\", \"curl https://x.org\"]).status().ok(); }",
        "network-command",
    ),
    (
        "fn f() { Command::new(\"sudo\").arg(\"wget\").arg(\"https://x.org\"); }",
        "network-command",
    ),
    (
        "fn f() { let argv = vec![\"sh\", \"-c\", \"echo hi && curl https://x.org\"]; }",
        "network-command",
    ),
    (
        "fn f() { Command::new(\"powershell\").args(&[\"-c\", \"iwr https://x.org\"]); }",
        "network-command",
    ),
    (
        "fn f() { Command::new(\"cmd\").args([\"/c\", \"certutil -urlcache -f x y\"]); }",
        "network-command",
    ),
    (
        "fn f() { run_command(&[\"ping\".into(), host], 1.0).ok(); }",
        "network-command",
    ),
    (
        "fn f() { let argv = (\"/usr/bin/wget\", args); }",
        "network-command",
    ),
    (
        "fn f() { unsafe { libc::system(c\"wget http://x.org\".as_ptr()) }; }",
        "network-command",
    ),
];

#[test]
fn test_detects_bypasses() {
    for (source, rule) in BYPASSES {
        assert!(
            rules(source).contains(rule),
            "{source:?} should break {rule}: {:?}",
            rules(source)
        );
    }
}

const LEGITIMATE: &[&str] = &[
    // Local IPC is the app's one use of sockets.
    "use interprocess::local_socket::{GenericNamespaced, Stream};",
    "use std::os::unix::net::UnixStream;",
    // Look-alike method names.
    "fn f() { button.connect(on_click); server.listen(name); socket.accept_later(); }",
    "fn f() { self.worker.connect(self.slot); watcher.bind(key); }",
    // Settings, traces and update files are written; frames are not.
    "fn f(s: &Settings) { s.save(&path).unwrap(); self.memory.save(); }",
    "fn f() { std::fs::write(&path, text).unwrap(); file.write_all(line.as_bytes()).unwrap(); }",
    "use image::RgbImage;\nfn f(i: &RgbImage) -> image::Rgb<u8> { *i.get_pixel(0, 0) }",
    "fn f(frame: &Frame) -> Vec<u8> { frame.data.clone() }",
    // Ordinary system tools, local modules and look-alike names are fine.
    "fn f() { Command::new(\"loginctl\").arg(\"lock-session\").status().ok(); }",
    "fn f() { Command::new(\"gdbus\").args([\"call\", \"--session\"]); }",
    "fn f() { Command::new(\"sh\").args([\"-c\", \"xset dpms force off\"]); }",
    "fn f() { Command::new(\"sudo\").arg(\"true\"); }",
    "fn f() { Command::new(\"xdg-open\").arg(\"/tmp\"); }",
    "use crate::net::thing;\nuse super::http;\nuse self::socket2::X;",
    "mod reqwest;\nuse reqwest::Thing;",
    "use sockets_helper::x;\nuse httpish::y;",
    "fn f() { unsafe { LoadLibraryW(w!(\"user32.dll\")) }.ok(); }",
    "fn f() { unsafe { libc::inotify_init1(0); libc::kill(pid, 0) }; }",
    "fn create_menu() {}",
    "const REPO_URL: &str = \"https://github.com/bugraskl/eye-tracker\";",
    "const WS: &str = \"curl is just a word in a string\";",
    // Test code is not part of the app.
    "#[cfg(test)]\nmod tests {\n    use std::net::TcpListener;\n}",
];

#[test]
fn test_allows_legitimate_code() {
    for source in LEGITIMATE {
        assert_eq!(rules(source), Vec::<&str>::new(), "{source:?}");
    }
}

const MACOS_NETWORKING: &[(&str, &str)] = &[
    // Foundation does HTTP(S) on its own, and the app's macOS code loads Foundation.
    (
        "fn f() { unsafe { NSURLSession::sharedSession().dataTaskWithRequest(&r).resume() }; }",
        "macos-network",
    ),
    ("use objc2_foundation::NSURLSession;", "macos-network"),
    (
        "fn f() { NSURLSessionConfiguration::defaultSessionConfiguration(); }",
        "macos-network",
    ),
    (
        "fn f() { let task: Option<&NSURLSessionDataTask> = None; }",
        "macos-network",
    ),
    (
        "fn f() { NSMutableURLRequest::requestWithURL(&url); }",
        "macos-network",
    ),
    (
        "fn f() { NSURLConnection::sendSynchronousRequest_returningResponse_error(&r, None); }",
        "macos-network",
    ),
    ("fn f() { NSURLDownload::alloc(); }", "macos-network"),
    (
        "fn f() { NSHost::hostWithName(ns_string!(\"example.org\")); }",
        "macos-network",
    ),
    ("fn f() { NSNetServiceBrowser::new(); }", "macos-network"),
    (
        "fn f() { AnyClass::get(c\"NSURLSession\"); }",
        "macos-network",
    ),
    (
        "fn f() { objc_getClass(c\"NSURLConnection\".as_ptr()); }",
        "macos-network",
    ),
    (
        "fn f() { NSClassFromString(&NSString::from_str(\"NSURLSession\")); }",
        "macos-network",
    ),
    (
        "fn f() { let c = class!(NSURLConnection); }",
        "macos-network",
    ),
    // Reading whatever a URL names: a web address is fetched over the network.
    (
        "fn f() { NSData::dataWithContentsOfURL(&NSURL::URLWithString(ns_string!(\"https://example.org/c.json\")).unwrap()); }",
        "macos-network",
    ),
    (
        "fn get(url: &NSURL) -> Option<Retained<NSData>> { NSData::dataWithContentsOfURL(url) }",
        "macos-network",
    ),
    (
        "fn f() { NSData::initWithContentsOfURL_options_error(NSData::alloc(), &url, o); }",
        "macos-network",
    ),
    (
        "fn f() { NSString::stringWithContentsOfURL_encoding_error(&url, 4); }",
        "macos-network",
    ),
    (
        "fn f() { NSDictionary::dictionaryWithContentsOfURL(&url); }",
        "macos-network",
    ),
    (
        "fn f() { NSImage::initByReferencingURL(NSImage::alloc(), &url); }",
        "macos-network",
    ),
    (
        "fn f() { NSData::dataWithContentsOfURL(); }",
        "macos-network",
    ),
    (
        "fn f() { let read = NSData::dataWithContentsOfURL; }",
        "macos-network",
    ),
    (
        "fn f() { let d: *mut AnyObject = unsafe { msg_send![data, initWithContentsOfURL: url] }; }",
        "macos-network",
    ),
    // A name that may hold a web address is not a file URL.
    (
        "fn f(remote: bool) {\n    let mut url = NSURL::fileURLWithPath(&p);\n    if remote {\n        \
         url = NSURL::URLWithString(&s).unwrap();\n    }\n    NSData::dataWithContentsOfURL(&url);\n}",
        "macos-network",
    ),
    // Core Foundation's sockets to hosts and HTTP streams.
    (
        "fn f() { unsafe { CFStreamCreatePairWithSocketToHost(None, host, 80, r, w) }; }",
        "macos-network",
    ),
    (
        "use objc2_core_foundation::CFReadStreamCreateForHTTPRequest;",
        "macos-network",
    ),
    // Apple's networking frameworks, imported or loaded.
    ("use objc2_cf_network::CFHTTPMessage;", "network-import"),
    ("use objc2_network::nw_connection_t;", "network-import"),
    ("use objc2_web_kit::WKWebView;", "network-import"),
    (
        "fn f() { NSBundle::bundleWithPath(ns_string!(\"/System/Library/Frameworks/CFNetwork.framework\")); }",
        "macos-network",
    ),
    (
        "fn f() { NSBundle::bundleWithPath(ns_string!(\"/System/Library/Frameworks/WebKit.framework\")); }",
        "macos-network",
    ),
    (
        "fn f() { NSBundle::bundleWithIdentifier(ns_string!(\"com.apple.Network\")); }",
        "macos-network",
    ),
    (
        "fn f() { unsafe { dlopen(c\"/System/Library/Frameworks/WebKit.framework/WebKit\".as_ptr(), 1) }; }",
        "native-network",
    ),
];

#[test]
fn test_detects_macos_networking() {
    for (source, rule) in MACOS_NETWORKING {
        assert!(
            rules(source).contains(rule),
            "{source:?} should break {rule}: {:?}",
            rules(source)
        );
    }
}

const APPS_OBJC2_CODE: &[&str] = &[
    // What the app's macOS code does with objc2 (platform/macos.rs).
    "fn f() { let app = unsafe { NSWorkspace::sharedWorkspace().frontmostApplication() }; }",
    "fn f() { NSProcessInfo::processInfo().beginActivityWithOptions_reason(o, ns_string!(\"x\")); }",
    "fn f() { AnyClass::get(c\"AVCaptureDevice\"); }",
    "fn f() { NSBundle::bundleWithPath(ns_string!(\"/System/Library/Frameworks/AVFoundation.framework\")); }",
    "fn f() { AVCaptureDevice::authorizationStatusForMediaType(media); }",
    "fn f() { NSRunningApplication::runningApplicationWithProcessIdentifier(pid); }",
    // File URLs are read locally.
    "fn f() { NSData::dataWithContentsOfURL(&NSURL::fileURLWithPath(&path)); }",
    "fn f() { NSData::initWithContentsOfURL(NSData::alloc(), &NSURL::fileURLWithPath_isDirectory(&p, false)); }",
    "fn f() {\n    let url = NSURL::fileURLWithPath(&path);\n    NSString::stringWithContentsOfURL_encoding_error(&url, 4);\n}",
    "impl A {\n    fn f(&mut self) {\n        self.url = NSURL::fileURLWithPath(&p);\n        let other = self.url.clone();\n        \
     NSData::dataWithContentsOfURL(&other);\n    }\n}",
    "fn f() { NSImage::initByReferencingURL(NSImage::alloc(), &NSURL::URLWithString(ns_string!(\"file:///tmp/icon.png\")).unwrap()); }",
    // A URL object that is not read, and look-alike names.
    "fn f() { NSURL::URLWithString(ns_string!(\"x-apple.systempreferences:com.apple.preference.security\")); }",
    "fn f() { manager.contentsOfDirectoryAtURL_includingPropertiesForKeys_options_error(&u, None, o); }",
    "const NSURLRequestUseProtocolCachePolicy: u32 = 0;\nconst HOSTNAME: &str = \"NSHostName\";",
];

#[test]
fn test_allows_the_apps_objc2_code() {
    for source in APPS_OBJC2_CODE {
        assert_eq!(rules(source), Vec::<&str>::new(), "{source:?}");
    }
}

#[test]
fn test_reports_location() {
    let dir = TempDir::new().unwrap();
    let bad = dir.path().join("leaky.rs");
    fs::write(&bad, "use std::fs;\n\nuse std::net;\n").unwrap();
    let [violation] = scan_file(&bad).try_into().unwrap();
    assert_eq!(
        (&violation.path, violation.line, violation.col),
        (&bad, 3, 1)
    );
    assert_eq!(violation.rule, "network-import");
    assert!(
        violation
            .to_string()
            .ends_with("3:1: network-import: import of networking module 'std::net'")
    );
}

#[test]
fn test_synthetic_violating_package_is_detected() {
    let dir = TempDir::new().unwrap();
    let package = dir.path().join("pkg");
    fs::create_dir_all(package.join("sub")).unwrap();
    fs::write(package.join("lib.rs"), "mod clean;\nmod sub;\n").unwrap();
    fs::write(package.join("clean.rs"), "use std::fs;\n").unwrap();
    fs::write(
        package.join("sub").join("leak.rs"),
        "use std::os::unix::net::UnixStream;\nuse reqwest::blocking::Client;\n\n\n\
         fn save(frame: &Frame) {\n    std::fs::write(\"frame.raw\", &frame.data).unwrap();\n}\n",
    )
    .unwrap();
    // Build output, test files and test helpers are never scanned.
    for junk in ["target/junk.rs", "sub/leak_tests.rs", "testing/fakes.rs"] {
        let path = package.join(junk);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "use std::net;\n").unwrap();
    }

    let result = scan_paths(std::slice::from_ref(&package), &package);

    assert_eq!(result.files.len(), 3);
    assert!(!result.ok());
    let found: BTreeSet<(String, usize, &str)> = result
        .violations
        .iter()
        .map(|v| {
            (
                v.path.file_name().unwrap().to_string_lossy().into(),
                v.line,
                v.rule,
            )
        })
        .collect();
    assert_eq!(
        found,
        BTreeSet::from([
            ("leak.rs".into(), 2, "network-import"),
            ("leak.rs".into(), 6, "frame-write")
        ])
    );
}

#[test]
fn test_non_utf8_source_is_not_verified() {
    let dir = TempDir::new().unwrap();
    let legacy = dir.path().join("legacy.rs");
    fs::write(&legacy, b"const NAME: &str = \"\xe7a\";\n").unwrap();
    let found: Vec<&str> = scan_file(&legacy).iter().map(|v| v.rule).collect();
    assert_eq!(found, ["syntax-error"]);
}

// ------------------------------------------------------------------------------ command line
#[test]
fn test_main_exit_codes() {
    let dir = TempDir::new().unwrap();
    let clean = dir.path().join("clean");
    fs::create_dir(&clean).unwrap();
    fs::write(
        clean.join("ok.rs"),
        "use interprocess::local_socket::Stream;\n",
    )
    .unwrap();
    let leaky = dir.path().join("leaky");
    fs::create_dir(&leaky).unwrap();
    fs::write(leaky.join("bad.rs"), "use reqwest;\n").unwrap();

    let report = run(&source_args(&[&clean]), None);
    assert_eq!(report.code, 0);
    assert!(report.out.contains("OK: 1 files"), "{}", report.out);

    let report = run(&source_args(&[&leaky]), None);
    assert_eq!(report.code, 1);
    assert!(
        report.out.contains("bad.rs:1:1: network-import"),
        "{}",
        report.out
    );
    assert!(
        report.err.contains("FAILED: 1 violation(s)"),
        "{}",
        report.err
    );

    assert_eq!(
        run(&source_args(&[&dir.path().join("missing")]), None).code,
        2
    );
}

/// Under GitHub Actions each violation becomes an error annotation on its line.
#[test]
fn test_github_annotations() {
    let dir = TempDir::new().unwrap();
    let bad = dir.path().join("bad.rs");
    fs::write(&bad, "use std::fs;\nuse rustls;\n").unwrap();
    let clean = dir.path().join("clean");
    fs::create_dir(&clean).unwrap();
    fs::write(clean.join("ok.rs"), "use std::fs;\n").unwrap();

    let report = run(&source_args(&[&bad]), Some("true"));
    assert_eq!(report.code, 1);
    assert_eq!(
        report.out,
        format!(
            "::error file={},line=2,col=1,title=Privacy check::\
             network-import: import of networking module 'rustls'\n",
            display_path(&bad)
        )
    );
    // The summary stays a plain line on stderr.
    assert!(
        report
            .err
            .starts_with("check_privacy: FAILED: 1 violation(s) in 1 of 1 files.")
    );

    let report = run(&source_args(&[&clean]), Some("true"));
    assert_eq!(report.code, 0);
    assert!(report.out.starts_with("check_privacy: OK: 1 files in "));
}

/// The runner sets GITHUB_ACTIONS=true; anything else gets plain lines.
#[test]
fn test_github_annotations_only_under_github_actions() {
    let dir = TempDir::new().unwrap();
    let bad = dir.path().join("bad.rs");
    fs::write(&bad, "use rustls;\n").unwrap();
    for value in ["", "false", "1", "TRUE"] {
        let report = run(&source_args(&[&bad]), Some(value));
        assert_eq!(report.code, 1);
        assert_eq!(
            report.out,
            format!(
                "{}:1:1: network-import: import of networking module 'rustls'\n",
                display_path(&bad)
            )
        );
    }
}

#[test]
fn test_github_annotation_escapes_the_message() {
    let violation = Violation {
        path: Path::new("pkg").join("a.rs"),
        line: 3,
        col: 5,
        rule: "frame-write",
        message: "50%\nof it".into(),
    };
    assert_eq!(
        github_annotation(&violation),
        "::error file=pkg/a.rs,line=3,col=5,title=Privacy check::frame-write: 50%25%0Aof it"
    );
    assert_eq!(escape("a%b\r\nc"), "a%25b%0D%0Ac");
}

/// The `check-privacy` subcommand of `cargo xtask` (the Python ran the script standalone).
#[test]
fn test_command_line_runs_check_privacy() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("bad.rs"), "use std::net;\n").unwrap();
    let path = dir.path().to_string_lossy().into_owned();
    let cli = crate::Cli::try_parse_from(["xtask", "check-privacy", "--quiet", &path]).unwrap();
    let crate::Command::CheckPrivacy(args) = cli.command else {
        panic!("not check-privacy")
    };
    assert!(args.quiet);
    let report = run(&args, None);
    assert_eq!(report.code, 1, "{}", report.err);
    assert!(report.out.contains("network-import"));

    // --bundle takes no source paths: a usage error, exit status 2.
    let error = crate::Cli::try_parse_from(["xtask", "check-privacy", "--bundle", &path, "src"])
        .err()
        .unwrap();
    assert_eq!(error.exit_code(), 2);
}

// ================================================================================ dependencies
fn node(name: &str, dependencies: &[usize]) -> CrateNode {
    CrateNode {
        name: name.into(),
        version: "1.0.0".into(),
        dependencies: dependencies.to_vec(),
    }
}

#[test]
fn test_network_crates_in_the_tree_fail_unless_allowed() {
    let crates = [
        node("eye-tracker", &[1, 2]),
        node("tokio", &[3, 4, 6]),
        node("uploader", &[5]),
        node("mio", &[]),
        node("socket2", &[]),
        node("reqwest", &[]),
        node("hyper", &[]),
    ];
    let result = scan_dependencies(&crates, DEPENDENCY_ALLOWLIST);
    assert_eq!(result.crates, 7);
    let found: Vec<String> = result
        .violations
        .iter()
        .map(BundleFinding::to_string)
        .collect();
    // The allow rule is per indicator: tokio may need mio and socket2, not hyper.
    assert_eq!(
        found,
        [
            "tokio 1.0.0: network-crate: needs hyper 1.0.0",
            "uploader 1.0.0: network-crate: needs reqwest 1.0.0"
        ]
    );
    let allowed: Vec<&str> = result.allowed.iter().map(|f| f.detail.as_str()).collect();
    assert_eq!(allowed, ["needs mio 1.0.0", "needs socket2 1.0.0"]);
}

#[test]
fn test_the_app_crate_may_not_depend_on_a_network_crate() {
    let crates = [node("eye-tracker", &[1]), node("ureq", &[])];
    let result = scan_dependencies(&crates, DEPENDENCY_ALLOWLIST);
    assert!(!result.ok());
    assert_eq!(result.violations[0].relative, "eye-tracker 1.0.0");
}

#[test]
fn test_app_dependencies_follow_normal_dependencies_only() {
    let kind = |k: Option<&str>| json!([{ "kind": k, "target": null }]);
    let metadata = json!({
        "workspace_members": ["app"],
        "packages": [
            { "id": "app", "name": "eye-tracker", "version": "0.3.0" },
            { "id": "a", "name": "alpha", "version": "1.0.0" },
            { "id": "b", "name": "reqwest", "version": "0.12.0" },
            { "id": "d", "name": "ureq", "version": "3.0.0" },
            { "id": "x", "name": "cc", "version": "1.0.0" },
        ],
        "resolve": { "nodes": [
            { "id": "app", "deps": [
                { "pkg": "a", "dep_kinds": kind(None) },
                { "pkg": "d", "dep_kinds": kind(Some("dev")) },
                { "pkg": "x", "dep_kinds": kind(Some("build")) },
            ] },
            { "id": "a", "deps": [{ "pkg": "b", "dep_kinds": kind(None) }] },
            { "id": "b", "deps": [] },
            { "id": "d", "deps": [] },
            { "id": "x", "deps": [] },
        ] },
    });
    let crates = app_dependencies(&metadata).unwrap();
    let names: Vec<&str> = crates.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["eye-tracker", "alpha", "reqwest"]);
    assert_eq!(crates[0].dependencies, [1]);
    assert_eq!(crates[1].dependencies, [2]);
    let result = scan_dependencies(&crates, DEPENDENCY_ALLOWLIST);
    let found: Vec<String> = result
        .violations
        .iter()
        .map(BundleFinding::to_string)
        .collect();
    assert_eq!(found, ["alpha 1.0.0: network-crate: needs reqwest 0.12.0"]);
}

#[test]
fn test_the_real_dependency_tree_is_allowed() {
    let crates = app_dependencies(&load_metadata(&repo_root()).unwrap()).unwrap();
    assert!(crates.len() > 100, "the scan should cover the whole tree");
    let result = scan_dependencies(&crates, DEPENDENCY_ALLOWLIST);
    let found: Vec<String> = result
        .violations
        .iter()
        .map(BundleFinding::to_string)
        .collect();
    assert!(result.ok(), "networking crates:\n{}", found.join("\n"));
}

// ================================================================ synthetic native binaries
/// A minimal PE image with one section holding the import tables.
fn pe(imports: &[&str], delay: &[&str], pe32: bool) -> Vec<u8> {
    let (section_rva, section_offset) = (0x1000u32, 0x200usize);
    let descriptors = 20 * (imports.len() + 1);
    let delay_start = descriptors;
    let names_start = delay_start + 32 * (delay.len() + 1);
    let mut names = Vec::new();
    let mut name_rvas = Vec::new();
    for name in imports.iter().chain(delay) {
        name_rvas.push(section_rva + (names_start + names.len()) as u32);
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    let mut body = Vec::new();
    for rva in &name_rvas[..imports.len()] {
        for value in [0, 0, 0, *rva, 0] {
            body.extend(value.to_le_bytes());
        }
    }
    body.extend([0; 20]);
    for rva in &name_rvas[imports.len()..] {
        for value in [1, *rva, 0, 0, 0, 0, 0, 0] {
            body.extend(u32::to_le_bytes(value));
        }
    }
    body.extend([0; 32]);
    body.extend(names);
    body.resize(body.len().next_multiple_of(0x200), 0);

    let optional_size = if pe32 { 224 } else { 240 };
    let mut optional = vec![0u8; optional_size];
    optional[..2].copy_from_slice(&(if pe32 { 0x10Bu16 } else { 0x20B }).to_le_bytes());
    let directories = if pe32 { 96 } else { 112 };
    optional[directories - 4..directories].copy_from_slice(&16u32.to_le_bytes());
    optional[directories + 8..directories + 12].copy_from_slice(&section_rva.to_le_bytes());
    optional[directories + 12..directories + 16]
        .copy_from_slice(&(descriptors as u32).to_le_bytes());
    if !delay.is_empty() {
        let at = directories + 8 * 13;
        optional[at..at + 4].copy_from_slice(&(section_rva + delay_start as u32).to_le_bytes());
        optional[at + 4..at + 8].copy_from_slice(&32u32.to_le_bytes());
    }
    let mut header = vec![0u8; 0x40];
    header[..2].copy_from_slice(b"MZ");
    header[0x3C..0x40].copy_from_slice(&0x40u32.to_le_bytes());
    header.extend(b"PE\0\0");
    header.extend((if pe32 { 0x14Cu16 } else { 0x8664 }).to_le_bytes());
    header.extend(1u16.to_le_bytes());
    header.extend([0; 12]);
    header.extend((optional_size as u16).to_le_bytes());
    header.extend(0x2022u16.to_le_bytes());
    header.extend(optional);
    header.extend(b".idata\0\0");
    let size = body.len() as u32;
    for value in [size, section_rva, size, section_offset as u32, 0, 0] {
        header.extend(value.to_le_bytes());
    }
    header.extend([0; 8]);
    header.resize(section_offset, 0);
    header.extend(body);
    header
}

/// A minimal ELF shared object: .dynstr, .dynamic and .dynsym sections only.
fn elf(
    needed: &[&str],
    undefined: &[&str],
    defined: &[&str],
    is64: bool,
    big_endian: bool,
) -> Vec<u8> {
    let half = |v: u16| {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    };
    let word = |v: u32| {
        if big_endian {
            v.to_be_bytes()
        } else {
            v.to_le_bytes()
        }
    };
    // An address or offset: 8 bytes in ELF64, 4 in ELF32.
    let addr = |v: u64| -> Vec<u8> {
        match (is64, big_endian) {
            (true, true) => v.to_be_bytes().to_vec(),
            (true, false) => v.to_le_bytes().to_vec(),
            (false, _) => word(v as u32).to_vec(),
        }
    };
    let mut strings = vec![0u8];
    let mut offsets = std::collections::HashMap::new();
    for name in needed.iter().chain(undefined).chain(defined) {
        offsets.insert(*name, strings.len() as u64);
        strings.extend(name.as_bytes());
        strings.push(0);
    }
    let entry = if is64 { 16 } else { 8 };
    let mut dynamic = Vec::new();
    for library in needed {
        dynamic.extend(addr(1));
        dynamic.extend(addr(offsets[library]));
    }
    dynamic.extend(vec![0; entry]);
    let mut symbols = vec![0u8; if is64 { 24 } else { 16 }]; // the null symbol
    let info = (1 << 4) | 2; // GLOBAL FUNC
    for (name, index) in undefined
        .iter()
        .map(|n| (n, 0))
        .chain(defined.iter().map(|n| (n, 7)))
    {
        symbols.extend(word(offsets[name] as u32));
        if is64 {
            symbols.extend([info, 0]);
            symbols.extend(half(index));
            symbols.extend([0; 16]);
        } else {
            symbols.extend([0; 8]);
            symbols.extend([info, 0]);
            symbols.extend(half(index));
        }
    }
    let header_size = if is64 { 64 } else { 52 };
    let strings_offset = header_size as u64;
    let dynamic_offset = strings_offset + strings.len() as u64;
    let symbols_offset = dynamic_offset + dynamic.len() as u64;
    let sections_offset = symbols_offset + symbols.len() as u64;

    let section = |kind: u32, offset: u64, size: u64, link: u32| -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(word(0));
        out.extend(word(kind));
        out.extend(addr(0));
        out.extend(addr(0));
        out.extend(addr(offset));
        out.extend(addr(size));
        out.extend(word(link));
        out.extend(word(0));
        out.extend(addr(0));
        out.extend(addr(0));
        out
    };
    let mut data = b"\x7fELF".to_vec();
    data.extend([if is64 { 2 } else { 1 }, if big_endian { 2 } else { 1 }, 1]);
    data.extend([0; 9]);
    data.extend(half(3));
    data.extend(half(if is64 { 62 } else { 3 }));
    data.extend(word(1));
    data.extend(addr(0));
    data.extend(addr(0));
    data.extend(addr(sections_offset));
    data.extend(word(0));
    for value in [
        header_size,
        if is64 { 56 } else { 32 },
        0,
        if is64 { 64 } else { 40 },
        4,
        0,
    ] {
        data.extend(half(value));
    }
    data.extend(strings.iter());
    data.extend(dynamic);
    data.extend(symbols);
    data.extend(section(0, 0, 0, 0));
    data.extend(section(3, strings_offset, strings.len() as u64, 0));
    data.extend(section(
        6,
        dynamic_offset,
        symbols_offset - dynamic_offset,
        1,
    ));
    data.extend(section(
        11,
        symbols_offset,
        sections_offset - symbols_offset,
        1,
    ));
    data
}

/// A minimal 64-bit little-endian Mach-O dylib, optionally in a universal wrapper.
fn macho(dylibs: &[&str], undefined: &[&str], defined: &[&str], fat: bool) -> Vec<u8> {
    let le = |values: &[u32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let mut commands = Vec::new();
    for path in dylibs {
        let mut name = path.as_bytes().to_vec();
        name.push(0);
        name.resize((24 + name.len()).next_multiple_of(8) - 24, 0);
        commands.extend(le(&[0xC, 24 + name.len() as u32, 24, 0, 0, 0]));
        commands.extend(name);
    }
    let mut strings = vec![0u8];
    let mut entries = Vec::new();
    for (names, kind) in [(undefined, 0x01u8), (defined, 0x0F)] {
        for symbol in names {
            entries.push((strings.len() as u32, kind));
            strings.push(b'_');
            strings.extend(symbol.as_bytes());
            strings.push(0);
        }
    }
    let header_size = 32;
    let commands_size = commands.len() as u32 + 24;
    let symbols_offset = header_size + commands_size;
    let mut symbols = Vec::new();
    for (offset, kind) in &entries {
        symbols.extend(offset.to_le_bytes());
        symbols.extend([*kind, 0, 0, 0]);
        symbols.extend([0; 8]);
    }
    let strings_offset = symbols_offset + symbols.len() as u32;
    commands.extend(le(&[
        0x2,
        24,
        symbols_offset,
        entries.len() as u32,
        strings_offset,
        strings.len() as u32,
    ]));
    let count = dylibs.len() as u32 + 1;
    let mut thin = le(&[0xFEEDFACF, 0x0100000C, 0, 6, count, commands_size, 0, 0]);
    thin.extend(commands);
    thin.extend(symbols);
    thin.extend(strings);
    if !fat {
        return thin;
    }
    let offset = 0x1000;
    let mut wrapper: Vec<u8> = [0xCAFEBABE, 1, 0x0100000C, 0, offset, thin.len() as u32, 12]
        .iter()
        .flat_map(|v: &u32| v.to_be_bytes())
        .collect();
    wrapper.resize(offset as usize, 0);
    wrapper.extend(thin);
    wrapper
}

fn mz_without_pe() -> Vec<u8> {
    let mut data = b"MZ".to_vec();
    data.extend([0; 0x3A]);
    data.extend(0x40u32.to_le_bytes());
    data
}

// ------------------------------------------------------------------------- binary readers
#[test]
fn test_pe_imports_and_delay_imports() {
    for pe32 in [false, true] {
        let data = pe(&["KERNEL32.dll", "WS2_32.dll"], &["WININET.dll"], pe32);
        let binary = parse_binary(&data).unwrap().unwrap();
        assert_eq!(binary.format, "pe");
        assert_eq!(
            binary.libraries,
            ["KERNEL32.dll", "WS2_32.dll", "WININET.dll"]
        );
        let keys: Vec<String> = network_indicators(&binary)
            .into_iter()
            .map(|i| i.key)
            .collect();
        assert_eq!(keys, ["ws2_32", "wininet"]);
    }
}

#[test]
fn test_elf_needed_libraries_and_undefined_symbols() {
    for (is64, big_endian) in [(true, false), (false, true), (false, false)] {
        let data = elf(
            &["libc.so.6", "libcurl.so.4"],
            &["getaddrinfo", "connect", "socket"],
            &["gethostbyname"], // defined here, not imported: not an indicator
            is64,
            big_endian,
        );
        let binary = parse_binary(&data).unwrap().unwrap();
        assert_eq!(binary.format, "elf");
        assert_eq!(binary.libraries, ["libc.so.6", "libcurl.so.4"]);
        assert_eq!(
            binary.symbols,
            BTreeSet::from(["getaddrinfo", "connect", "socket"].map(String::from))
        );
        let indicators: Vec<(&str, String)> = network_indicators(&binary)
            .into_iter()
            .map(|i| (i.rule, i.key))
            .collect();
        // socket/connect are also local IPC: only the resolver and libcurl count.
        assert_eq!(
            indicators,
            [
                ("network-library", "libcurl".into()),
                ("network-symbol", "getaddrinfo".into())
            ]
        );
    }
}

#[test]
fn test_macho_dylibs_and_undefined_symbols() {
    for fat in [false, true] {
        let data = macho(
            &[
                "/System/Library/Frameworks/CFNetwork.framework/Versions/A/CFNetwork",
                "/usr/lib/libSystem.B.dylib",
                "/usr/lib/libresolv.9.dylib",
            ],
            &["getaddrinfo", "malloc"],
            &["gethostbyname"],
            fat,
        );
        let binary = parse_binary(&data).unwrap().unwrap();
        assert_eq!(binary.format, "macho");
        assert_eq!(binary.libraries[1], "/usr/lib/libSystem.B.dylib");
        assert_eq!(
            binary.symbols,
            BTreeSet::from(["getaddrinfo", "malloc"].map(String::from))
        );
        let keys: Vec<String> = network_indicators(&binary)
            .into_iter()
            .map(|i| i.key)
            .collect();
        assert_eq!(keys, ["cfnetwork", "libresolv", "getaddrinfo"]);
    }
}

#[test]
fn test_non_binaries_are_not_parsed() {
    let png = [b"\x89PNG\r\n\x1a\n".as_slice(), &[0; 64]].concat();
    // A Java class file shares the universal-binary magic number.
    let class = [b"\xca\xfe\xba\xbe\x00\x00\x00\x34".as_slice(), &[0; 64]].concat();
    for data in [b"".as_slice(), b"plain text, not a binary", &png, &class] {
        assert_eq!(parse_binary(data), Ok(None));
    }
}

#[test]
fn test_truncated_binaries_are_reported() {
    let pe_header = [mz_without_pe(), b"NOPE".to_vec(), vec![0; 64]].concat();
    let cases = [
        pe(&["WS2_32.dll"], &[], false)[..0x90].to_vec(),
        pe_header,
        elf(&["libc.so.6"], &["getaddrinfo"], &[], true, false)[..80].to_vec(),
        macho(&["/usr/lib/libSystem.B.dylib"], &[], &[], false)[..40].to_vec(),
    ];
    for data in cases {
        assert!(parse_binary(&data).is_err(), "{data:?}");
    }
}

#[test]
fn test_telemetry_markers_ascii_and_utf16_any_case() {
    assert_eq!(
        telemetry_markers(b"xx https://PLAY.googleapis.com/log xx"),
        ["play.googleapis.com"]
    );
    let wide: Vec<u8> = "AVClearcutLogger"
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .collect();
    assert_eq!(
        telemetry_markers(&[b"\0\0".as_slice(), &wide].concat()),
        ["clearcut"]
    );
    // The model NOTICE names where the weights came from; that is not telemetry.
    assert!(telemetry_markers(b"https://storage.googleapis.com/mediapipe-models").is_empty());
}

#[test]
fn test_committed_models_contain_no_telemetry_markers() {
    let models = repo_root()
        .join("crates")
        .join("eye-tracker")
        .join("models");
    let files: Vec<PathBuf> = fs::read_dir(models)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_file())
        .collect();
    assert!(!files.is_empty());
    for path in files {
        assert!(
            telemetry_markers(&fs::read(&path).unwrap()).is_empty(),
            "{}",
            path.display()
        );
    }
}

#[test]
fn test_allowlist_entries_are_explained_and_normalised() {
    for rule in BUNDLE_ALLOWLIST.iter().chain(DEPENDENCY_ALLOWLIST) {
        assert!(!rule.reason.trim().is_empty(), "{rule:?}");
        assert!(!rule.indicators.is_empty(), "{rule:?}");
        assert!(
            rule.indicators.iter().all(|i| *i == i.to_lowercase()),
            "{rule:?}"
        );
    }
    // Markers are never allow-listed: that list only holds libraries and symbols.
    let markers: Vec<String> = TELEMETRY_MARKERS.iter().map(|m| m.to_lowercase()).collect();
    assert!(
        BUNDLE_ALLOWLIST
            .iter()
            .flat_map(|r| r.indicators)
            .all(|i| !markers.contains(&i.to_string()))
    );
    // Dependency rules name networking crates, as their keys are written.
    for rule in DEPENDENCY_ALLOWLIST {
        assert!(
            rule.indicators.iter().all(|i| NETWORK_CRATES.contains(i)),
            "{rule:?}"
        );
    }
}

#[test]
fn test_allow_rules_match_from_the_right() {
    let rule = AllowRule {
        pattern: "eye-tracker*.exe",
        indicators: &["ws2_32"],
        reason: "x",
    };
    assert!(rule.matches("eye-tracker.exe"));
    assert!(rule.matches("bin/Eye-Tracker-GUI.EXE"));
    assert!(!rule.matches("eye-tracker.exe/other.dll"));
    assert!(!rule.matches("my-eye-tracker.exe"));
    assert!(rule.allows("ws2_32"));
    assert!(!rule.allows("wininet"));
    let nested = AllowRule {
        pattern: "lib/*.so",
        indicators: &["gethostby*"],
        reason: "x",
    };
    assert!(nested.matches("a/lib/x.so"));
    assert!(!nested.matches("x.so"));
    assert!(nested.allows("gethostbyname_r"));
}

// ------------------------------------------------------------------------------ bundle scans
fn bundle(dir: &Path, files: &[(&str, Vec<u8>)]) -> PathBuf {
    let root = dir.join("EyeTracker");
    for (relative, data) in files {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, data).unwrap();
    }
    root
}

fn summary(result: &BundleResult) -> BTreeSet<(String, &'static str)> {
    result
        .violations
        .iter()
        .map(|f| (f.relative.clone(), f.rule))
        .collect()
}

fn file_name(relative: &str) -> String {
    relative.rsplit('/').next().unwrap().to_string()
}

#[test]
fn test_clean_bundle_passes_with_allowed_networking() {
    let dir = TempDir::new().unwrap();
    let root = bundle(
        dir.path(),
        &[
            (
                "eye-tracker.exe",
                pe(&["KERNEL32.dll", "WS2_32.dll", "WINHTTP.dll"], &[], false),
            ),
            (
                "eye-tracker-gui.exe",
                pe(&["KERNEL32.dll", "WS2_32.dll", "WINHTTP.dll"], &[], false),
            ),
            (
                "eye-tracker",
                elf(
                    &["libc.so.6"],
                    &["getaddrinfo", "connect"],
                    &[],
                    true,
                    false,
                ),
            ),
            (
                "Eye Tracker.app/Contents/MacOS/eye-tracker-gui",
                macho(
                    &["/usr/lib/libSystem.B.dylib"],
                    &["getaddrinfo"],
                    &[],
                    false,
                ),
            ),
            (
                "models/NOTICE.md",
                b"https://storage.googleapis.com/mediapipe-models/face_landmarker.task".to_vec(),
            ),
            ("models/face_detection_yunet_2023mar.onnx", b"onnx".to_vec()),
        ],
    );
    let result = scan_bundle(&root, BUNDLE_ALLOWLIST);
    assert!(result.ok(), "{:?}", result.violations);
    assert_eq!(result.binaries, 4);
    assert_eq!(result.files, 6);
    let allowed: BTreeSet<(String, String)> = result
        .allowed
        .iter()
        .map(|f| (file_name(&f.relative), f.detail.clone()))
        .collect();
    assert!(allowed.contains(&("eye-tracker.exe".into(), "links WINHTTP.dll".into())));
    assert!(allowed.contains(&("eye-tracker".into(), "imports getaddrinfo".into())));
    assert!(allowed.contains(&("eye-tracker-gui".into(), "imports getaddrinfo".into())));
}

#[test]
fn test_foundation_url_loading_fails_the_gate() {
    // Foundation does HTTP(S) without CFNetwork being linked, so a library that uploads
    // with NSURLSession shows only in the classes it imports. A URL or a request alone
    // loads nothing.
    let foundation = "/System/Library/Frameworks/Foundation.framework/Versions/C/Foundation";
    let others = [
        "OBJC_CLASS_$_NSURL",
        "OBJC_CLASS_$_NSMutableURLRequest",
        "objc_msgSend",
    ];
    for symbol in [
        "OBJC_CLASS_$_NSURLSession",
        "OBJC_CLASS_$_NSURLSessionConfiguration",
        "OBJC_CLASS_$_NSURLConnection",
        "OBJC_CLASS_$_NSURLDownload",
        "CFReadStreamCreateForHTTPRequest",
    ] {
        let dir = TempDir::new().unwrap();
        let binary = macho(
            &[foundation, "/usr/lib/libobjc.A.dylib"],
            &[&[symbol][..], &others].concat(),
            &[],
            false,
        );
        let root = bundle(
            dir.path(),
            &[("Contents/Frameworks/libuploader.dylib", binary)],
        );
        let result = scan_bundle(&root, BUNDLE_ALLOWLIST);
        let found: Vec<(&str, String)> = result
            .violations
            .iter()
            .map(|f| (f.rule, f.detail.clone()))
            .collect();
        assert_eq!(found, [("network-symbol", format!("imports {symbol}"))]);
        let clean = bundle(
            &dir.path().join("clean"),
            &[("lib.dylib", macho(&[foundation], &others, &[], false))],
        );
        assert!(scan_bundle(&clean, BUNDLE_ALLOWLIST).ok());
    }
}

/// Apple's network frameworks, the counterparts of libgssapi, libkrb5 and libldap,
/// which the gate reports on Linux, and WebKit, a web browser engine.
#[test]
fn test_macos_network_frameworks_are_networking_libraries() {
    for framework in ["CFNetwork", "GSS", "Kerberos", "LDAP", "Network", "WebKit"] {
        let path =
            format!("/System/Library/Frameworks/{framework}.framework/Versions/A/{framework}");
        let data = macho(
            &[
                &path,
                "/System/Library/Frameworks/Security.framework/Security",
            ],
            &[],
            &[],
            false,
        );
        let binary = parse_binary(&data).unwrap().unwrap();
        let indicators: Vec<_> = network_indicators(&binary)
            .into_iter()
            .map(|i| (i.rule, i.name, i.key))
            .collect();
        assert_eq!(
            indicators,
            [("network-library", path.clone(), framework.to_lowercase())]
        );
    }
}

/// GLib's GIO resolves host names; nothing allows it into the bundle.
#[test]
fn test_gio_fails_the_gate() {
    let dir = TempDir::new().unwrap();
    let gio = elf(
        &["libglib-2.0.so.0"],
        &["getaddrinfo", "getnameinfo", "res_nquery", "malloc"],
        &[],
        true,
        false,
    );
    let result = scan_bundle(
        &bundle(dir.path(), &[("lib/libgio-2.0.so.0", gio)]),
        BUNDLE_ALLOWLIST,
    );
    let details: BTreeSet<&str> = result
        .violations
        .iter()
        .map(|f| f.detail.as_str())
        .collect();
    assert_eq!(
        details,
        BTreeSet::from([
            "imports getaddrinfo",
            "imports getnameinfo",
            "imports res_nquery"
        ])
    );
}

/// What the MediaPipe runtime put into the 0.1 bundle must never pass again.
#[test]
fn test_mediapipe_like_bundle_fails() {
    let dir = TempDir::new().unwrap();
    let libmediapipe = [
        pe(&["KERNEL32.dll", "WININET.dll"], &[], false),
        b"https://play.googleapis.com/log\0".to_vec(),
    ]
    .concat();
    let root = bundle(
        dir.path(),
        &[("mediapipe/tasks/c/libmediapipe.dll", libmediapipe)],
    );
    let result = scan_bundle(&root, BUNDLE_ALLOWLIST);
    assert!(!result.ok());
    let relative = "mediapipe/tasks/c/libmediapipe.dll".to_string();
    assert_eq!(
        summary(&result),
        BTreeSet::from([
            (relative.clone(), "telemetry-endpoint"),
            (relative, "network-library")
        ])
    );
}

#[test]
fn test_unknown_or_unexpected_networking_fails() {
    let dir = TempDir::new().unwrap();
    let root = bundle(
        dir.path(),
        &[
            // An allowed file that gains a new networking dependency.
            (
                "eye-tracker.exe",
                pe(&["WS2_32.dll", "DNSAPI.dll"], &[], false),
            ),
            // Networking where none is expected.
            ("lib/somelib.dll", pe(&["WS2_32.dll"], &[], false)),
            (
                "lib/libsomething.so.1",
                elf(&["libc.so.6"], &["gethostbyname"], &[], true, false),
            ),
            (
                "lib/libfetch.so",
                elf(&["libcurl-gnutls.so.4"], &[], &[], true, false),
            ),
            (
                "lib/broken.dll",
                [mz_without_pe(), b"XXXX".to_vec()].concat(),
            ),
            (
                "data/config.json",
                b"{\"endpoint\": \"https://api.segment.io/v1\"}".to_vec(),
            ),
        ],
    );
    let result = scan_bundle(&root, BUNDLE_ALLOWLIST);
    let details: BTreeSet<(String, &str, String)> = result
        .violations
        .iter()
        .map(|f| (file_name(&f.relative), f.rule, f.detail.clone()))
        .collect();
    let expected = [
        ("eye-tracker.exe", "network-library", "links DNSAPI.dll"),
        ("somelib.dll", "network-library", "links WS2_32.dll"),
        (
            "libsomething.so.1",
            "network-symbol",
            "imports gethostbyname",
        ),
        (
            "libfetch.so",
            "network-library",
            "links libcurl-gnutls.so.4",
        ),
        (
            "broken.dll",
            "unreadable-binary",
            "cannot be verified: MZ file without a PE header",
        ),
        (
            "config.json",
            "telemetry-endpoint",
            "contains the telemetry marker 'api.segment.io'",
        ),
    ];
    assert_eq!(
        details,
        expected
            .map(|(a, b, c)| (a.to_string(), b, c.to_string()))
            .into()
    );
    // The allowed part of eye-tracker.exe is still reported as allowed.
    let allowed: Vec<&str> = result.allowed.iter().map(|f| f.detail.as_str()).collect();
    assert_eq!(allowed, ["links WS2_32.dll"]);
}

#[cfg(unix)]
#[test]
fn test_symlinks_are_not_followed() {
    let dir = TempDir::new().unwrap();
    let root = bundle(
        dir.path(),
        &[("Versions/A/QtCore", pe(&["KERNEL32.dll"], &[], false))],
    );
    let outside = dir.path().join("outside.dll");
    fs::write(&outside, pe(&["WININET.dll"], &[], false)).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("linked.dll")).unwrap();
    std::os::unix::fs::symlink("A", root.join("Versions").join("Current")).unwrap();
    let result = scan_bundle(&root, BUNDLE_ALLOWLIST);
    assert!(result.ok());
    assert_eq!(result.files, 1);
}

#[test]
fn test_bundle_command_line() {
    let dir = TempDir::new().unwrap();
    let good = bundle(
        &dir.path().join("good"),
        &[("eye-tracker.exe", pe(&["WS2_32.dll"], &[], false))],
    );
    let bad = bundle(
        &dir.path().join("bad"),
        &[("lib/x.dll", pe(&["WININET.dll"], &[], false))],
    );

    let report = run(&bundle_args(&good, false), None);
    assert_eq!(report.code, 0);
    assert!(
        report
            .out
            .contains("eye-tracker.exe: allowed WS2_32.dll: Rust's standard library"),
        "{}",
        report.out
    );
    assert!(
        report.out.contains("OK: 1 native binaries in 1 files"),
        "{}",
        report.out
    );

    let report = run(&bundle_args(&bad, true), None);
    assert_eq!(report.code, 1);
    assert_eq!(
        report.out,
        "lib/x.dll: network-library: links WININET.dll\n"
    );
    assert!(report.err.contains("FAILED: 1 problem(s)"));

    // Under GitHub Actions: a notice per allow-listed file, an error per problem
    // (bundled files have no source line to point at); the summary stays plain.
    let report = run(&bundle_args(&good, false), Some("true"));
    assert_eq!(report.code, 0);
    let lines: Vec<&str> = report.out.lines().collect();
    let [notice, summary] = lines[..] else {
        panic!("{}", report.out)
    };
    assert!(notice.starts_with(
        "::notice title=Privacy check (bundle)::eye-tracker.exe: allowed WS2_32.dll: "
    ));
    assert!(summary.starts_with("check_privacy: OK: 1 native binaries"));
    let report = run(&bundle_args(&bad, false), Some("true"));
    assert_eq!(report.code, 1);
    assert_eq!(
        report.out,
        "::error title=Privacy check (bundle)::lib/x.dll: network-library: links WININET.dll\n"
    );
    assert!(
        report
            .err
            .starts_with("check_privacy: FAILED: 1 problem(s) in the bundle ")
    );

    assert_eq!(
        run(&bundle_args(&dir.path().join("missing"), false), None).code,
        2
    );
}

#[test]
fn test_bundle_scan_of_a_single_file() {
    let dir = TempDir::new().unwrap();
    let library = dir.path().join("libmediapipe.so");
    fs::write(
        &library,
        [
            elf(&["libc.so.6"], &[], &[], true, false),
            b"clearcut".to_vec(),
        ]
        .concat(),
    )
    .unwrap();
    let result = scan_bundle(&library, BUNDLE_ALLOWLIST);
    assert_eq!(
        summary(&result),
        BTreeSet::from([("libmediapipe.so".to_string(), "telemetry-endpoint")])
    );
}

#[test]
fn test_package_source_contains_no_telemetry_markers() {
    // The app's sources end up in the binaries the bundle check reads: a comment naming
    // a telemetry endpoint would fail the release, so fail here first.
    let sources: Vec<PathBuf> = walkdir::WalkDir::new(default_target())
        .into_iter()
        .filter_map(Result::ok)
        .map(|e| e.into_path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    assert!(!sources.is_empty());
    for path in sources {
        assert!(
            telemetry_markers(&fs::read(&path).unwrap()).is_empty(),
            "{}",
            path.display()
        );
    }
}
