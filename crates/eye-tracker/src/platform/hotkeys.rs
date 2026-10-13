//! System-wide keyboard shortcuts.
//!
//! The module has two halves:
//!
//! * **Parsing and formatting** ([`parse_hotkey`], [`format_hotkey`]) are pure and work
//!   everywhere. Hotkeys are stored in the settings file as short, case-insensitive strings such
//!   as `"ctrl+alt+p"`.
//! * **Registration** is done by a [`HotkeyManager`]. [`create_hotkey_manager`] picks the native
//!   implementation for the running system:
//!
//!   | System    | Backend                                                                       |
//!   |-----------|-------------------------------------------------------------------------------|
//!   | Windows   | `RegisterHotKey` on a dedicated message-loop thread                           |
//!   | macOS     | Carbon `RegisterEventHotKey` (no permission prompt needed); the events are     |
//!   |           | delivered by the Cocoa run loop that the UI already spins                     |
//!   | Linux/X11 | passive key grabs through x11rb on a dedicated thread                         |
//!   | Wayland   | unsupported: bind the `ctl` commands in the desktop's own keyboard settings   |
//!   |           | instead (the manager's note names them as this copy is run)                   |
//!
//! A global hotkey swallows its key system-wide, so combinations that the user types text with
//! are refused ([`HotkeyManager::layout_conflict`]): on Windows AltGr arrives as Ctrl+Alt
//! (Ctrl+Alt+C is 'ć' on a Polish keyboard), on macOS Option alone types characters, and on X11
//! a keyboard-layout switch such as Alt+Shift makes every hotkey containing both keys impossible
//! to press. Keys follow the keyboard layout on every system, and
//! [`HotkeyManager::last_error`] explains any refusal in a user-presentable way.
//!
//! Callbacks may be invoked on *any* thread (the hotkey thread on Windows/X11, the main thread
//! on macOS). They must return quickly and marshal work to the UI thread themselves, for example
//! by sending on a channel.
//!
//! Only the registered combinations are ever observed. Nothing here installs a keyboard hook or
//! sees any other key press.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle, ThreadId};
use std::time::Duration;

use parking_lot::{Condvar, Mutex, ReentrantMutex};

use crate::config::Platform;

/// A hotkey callback. Runs on the hotkey thread (Windows, X11) or the main thread (macOS).
pub type HotkeyCallback = Arc<dyn Fn() + Send + Sync>;

/// Canonical modifier names. `meta` is the Windows key, Command on macOS and Super on Linux.
pub const MODIFIERS: [&str; 4] = ["ctrl", "alt", "shift", "meta"];

/// Every key name accepted by [`Hotkey`] (canonical, lower case).
#[rustfmt::skip]
pub const KEYS: [&str; 86] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r",
    "s", "t", "u", "v", "w", "x", "y", "z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14",
    "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24",
    "space", "enter", "tab", "escape", "backspace", "insert", "delete", "home", "end", "pageup",
    "pagedown", "left", "right", "up", "down",
    "minus", "equal", "bracketleft", "bracketright", "backslash", "semicolon", "quote", "grave",
    "comma", "period", "slash",
];

// Canonical punctuation names follow the X11 keysym names; the value is the character the key
// produces on a US layout.
const PUNCTUATION: [(&str, char); 11] = [
    ("minus", '-'),
    ("equal", '='),
    ("bracketleft", '['),
    ("bracketright", ']'),
    ("backslash", '\\'),
    ("semicolon", ';'),
    ("quote", '\''),
    ("grave", '`'),
    ("comma", ','),
    ("period", '.'),
    ("slash", '/'),
];

/// The US character of punctuation key `key`, or `None` for other keys.
fn punctuation(key: &str) -> Option<char> {
    PUNCTUATION.iter().find(|(k, _)| *k == key).map(|(_, c)| *c)
}

/// The canonical (`'static`) spelling of key `key`, if it is one of [`KEYS`].
fn canonical_key(key: &str) -> Option<&'static str> {
    KEYS.iter().copied().find(|k| *k == key)
}

/// Keys that produce text or drive text editing. Grabbing them with Shift as the only modifier
/// would make it impossible, system-wide, to type capitals, use Shift+Tab or extend a selection
/// (Shift+arrows/Home/End/PgUp/PgDn, which also scroll terminals).
fn is_typing_key(key: &str) -> bool {
    let single = key.chars().count() == 1; // letters and digits
    single
        || punctuation(key).is_some()
        || matches!(
            key,
            "space"
                | "enter"
                | "tab"
                | "backspace"
                | "delete"
                | "insert"
                | "left"
                | "right"
                | "up"
                | "down"
                | "home"
                | "end"
                | "pageup"
                | "pagedown"
        )
}

const MODIFIER_SYMBOLS: [char; 4] = ['⌃', '⌥', '⇧', '⌘'];

fn modifier_alias(token: &str) -> Option<&'static str> {
    Some(match token {
        "ctrl" | "control" | "ctl" | "strg" | "⌃" => "ctrl",
        "alt" | "option" | "opt" | "⌥" => "alt",
        "shift" | "⇧" => "shift",
        "meta" | "cmd" | "command" | "win" | "windows" | "super" | "⌘" => "meta",
        _ => return None,
    })
}

/// Qt portable names, common spellings and the macOS menu glyphs (so that
/// `format_hotkey(.., Some(true))` round-trips through [`parse_hotkey`]), plus the punctuation
/// characters themselves.
fn key_alias(token: &str) -> Option<&'static str> {
    Some(match token {
        "esc" | "⎋" => "escape",
        "return" | "↩" | "⏎" | "↵" => "enter",
        "⇥" => "tab",
        "spacebar" | "␣" => "space",
        "bksp" | "⌫" => "backspace",
        "ins" | "help" => "insert",
        "del" | "forwarddelete" | "⌦" => "delete",
        "↖" => "home",
        "↘" => "end",
        "pgup" | "prior" | "⇞" => "pageup",
        "pgdn" | "pgdown" | "next" | "⇟" => "pagedown",
        "arrowleft" | "leftarrow" | "←" => "left",
        "arrowright" | "rightarrow" | "→" => "right",
        "arrowup" | "uparrow" | "↑" => "up",
        "arrowdown" | "downarrow" | "↓" => "down",
        "dash" | "hyphen" => "minus",
        "equals" => "equal",
        "leftbracket" | "lbracket" => "bracketleft",
        "rightbracket" | "rbracket" => "bracketright",
        "apostrophe" => "quote",
        "backquote" | "backtick" => "grave",
        "dot" | "fullstop" => "period",
        "forwardslash" => "slash",
        _ => {
            let mut chars = token.chars();
            let (Some(ch), None) = (chars.next(), chars.next()) else {
                return None;
            };
            return PUNCTUATION.iter().find(|(_, c)| *c == ch).map(|(k, _)| *k);
        }
    })
}

/// Display name of a canonical key (`"pageup"` → `"PgUp"`, `"t"` → `"T"`).
fn key_label(key: &str, macos: bool) -> String {
    if let Some(ch) = punctuation(key) {
        return ch.to_string();
    }
    let mac = match key {
        "enter" => "↩",
        "tab" => "⇥",
        "escape" => "⎋",
        "backspace" => "⌫",
        "insert" => "Help",
        "delete" => "⌦",
        "home" => "↖",
        "end" => "↘",
        "pageup" => "⇞",
        "pagedown" => "⇟",
        "left" => "←",
        "right" => "→",
        "up" => "↑",
        "down" => "↓",
        _ => "",
    };
    if macos && !mac.is_empty() {
        return mac.to_string();
    }
    let label = match key {
        "space" => "Space",
        "enter" => "Enter",
        "tab" => "Tab",
        "escape" => "Esc",
        "backspace" => "Backspace",
        "insert" => "Insert",
        "delete" => "Delete",
        "home" => "Home",
        "end" => "End",
        "pageup" => "PgUp",
        "pagedown" => "PgDn",
        "left" => "Left",
        "right" => "Right",
        "up" => "Up",
        "down" => "Down",
        _ => return key.to_uppercase(),
    };
    label.to_string()
}

fn mac_modifier_symbol(modifier: &str) -> char {
    match modifier {
        "ctrl" => '⌃',
        "alt" => '⌥',
        "shift" => '⇧',
        _ => '⌘',
    }
}

/// Why a hotkey string or [`Hotkey`] is invalid (Python's `ValueError`). The message is
/// user-presentable.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct HotkeyError(pub String);

fn invalid(message: impl Into<String>) -> HotkeyError {
    HotkeyError(message.into())
}

/// `text` quoted as Python's `repr()` shows a string (the messages keep the Python wording).
fn py_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() => out.push_str(&format!("\\x{:02x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// --------------------------------------------------------------------------- model

/// A key plus at least one modifier.
///
/// `modifiers` is a subset of [`MODIFIERS`]; `key` is one of [`KEYS`]. `to_string()` gives the
/// canonical settings form, e.g. `"ctrl+alt+p"`. Construction validates both and fails with
/// [`HotkeyError`] otherwise.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Hotkey {
    modifiers: BTreeSet<&'static str>,
    key: &'static str,
}

impl Hotkey {
    /// Accepts any iterable of modifier names but always stores a set, so that equality and
    /// hashing do not depend on how the value was built.
    pub fn new<I, S>(modifiers: I, key: &str) -> Result<Hotkey, HotkeyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let names: BTreeSet<String> = modifiers
            .into_iter()
            .map(|m| m.as_ref().to_string())
            .collect();
        let unknown: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|m| !MODIFIERS.contains(m))
            .collect();
        if !unknown.is_empty() {
            return Err(invalid(format!(
                "unknown modifier(s): {}",
                unknown.join(", ")
            )));
        }
        let mods: BTreeSet<&'static str> = MODIFIERS
            .iter()
            .copied()
            .filter(|m| names.contains(*m))
            .collect();
        let Some(key) = canonical_key(key) else {
            return Err(invalid(format!("unsupported key: {}", py_repr(key))));
        };
        if mods.is_empty() {
            return Err(invalid(
                "a global hotkey needs at least one modifier (Ctrl, Alt, Shift, Meta)",
            ));
        }
        if mods.len() == 1 && mods.contains("shift") && is_typing_key(key) {
            return Err(invalid(
                "Shift alone is not enough for a typing or text-navigation key; \
                 add Ctrl, Alt or Meta",
            ));
        }
        Ok(Hotkey {
            modifiers: mods,
            key,
        })
    }

    /// The modifiers (a subset of [`MODIFIERS`]).
    pub fn modifiers(&self) -> &BTreeSet<&'static str> {
        &self.modifiers
    }

    /// The key (one of [`KEYS`]).
    pub fn key(&self) -> &'static str {
        self.key
    }

    /// Modifiers in canonical order (ctrl, alt, shift, meta).
    pub fn ordered_modifiers(&self) -> Vec<&'static str> {
        MODIFIERS
            .iter()
            .copied()
            .filter(|m| self.modifiers.contains(m))
            .collect()
    }

    fn has(&self, modifier: &str) -> bool {
        self.modifiers.contains(modifier)
    }
}

impl fmt::Display for Hotkey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = self.ordered_modifiers();
        parts.push(self.key);
        f.write_str(&parts.join("+"))
    }
}

fn normalise_token(token: &str) -> String {
    let token = token.trim().to_lowercase();
    // Single characters are keys in their own right ("-" is the minus key), so only
    // multi-character words have their inner separators removed ("Page Up" → "pageup").
    if token.chars().count() > 1 {
        return token
            .chars()
            .filter(|c| !(c.is_whitespace() || *c == '_' || *c == '-'))
            .collect();
    }
    token
}

/// Split `"Ctrl + Alt+P"` or `"⌃⌥P"` into raw tokens (the last one is the key).
fn split_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for raw in text.split('+') {
        let mut part = raw.trim();
        let mut peeled = false;
        // macOS style glyphs may be glued to each other and to the key: "⌃⌥⇧P".
        while part.chars().count() > 1 {
            let first = part.chars().next().unwrap_or_default();
            if !MODIFIER_SYMBOLS.contains(&first) {
                break;
            }
            tokens.push(first.to_string());
            part = part[first.len_utf8()..].trim();
            peeled = true;
        }
        if !part.is_empty() || !peeled {
            tokens.push(part.to_string());
        }
    }
    tokens
}

/// Parse a user-facing hotkey string.
///
/// Accepts `"Ctrl+Alt+P"`, `"ctrl + alt + p"`, `"cmd+shift+g"` and macOS glyph notation such as
/// `"⌃⌥P"`. Matching is case-insensitive and tolerant of spaces. Aliases:
/// `cmd`/`command`/`win`/`windows`/`super` → `meta`, `option`/`opt` → `alt`,
/// `control`/`ctl` → `ctrl`; key aliases include `esc`, `return`, `del`, `ins`, `pgup`/`pgdn`
/// and punctuation characters.
///
/// Note for callers holding a Qt-style portable key sequence: on macOS those call the Command
/// key `Ctrl` and the Control key `Meta`; swap them before parsing.
///
/// Fails if the text is empty, uses an unknown modifier or key, has no modifier, or consists of
/// modifiers only.
pub fn parse_hotkey(text: &str) -> Result<Hotkey, HotkeyError> {
    let stripped = text.trim();
    if stripped.is_empty() {
        return Err(invalid("hotkey is empty"));
    }
    if stripped == "+"
        || (stripped.ends_with('+') && stripped[..stripped.len() - 1].trim_end().ends_with('+'))
    {
        // "Ctrl++": '+' is Shift+'=' on most layouts, so it has no key of its own.
        return Err(invalid(format!(
            "the '+' key cannot be used in hotkey {}; use '=' instead",
            py_repr(text)
        )));
    }
    let mut tokens = split_tokens(stripped);
    let key_token = tokens.pop().unwrap_or_default();

    let mut modifiers = BTreeSet::new();
    for raw in &tokens {
        let token = normalise_token(raw);
        if token.is_empty() {
            return Err(invalid(format!(
                "malformed hotkey {}: empty part",
                py_repr(text)
            )));
        }
        let Some(modifier) = modifier_alias(&token) else {
            return Err(invalid(format!(
                "unknown modifier {} in hotkey {}",
                py_repr(raw.trim()),
                py_repr(text)
            )));
        };
        modifiers.insert(modifier);
    }

    let key = normalise_token(&key_token);
    if key.is_empty() {
        return Err(invalid(format!(
            "malformed hotkey {}: missing key",
            py_repr(text)
        )));
    }
    if modifier_alias(&key).is_some() {
        return Err(invalid(format!(
            "hotkey {} needs a key in addition to its modifiers",
            py_repr(text)
        )));
    }
    let key = key_alias(&key).unwrap_or(&key);
    if canonical_key(key).is_none() {
        return Err(invalid(format!(
            "unsupported key {} in hotkey {}",
            py_repr(key_token.trim()),
            py_repr(text)
        )));
    }
    if modifiers.is_empty() {
        return Err(invalid(format!(
            "hotkey {} needs at least one modifier (Ctrl, Alt, Shift or Meta)",
            py_repr(text)
        )));
    }
    Hotkey::new(modifiers, key)
}

/// Human-readable label: `"Ctrl+Alt+P"` (Windows/Linux) or `"⌃⌥P"` (macOS).
///
/// `macos = None` follows the running system. The meta key is shown as `Win` on Windows and
/// `Super` elsewhere. The result is always accepted by [`parse_hotkey`].
pub fn format_hotkey(hk: &Hotkey, macos: Option<bool>) -> String {
    format_hotkey_on(hk, macos, Platform::CURRENT)
}

/// [`format_hotkey`] as it reads on `platform`.
fn format_hotkey_on(hk: &Hotkey, macos: Option<bool>, platform: Platform) -> String {
    if macos.unwrap_or(platform == Platform::MacOs) {
        let mut label: String = hk
            .ordered_modifiers()
            .into_iter()
            .map(mac_modifier_symbol)
            .collect();
        label.push_str(&key_label(hk.key, true));
        return label;
    }
    let mut parts = modifier_labels(hk.modifiers.iter().copied(), platform);
    parts.push(key_label(hk.key, false));
    parts.join("+")
}

/// Windows/Linux labels of `modifiers` in canonical order (`["Ctrl", "Alt"]`).
fn modifier_labels<'a>(
    modifiers: impl IntoIterator<Item = &'a str>,
    platform: Platform,
) -> Vec<String> {
    let wanted: HashSet<&str> = modifiers.into_iter().collect();
    let meta = if platform == Platform::Windows {
        "Win"
    } else {
        "Super"
    };
    MODIFIERS
        .iter()
        .filter(|m| wanted.contains(*m))
        .map(|m| {
            match *m {
                "ctrl" => "Ctrl",
                "alt" => "Alt",
                "shift" => "Shift",
                _ => meta,
            }
            .to_string()
        })
        .collect()
}

/// `text` (what a key types) quoted for a user-facing message.
///
/// Visible characters are quoted as they are (`'ć'`). Blank and invisible ones, such as the
/// no-break space that Option+Space types on a Mac, and lone combining marks (which would sit on
/// the quote sign) are named instead (`no-break space (U+00A0)`): an escape like `'\xa0'` means
/// nothing to users.
fn describe_typed(text: &str) -> String {
    let text = compose(text); // 'a' + combining acute → 'á'
    if !text.is_empty() && text.chars().all(is_visible) {
        let quote = if text.contains('\'') { '"' } else { '\'' };
        return format!("{quote}{text}{quote}");
    }
    let names: Vec<String> = text
        .chars()
        .map(|ch| {
            let code = format!("U+{:04X}", u32::from(ch));
            match char_name(ch) {
                Some(name) => format!("{name} ({code})"),
                None => code,
            }
        })
        .collect();
    if names.is_empty() {
        "nothing".to_string()
    } else {
        names.join(" + ")
    }
}

// shortcut: the workspace has no Unicode normalisation or character-name tables, so `compose`,
// `char_name` and `is_visible` cover what keyboard layouts type (Latin letters with the common
// accents, the blank and invisible characters, the frequent combining marks); upgrade to the
// `unicode-normalization` / `unicode_names2` / `unicode-general-category` crates if a message
// ever shows a bare code point for a character users would recognise.

/// Canonical composition (NFC) of a base letter followed by a common combining accent.
fn compose(text: &str) -> String {
    const TABLE: [(char, &str, &str); 8] = [
        ('\u{300}', "aeiouAEIOU", "àèìòùÀÈÌÒÙ"),
        ('\u{301}', "aeiouyAEIOUYcCnNsSzZ", "áéíóúýÁÉÍÓÚÝćĆńŃśŚźŹ"),
        ('\u{302}', "aeiouAEIOU", "âêîôûÂÊÎÔÛ"),
        ('\u{303}', "anoANO", "ãñõÃÑÕ"),
        ('\u{308}', "aeiouyAEIOUY", "äëïöüÿÄËÏÖÜŸ"),
        ('\u{30A}', "aAuU", "åÅůŮ"),
        ('\u{30C}', "cCdDeEnNrRsStTzZ", "čČďĎěĚňŇřŘšŠťŤžŽ"),
        ('\u{327}', "cCsStT", "çÇşŞţŢ"),
    ];
    let mut out: Vec<char> = Vec::new();
    for ch in text.chars() {
        let composed = out.last().and_then(|base| {
            let (_, bases, results) = TABLE.iter().find(|(mark, _, _)| *mark == ch)?;
            let index = bases.chars().position(|b| b == *base)?;
            results.chars().nth(index)
        });
        match composed {
            Some(c) => {
                out.pop();
                out.push(c);
            }
            None => out.push(ch),
        }
    }
    out.into_iter().collect()
}

/// The lower-case Unicode name of `ch` where it is known here (see the shortcut note above).
fn char_name(ch: char) -> Option<String> {
    const DIGITS: [&str; 10] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    if ch.is_ascii_lowercase() {
        return Some(format!("latin small letter {ch}"));
    }
    if ch.is_ascii_uppercase() {
        return Some(format!("latin capital letter {}", ch.to_ascii_lowercase()));
    }
    if let Some(digit) = ch.to_digit(10).filter(|_| ch.is_ascii_digit()) {
        return Some(format!("digit {}", DIGITS[digit as usize]));
    }
    let name = match ch {
        ' ' => "space",
        '\u{A0}' => "no-break space",
        '\u{AD}' => "soft hyphen",
        '\u{1680}' => "ogham space mark",
        '\u{2000}' => "en quad",
        '\u{2001}' => "em quad",
        '\u{2002}' => "en space",
        '\u{2003}' => "em space",
        '\u{2004}' => "three-per-em space",
        '\u{2005}' => "four-per-em space",
        '\u{2006}' => "six-per-em space",
        '\u{2007}' => "figure space",
        '\u{2008}' => "punctuation space",
        '\u{2009}' => "thin space",
        '\u{200A}' => "hair space",
        '\u{200B}' => "zero width space",
        '\u{200C}' => "zero width non-joiner",
        '\u{200D}' => "zero width joiner",
        '\u{200E}' => "left-to-right mark",
        '\u{200F}' => "right-to-left mark",
        '\u{2028}' => "line separator",
        '\u{2029}' => "paragraph separator",
        '\u{202F}' => "narrow no-break space",
        '\u{205F}' => "medium mathematical space",
        '\u{2060}' => "word joiner",
        '\u{3000}' => "ideographic space",
        '\u{FEFF}' => "zero width no-break space",
        '\u{300}' => "combining grave accent",
        '\u{301}' => "combining acute accent",
        '\u{302}' => "combining circumflex accent",
        '\u{303}' => "combining tilde",
        '\u{304}' => "combining macron",
        '\u{306}' => "combining breve",
        '\u{307}' => "combining dot above",
        '\u{308}' => "combining diaeresis",
        '\u{30A}' => "combining ring above",
        '\u{30B}' => "combining double acute accent",
        '\u{30C}' => "combining caron",
        '\u{327}' => "combining cedilla",
        '\u{328}' => "combining ogonek",
        _ => return None,
    };
    Some(name.to_string())
}

/// Whether `ch` shows up as a glyph of its own when printed between quotes.
fn is_visible(ch: char) -> bool {
    let code = u32::from(ch);
    let format = matches!(code,
        0xAD | 0x600..=0x605 | 0x61C | 0x6DD | 0x70F | 0x180E | 0x200B..=0x200F
        | 0x202A..=0x202E | 0x2060..=0x2064 | 0x2066..=0x206F | 0xFEFF | 0xFFF9..=0xFFFB
        | 0xE0001 | 0xE0020..=0xE007F);
    let private_use = matches!(code, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD);
    let combining = matches!(code,
        0x300..=0x36F | 0x483..=0x489 | 0x591..=0x5BD | 0x610..=0x61A | 0x64B..=0x65F | 0x670
        | 0x6D6..=0x6DC | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF
        | 0x302A..=0x302F | 0x3099..=0x309A | 0xFE00..=0xFE0F | 0xFE20..=0xFE2F);
    !(ch.is_control() || ch.is_whitespace() || format || private_use || combining)
}

/// What [`HotkeyManager`] methods accept as a hotkey: a [`Hotkey`] or a string to parse.
#[derive(Debug, Clone)]
pub enum HotkeyArg<'a> {
    Hotkey(Hotkey),
    Text(&'a str),
}

impl HotkeyArg<'_> {
    /// The hotkey itself, or the parsed string.
    fn parse(self) -> Result<Hotkey, HotkeyError> {
        match self {
            HotkeyArg::Hotkey(hk) => Ok(hk),
            HotkeyArg::Text(text) => parse_hotkey(text),
        }
    }
}

impl From<Hotkey> for HotkeyArg<'_> {
    fn from(hk: Hotkey) -> Self {
        HotkeyArg::Hotkey(hk)
    }
}

impl From<&Hotkey> for HotkeyArg<'_> {
    fn from(hk: &Hotkey) -> Self {
        HotkeyArg::Hotkey(hk.clone())
    }
}

impl<'a> From<&'a str> for HotkeyArg<'a> {
    fn from(text: &'a str) -> Self {
        HotkeyArg::Text(text)
    }
}

impl<'a> From<&'a String> for HotkeyArg<'a> {
    fn from(text: &'a String) -> Self {
        HotkeyArg::Text(text)
    }
}

/// The message of a caught panic (Python's exception text).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "panic".to_string()
    }
}

// ------------------------------------------------------------------------- managers

/// Registers global hotkeys.
///
/// Native implementations share these semantics:
///
/// * [`register`](Self::register) starts the backend on demand and returns whether the OS
///   accepted the combination (`false` if another application owns it, or if it would swallow a
///   character the user types, see [`layout_conflict`](Self::layout_conflict)). Re-registering a
///   name replaces its previous hotkey; registering a combination that another name in this
///   manager already uses returns `false`. [`last_error`](Self::last_error) says why a
///   registration failed.
/// * [`start`](Self::start) is optional (it pre-warms the backend). [`stop`](Self::stop)
///   releases every hotkey, ends the backend and is idempotent; a later `register` starts it
///   again.
/// * Callbacks run on an arbitrary thread; panics they raise are logged.
///
/// [`UnsupportedHotkeyManager`] is the no-op used where hotkeys cannot work.
pub trait HotkeyManager: Send + Sync {
    /// Whether hotkeys work on this system.
    fn supported(&self) -> bool;

    /// Backend name: `"none"`, `"win32"`, `"carbon"` or `"x11"`.
    fn name(&self) -> &'static str;

    /// Human-readable remark on availability (why hotkeys are unavailable or limited here);
    /// `None` when fully supported. Shown by the UI and doctor.
    fn note(&self) -> Option<String>;

    /// Bind `hotkey` to `callback` under `name`. Returns success.
    fn register(&self, name: &str, hotkey: HotkeyArg<'_>, callback: HotkeyCallback) -> bool;

    /// Release the hotkey registered under `name`. Returns whether one existed.
    fn unregister(&self, name: &str) -> bool;

    /// Release every hotkey (the backend keeps running).
    fn unregister_all(&self);

    /// Registrations that are currently active, by name.
    ///
    /// A hotkey that stopped working (for example after a keyboard layout change removed its
    /// key) is left out until it works again; see [`last_error`](Self::last_error).
    fn registered(&self) -> HashMap<String, Hotkey>;

    /// Why the latest [`register`](Self::register) of `name` failed, or why it stopped working.
    ///
    /// The text is a short, user-presentable sentence such as `"Ctrl+Alt+C is AltGr+C and types
    /// 'ć' on the Polish (Programmers) keyboard layout"`. `None` if the hotkey is active or
    /// `name` was never registered.
    fn last_error(&self, name: &str) -> Option<String>;

    /// Why `hotkey` clashes with how the user types, or `None`.
    ///
    /// On Windows, Ctrl+Alt is AltGr: a global Ctrl+Alt+C would eat every 'ć' typed on a Polish
    /// keyboard. On macOS, Option (+Shift) alone types characters. On X11, a keyboard-layout
    /// switch such as Alt+Shift (XKB option `grp:alt_shift_toggle`) means a hotkey containing
    /// both keys can never be pressed: the layout changes instead and the rest of the chord
    /// reaches other applications (likewise when an XKB option makes Win or Alt a layout or
    /// AltGr key). The native managers check this on every installed (Windows), the current
    /// (macOS) or the configured (X11) keyboard layout and refuse such registrations; this
    /// method lets settings UIs and diagnostics warn before registering. It never changes the
    /// keyboard state or the active layout.
    ///
    /// Fails if `hotkey` is a string that is not a valid hotkey.
    fn layout_conflict(&self, hotkey: HotkeyArg<'_>) -> Result<Option<String>, HotkeyError>;

    /// Start the backend early (optional; [`register`](Self::register) starts it on demand).
    fn start(&self);

    /// Release every hotkey and stop the backend. Safe to call repeatedly.
    fn stop(&self);
}

/// The *unsupported* manager: registers nothing (Python's `HotkeyManager` base class).
pub struct UnsupportedHotkeyManager {
    note: Option<String>,
    // name → why its last registration failed.
    errors: Mutex<HashMap<String, String>>,
}

impl UnsupportedHotkeyManager {
    pub fn new(note: Option<String>) -> UnsupportedHotkeyManager {
        UnsupportedHotkeyManager {
            note,
            errors: Mutex::new(HashMap::new()),
        }
    }
}

impl HotkeyManager for UnsupportedHotkeyManager {
    fn supported(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "none"
    }

    fn note(&self) -> Option<String> {
        self.note.clone()
    }

    fn register(&self, name: &str, _hotkey: HotkeyArg<'_>, _callback: HotkeyCallback) -> bool {
        log::debug!("Global hotkeys unsupported here; {name:?} not registered");
        let message = self
            .note
            .clone()
            .unwrap_or_else(|| "global hotkeys are not supported on this system".to_string());
        self.errors.lock().insert(name.to_string(), message);
        false
    }

    fn unregister(&self, _name: &str) -> bool {
        false
    }

    fn unregister_all(&self) {}

    fn registered(&self) -> HashMap<String, Hotkey> {
        HashMap::new()
    }

    fn last_error(&self, name: &str) -> Option<String> {
        self.errors.lock().get(name).cloned()
    }

    fn layout_conflict(&self, hotkey: HotkeyArg<'_>) -> Result<Option<String>, HotkeyError> {
        hotkey.parse()?;
        Ok(None)
    }

    fn start(&self) {}

    fn stop(&self) {}
}

struct Binding {
    id: u32,
    name: String,
    hotkey: Hotkey,
    callback: Mutex<HotkeyCallback>,
    /// Set by a native hook that refused the binding only because of the current keyboard
    /// layout or its options: the binding is kept (inactive) and tried again when the layout
    /// changes, like one whose key vanished later.
    retry_on_layout_change: AtomicBool,
}

impl Binding {
    fn new(id: u32, name: &str, hotkey: Hotkey, callback: HotkeyCallback) -> Binding {
        Binding {
            id,
            name: name.to_string(),
            hotkey,
            callback: Mutex::new(callback),
            retry_on_layout_change: AtomicBool::new(false),
        }
    }
}

/// What `Core::table` guards: only ever short dictionary operations.
#[derive(Default)]
struct Table {
    bindings: HashMap<String, Arc<Binding>>,
    by_id: HashMap<u32, Arc<Binding>>,
    /// Ids of bindings whose OS registration was lost.
    inactive: HashSet<u32>,
    /// name → why its last registration failed or its hotkey stopped working.
    errors: HashMap<String, String>,
    last_id: u32,
}

/// RegisterHotKey accepts application ids 0..0xBFFF.
const MAX_ID: u32 = 0xBFFF;

/// Bookkeeping shared by the native managers (Python's `_NativeHotkeyManager`).
///
/// Two locks keep dispatch deadlock-free: `api_lock` serialises public calls (and may be held
/// while waiting for a backend thread), `table` only ever guards short dictionary operations and
/// is the only lock taken when a hotkey fires.
///
/// A binding whose OS registration is lost later (an X11 or macOS keyboard layout change removed
/// its key) stays in `bindings` so that the next layout change can restore it, but it is marked
/// inactive: `registered` leaves it out and `last_error` explains why. So does a binding that a
/// native hook refused only because of the current layout (`Binding::retry_on_layout_change`):
/// `register` still returns `false` for it.
struct Core {
    name: &'static str,
    note: Mutex<Option<String>>,
    api_lock: ReentrantMutex<()>,
    table: Mutex<Table>,
}

impl Core {
    fn new(name: &'static str, note: Option<String>) -> Core {
        Core {
            name,
            note: Mutex::new(note),
            api_lock: ReentrantMutex::new(()),
            table: Mutex::new(Table::default()),
        }
    }

    fn note(&self) -> Option<String> {
        self.note.lock().clone()
    }

    fn set_note(&self, note: Option<String>) {
        *self.note.lock() = note;
    }

    fn allocate_id(&self) -> Option<u32> {
        let mut table = self.table.lock();
        for _ in 0..MAX_ID {
            table.last_id = table.last_id % MAX_ID + 1;
            if !table.by_id.contains_key(&table.last_id) {
                return Some(table.last_id);
            }
        }
        None // 49k live hotkeys
    }

    fn set_error(&self, name: &str, message: impl Into<String>) {
        self.table
            .lock()
            .errors
            .insert(name.to_string(), message.into());
    }

    /// Log why the OS did not take `binding`, remember it for `last_error`.
    ///
    /// Called by the native hooks (on whichever thread they run); always returns `false` so it
    /// can end a hook with `return self.core.reject(...)`.
    fn reject(&self, binding: &Binding, message: impl Into<String>) -> bool {
        let message = message.into();
        log::warn!("Hotkey for {:?} not registered: {message}", binding.name);
        self.set_error(&binding.name, message);
        false
    }

    /// Record whether a registered binding currently holds its OS registration.
    fn set_active(&self, binding: &Binding, active: bool) {
        let mut table = self.table.lock();
        if active {
            table.inactive.remove(&binding.id);
            table.errors.remove(&binding.name);
        } else if table.by_id.contains_key(&binding.id) {
            table.inactive.insert(binding.id);
        }
    }

    fn dispatch(&self, binding_id: u32) {
        let found = {
            let table = self.table.lock();
            table
                .by_id
                .get(&binding_id)
                .map(|b| (b.clone(), b.callback.lock().clone()))
        };
        let Some((binding, callback)) = found else {
            log::debug!("Ignoring event for unknown hotkey id {binding_id}");
            return;
        };
        log::debug!(
            "Hotkey {} pressed ({})",
            format_hotkey(&binding.hotkey, None),
            binding.name
        );
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| callback())) {
            log::error!(
                "Hotkey callback for {:?} raised: {}",
                binding.name,
                panic_message(payload.as_ref())
            );
        }
    }

    fn registered(&self) -> HashMap<String, Hotkey> {
        let table = self.table.lock();
        table
            .bindings
            .iter()
            .filter(|(_, b)| !table.inactive.contains(&b.id))
            .map(|(name, b)| (name.clone(), b.hotkey.clone()))
            .collect()
    }

    fn last_error(&self, name: &str) -> Option<String> {
        self.table.lock().errors.get(name).cloned()
    }

    /// Bindings in no particular order (a snapshot taken under the table lock).
    fn bindings(&self) -> Vec<Arc<Binding>> {
        self.table.lock().bindings.values().cloned().collect()
    }
}

/// The hooks a native manager implements, and the shared logic built on them.
trait Native: Send + Sync + Sized + 'static {
    fn core(&self) -> &Core;

    fn backend_start(self: &Arc<Self>) -> bool;

    fn backend_stop(self: &Arc<Self>);

    fn native_register(self: &Arc<Self>, binding: &Arc<Binding>) -> bool;

    fn native_unregister(self: &Arc<Self>, binding: &Arc<Binding>);

    /// The native check behind [`HotkeyManager::layout_conflict`].
    fn native_layout_conflict(self: &Arc<Self>, hotkey: &Hotkey) -> Option<String>;

    // ------------------------------------------------------------------ public
    fn register(
        self: &Arc<Self>,
        name: &str,
        hotkey: HotkeyArg<'_>,
        callback: HotkeyCallback,
    ) -> bool {
        let core = self.core();
        core.table.lock().errors.remove(name);
        let hotkey = match hotkey.parse() {
            Ok(hotkey) => hotkey,
            Err(err) => {
                log::warn!("Hotkey for {name:?} not registered: {err}");
                core.set_error(name, err.0);
                return false;
            }
        };
        let label = format_hotkey(&hotkey, None);
        let _api = core.api_lock.lock();
        let (existing, active, clash) = {
            let table = core.table.lock();
            let existing = table.bindings.get(name).cloned();
            let active = existing
                .as_ref()
                .is_some_and(|b| !table.inactive.contains(&b.id));
            let clash = table
                .bindings
                .values()
                .find(|b| b.hotkey == hotkey && b.name != name)
                .map(|b| b.name.clone());
            (existing, active, clash)
        };
        if let Some(existing) = &existing
            && existing.hotkey == hotkey
            && active
        {
            *existing.callback.lock() = callback;
            return true;
        }
        if let Some(clash) = clash {
            log::warn!("Hotkey {label} for {name:?} is already bound to {clash:?}");
            core.set_error(name, format!("{label} is already used for {clash}"));
            return false;
        }
        if let Some(existing) = existing {
            self.remove(&existing);
        }
        if !self.backend_start() {
            log::warn!("Hotkey {label} for {name:?} not registered: backend unavailable");
            let message = core
                .note()
                .unwrap_or_else(|| "the global hotkey service is unavailable".to_string());
            core.set_error(name, message);
            return false;
        }
        let Some(id) = core.allocate_id() else {
            core.set_error(name, "no free hotkey id");
            return false;
        };
        let binding = Arc::new(Binding::new(id, name, hotkey, callback));
        // Publish before the OS registration so a press that arrives immediately afterwards
        // can already be dispatched.
        core.table.lock().by_id.insert(binding.id, binding.clone());
        let ok = catch_unwind(AssertUnwindSafe(|| self.native_register(&binding))).unwrap_or_else(
            |payload| {
                log::error!(
                    "Registering hotkey {label} for {name:?} failed: {}",
                    panic_message(payload.as_ref())
                );
                false
            },
        );
        {
            let mut table = core.table.lock();
            if ok {
                table.bindings.insert(name.to_string(), binding.clone());
                table.errors.remove(name);
            } else {
                if binding.retry_on_layout_change.load(Ordering::SeqCst) {
                    // Kept, inactive, for the layout change that makes it work.
                    table.bindings.insert(name.to_string(), binding.clone());
                    table.inactive.insert(binding.id);
                } else {
                    table.by_id.remove(&binding.id);
                }
                table
                    .errors
                    .entry(name.to_string())
                    .or_insert_with(|| format!("{label} could not be registered"));
            }
        }
        if ok {
            log::info!("Registered hotkey {label} for {name}");
        }
        ok
    }

    fn unregister(self: &Arc<Self>, name: &str) -> bool {
        let _api = self.core().api_lock.lock();
        let binding = self.core().table.lock().bindings.get(name).cloned();
        match binding {
            Some(binding) => {
                self.remove(&binding);
                true
            }
            None => false,
        }
    }

    fn unregister_all(self: &Arc<Self>) {
        let _api = self.core().api_lock.lock();
        for binding in self.core().bindings() {
            self.remove(&binding);
        }
    }

    fn start(self: &Arc<Self>) {
        let _api = self.core().api_lock.lock();
        if !self.backend_start() {
            log::warn!("Global hotkey backend {} could not start", self.core().name);
        }
    }

    fn stop(self: &Arc<Self>) {
        let _api = self.core().api_lock.lock();
        for binding in self.core().bindings() {
            self.remove(&binding);
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.backend_stop())) {
            log::error!(
                "Stopping the hotkey backend failed: {}",
                panic_message(payload.as_ref())
            );
        }
    }

    fn layout_conflict(
        self: &Arc<Self>,
        hotkey: HotkeyArg<'_>,
    ) -> Result<Option<String>, HotkeyError> {
        let hotkey = hotkey.parse()?;
        Ok(self.native_layout_conflict(&hotkey))
    }

    // --------------------------------------------------------------- internals
    fn remove(self: &Arc<Self>, binding: &Arc<Binding>) {
        // Forget the binding first so that a press racing with the unregistration is simply
        // ignored instead of calling a callback the owner already dropped.
        {
            let mut table = self.core().table.lock();
            table.bindings.remove(&binding.name);
            table.by_id.remove(&binding.id);
            table.inactive.remove(&binding.id);
        }
        if let Err(payload) = catch_unwind(AssertUnwindSafe(|| self.native_unregister(binding))) {
            log::error!(
                "Unregistering hotkey {} failed: {}",
                format_hotkey(&binding.hotkey, None),
                panic_message(payload.as_ref())
            );
        }
    }
}

/// Defines the public manager type `$Manager` around the native implementation `$Inner`.
/// Dropping the manager stops it, so its backend thread never outlives it.
macro_rules! native_manager {
    ($(#[$doc:meta])* $Manager:ident($Inner:ty)) => {
        $(#[$doc])*
        pub struct $Manager(pub(in crate::platform::hotkeys) Arc<$Inner>);

        impl HotkeyManager for $Manager {
            fn supported(&self) -> bool {
                true
            }

            fn name(&self) -> &'static str {
                self.0.core().name
            }

            fn note(&self) -> Option<String> {
                self.0.core().note()
            }

            fn register(&self, name: &str, hotkey: HotkeyArg<'_>, callback: HotkeyCallback) -> bool {
                Native::register(&self.0, name, hotkey, callback)
            }

            fn unregister(&self, name: &str) -> bool {
                Native::unregister(&self.0, name)
            }

            fn unregister_all(&self) {
                Native::unregister_all(&self.0)
            }

            fn registered(&self) -> HashMap<String, Hotkey> {
                self.0.core().registered()
            }

            fn last_error(&self, name: &str) -> Option<String> {
                self.0.core().last_error(name)
            }

            fn layout_conflict(
                &self,
                hotkey: HotkeyArg<'_>,
            ) -> Result<Option<String>, HotkeyError> {
                Native::layout_conflict(&self.0, hotkey)
            }

            fn start(&self) {
                Native::start(&self.0)
            }

            fn stop(&self) {
                Native::stop(&self.0)
            }
        }

        impl Drop for $Manager {
            fn drop(&mut self) {
                Native::stop(&self.0);
            }
        }
    };
}

/// A flag threads wait on (Python's `threading.Event`).
#[derive(Default)]
struct Event {
    flag: Mutex<bool>,
    changed: Condvar,
}

impl Event {
    fn set(&self) {
        *self.flag.lock() = true;
        self.changed.notify_all();
    }

    fn clear(&self) {
        *self.flag.lock() = false;
    }

    fn is_set(&self) -> bool {
        *self.flag.lock()
    }

    /// Wait until the flag is set or `timeout` passes; returns the flag.
    fn wait(&self, timeout: Duration) -> bool {
        let mut flag = self.flag.lock();
        if !*flag {
            let _ = self.changed.wait_while_for(&mut flag, |set| !*set, timeout);
        }
        *flag
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CallState {
    Pending,
    Running,
    Finished,
    Cancelled,
}

/// A function to run on the backend thread, with a result the caller waits for.
struct PendingCall {
    function: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    state: Mutex<CallState>,
    done: Event,
    error: Mutex<Option<String>>,
}

impl PendingCall {
    fn new(function: impl FnOnce() + Send + 'static) -> Arc<PendingCall> {
        Arc::new(PendingCall {
            function: Mutex::new(Some(Box::new(function))),
            state: Mutex::new(CallState::Pending),
            done: Event::default(),
            error: Mutex::new(None),
        })
    }

    fn run(&self) {
        {
            let mut state = self.state.lock();
            if *state != CallState::Pending {
                return;
            }
            *state = CallState::Running;
        }
        let function = self.function.lock().take();
        if let Some(function) = function
            && let Err(payload) = catch_unwind(AssertUnwindSafe(function))
        {
            *self.error.lock() = Some(panic_message(payload.as_ref()));
        }
        *self.state.lock() = CallState::Finished;
        self.done.set();
    }

    /// Prevent a call that has not started yet. Returns `true` if it was prevented.
    fn cancel(&self) -> bool {
        {
            let mut state = self.state.lock();
            if *state != CallState::Pending {
                return false;
            }
            *state = CallState::Cancelled;
        }
        self.done.set();
        true
    }

    fn finished(&self) -> bool {
        *self.state.lock() == CallState::Finished
    }
}

/// The backend thread of a [`Threaded`] manager.
struct BackendThread {
    id: ThreadId,
    /// Set when the thread function has returned (after teardown).
    exited: Arc<Event>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl BackendThread {
    fn is_alive(&self) -> bool {
        !self.exited.is_set()
    }

    fn is_current(&self) -> bool {
        thread::current().id() == self.id
    }

    /// Wait up to `timeout` for the thread to end (Python's `Thread.join(timeout)`).
    fn join(&self, timeout: Duration) {
        if self.exited.wait(timeout)
            && let Some(handle) = self.handle.lock().take()
        {
            let _ = handle.join();
        }
    }
}

/// State of a native manager whose OS registrations live on one dedicated thread (Python's
/// `_ThreadedHotkeyManager`).
///
/// Windows hotkeys are bound to the registering thread's message queue and an X11 connection is
/// served by one thread, so every OS call is executed on the backend thread via
/// [`call_on_thread`].
struct Threaded {
    thread: Mutex<Option<Arc<BackendThread>>>,
    calls: Mutex<VecDeque<Arc<PendingCall>>>,
    stopping: Event,
    ready: Event,
    start_ok: AtomicBool,
    call_timeout: Duration,
    join_timeout: Duration,
    /// Tests: run every call on the caller's thread and report the backend as started.
    #[cfg(test)]
    inline: AtomicBool,
}

const CALL_TIMEOUT: Duration = Duration::from_secs(2);
const JOIN_TIMEOUT: Duration = Duration::from_secs(2);

impl Threaded {
    fn new() -> Threaded {
        Threaded {
            thread: Mutex::new(None),
            calls: Mutex::new(VecDeque::new()),
            stopping: Event::default(),
            ready: Event::default(),
            start_ok: AtomicBool::new(false),
            call_timeout: CALL_TIMEOUT,
            join_timeout: JOIN_TIMEOUT,
            #[cfg(test)]
            inline: AtomicBool::new(false),
        }
    }

    fn current_thread(&self) -> Option<Arc<BackendThread>> {
        self.thread.lock().clone()
    }

    fn is_stopping(&self) -> bool {
        self.stopping.is_set()
    }

    fn run_pending_calls(&self) {
        loop {
            let call = self.calls.lock().pop_front();
            match call {
                Some(call) => call.run(),
                None => return,
            }
        }
    }

    fn cancel_pending_calls(&self) {
        loop {
            let call = self.calls.lock().pop_front();
            match call {
                Some(call) => {
                    call.cancel();
                }
                None => return,
            }
        }
    }
}

/// The hooks of a [`Threaded`] manager.
trait ThreadedNative: Native {
    fn threaded(&self) -> &Threaded;

    /// Prepare resources the thread needs (runs on the caller's thread).
    fn before_thread_start(&self) -> bool {
        true
    }

    fn thread_setup(self: &Arc<Self>) -> bool;

    fn thread_loop(self: &Arc<Self>);

    fn thread_teardown(self: &Arc<Self>);

    /// Make the thread run [`Threaded::run_pending_calls`] soon. Returns success.
    fn wake(&self) -> bool;

    fn wake_for_stop(&self) {
        self.wake();
    }
}

/// `backend_start` of a threaded manager.
fn threaded_backend_start<T: ThreadedNative>(this: &Arc<T>) -> bool {
    let threaded = this.threaded();
    #[cfg(test)]
    if threaded.inline.load(Ordering::SeqCst) {
        return true;
    }
    if let Some(thread) = threaded.current_thread()
        && thread.is_alive()
    {
        if !threaded.is_stopping() {
            return threaded.start_ok.load(Ordering::SeqCst);
        }
        // A stop() issued from inside a callback leaves the thread finishing its current
        // message; never run two backend threads at once.
        if thread.is_current() {
            return false;
        }
        thread.join(threaded.join_timeout);
        if thread.is_alive() {
            log::warn!("Previous hotkey thread is still running");
            return false;
        }
    }
    threaded.stopping.clear();
    threaded.ready.clear();
    threaded.start_ok.store(false, Ordering::SeqCst);
    if !this.before_thread_start() {
        return false;
    }
    let exited = Arc::new(Event::default());
    let worker = this.clone();
    let worker_exited = exited.clone();
    let spawned = thread::Builder::new()
        .name(format!("eye-tracker-hotkeys-{}", this.core().name))
        .spawn(move || thread_main(worker, worker_exited));
    let handle = match spawned {
        Ok(handle) => handle,
        Err(err) => {
            log::warn!("Cannot start the hotkey thread: {err}");
            return false;
        }
    };
    *threaded.thread.lock() = Some(Arc::new(BackendThread {
        id: handle.thread().id(),
        exited,
        handle: Mutex::new(Some(handle)),
    }));
    if !threaded.ready.wait(threaded.call_timeout) {
        log::warn!("Hotkey thread did not become ready in time");
        return false;
    }
    threaded.start_ok.load(Ordering::SeqCst)
}

/// `backend_stop` of a threaded manager.
fn threaded_backend_stop<T: ThreadedNative>(this: &Arc<T>) {
    let threaded = this.threaded();
    let Some(thread) = threaded.current_thread() else {
        return;
    };
    threaded.stopping.set();
    this.wake_for_stop();
    if !thread.is_current() {
        thread.join(threaded.join_timeout);
        if thread.is_alive() {
            log::warn!(
                "Hotkey thread did not stop within {:.1} s",
                threaded.join_timeout.as_secs_f64()
            );
        }
    }
    if !thread.is_alive() {
        *threaded.thread.lock() = None;
    }
}

fn thread_main<T: ThreadedNative>(this: Arc<T>, exited: Arc<Event>) {
    let name = this.core().name;
    let threaded = this.threaded();
    let ok = catch_unwind(AssertUnwindSafe(|| this.thread_setup())).unwrap_or_else(|payload| {
        log::error!(
            "Hotkey backend {name} failed to initialise: {}",
            panic_message(payload.as_ref())
        );
        false
    });
    threaded.start_ok.store(ok, Ordering::SeqCst);
    threaded.ready.set();
    if ok && let Err(payload) = catch_unwind(AssertUnwindSafe(|| this.thread_loop())) {
        log::error!(
            "Hotkey thread ({name}) crashed: {}",
            panic_message(payload.as_ref())
        );
    }
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| this.thread_teardown())) {
        log::error!(
            "Hotkey backend {name} cleanup failed: {}",
            panic_message(payload.as_ref())
        );
    }
    threaded.cancel_pending_calls();
    exited.set();
}

/// Run `function` on the backend thread and return its result, or `default` if the thread is
/// not running, does not answer within the call timeout or the function panics.
fn call_on_thread<T, R>(
    this: &Arc<T>,
    function: impl FnOnce() -> R + Send + 'static,
    default: R,
) -> R
where
    T: ThreadedNative,
    R: Send + 'static,
{
    let threaded = this.threaded();
    #[cfg(test)]
    if threaded.inline.load(Ordering::SeqCst) {
        return function();
    }
    let Some(thread) = threaded.current_thread().filter(|t| t.is_alive()) else {
        return default;
    };
    if thread.is_current() {
        return function();
    }
    if threaded.is_stopping() {
        return default;
    }
    let slot: Arc<Mutex<Option<R>>> = Arc::new(Mutex::new(None));
    let result = slot.clone();
    let call = PendingCall::new(move || *result.lock() = Some(function()));
    threaded.calls.lock().push_back(call.clone());
    if !this.wake() {
        call.cancel();
        return default;
    }
    if !call.done.wait(threaded.call_timeout) && !call.cancel() {
        // Already running: give the native call a little longer to finish.
        call.done.wait(threaded.call_timeout);
    }
    if !call.finished() {
        log::warn!("Hotkey thread did not answer in time");
        return default;
    }
    if let Some(error) = call.error.lock().take() {
        log::error!("Hotkey operation failed on the hotkey thread: {error}");
        return default;
    }
    slot.lock().take().unwrap_or(default)
}

// ------------------------------------------------------------------------- Windows
// The layout helpers are pure and tested on every OS; only the manager needs Windows.

#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const MOD_ALT: u32 = 0x0001;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const MOD_CONTROL: u32 = 0x0002;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const MOD_SHIFT: u32 = 0x0004;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const MOD_WIN: u32 = 0x0008;
#[cfg(windows)]
const MOD_NOREPEAT: u32 = 0x4000;
#[cfg(windows)]
const WM_QUIT: u32 = 0x0012;
#[cfg(windows)]
const WM_HOTKEY: u32 = 0x0312;
#[cfg(windows)]
const WM_USER: u32 = 0x0400;
#[cfg(windows)]
const WM_APP_CALL: u32 = 0x8000 + 0x2E7; // WM_APP + private offset
#[cfg(windows)]
const ERROR_HOTKEY_ALREADY_REGISTERED: u32 = 1409;

#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
fn win_modifiers<'a>(modifiers: impl IntoIterator<Item = &'a str>) -> u32 {
    modifiers.into_iter().fold(0, |mask, m| {
        mask | match m {
            "ctrl" => MOD_CONTROL,
            "alt" => MOD_ALT,
            "shift" => MOD_SHIFT,
            _ => MOD_WIN,
        }
    })
}

/// The US virtual-key code of `key` (OEM keys at their US positions; other layouts are
/// resolved with `VkKeyScanW` first, see [`win_virtual_key`]).
fn win_vk(key: &str) -> Option<u8> {
    let mut chars = key.chars();
    if let (Some(ch), None) = (chars.next(), chars.next())
        && ch.is_ascii_alphanumeric()
    {
        return Some(ch.to_ascii_uppercase() as u8);
    }
    if let Some(n) = key.strip_prefix('f').and_then(|n| n.parse::<u8>().ok()) {
        return (1..=24).contains(&n).then_some(0x6F + n); // VK_F1 = 0x70 … VK_F24 = 0x87
    }
    Some(match key {
        "space" => 0x20,
        "enter" => 0x0D,
        "tab" => 0x09,
        "escape" => 0x1B,
        "backspace" => 0x08,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "home" => 0x24,
        "end" => 0x23,
        "pageup" => 0x21,
        "pagedown" => 0x22,
        "left" => 0x25,
        "up" => 0x26,
        "right" => 0x27,
        "down" => 0x28,
        "semicolon" => 0xBA,
        "equal" => 0xBB,
        "comma" => 0xBC,
        "minus" => 0xBD,
        "period" => 0xBE,
        "slash" => 0xBF,
        "grave" => 0xC0,
        "bracketleft" => 0xDB,
        "backslash" => 0xDC,
        "bracketright" => 0xDD,
        "quote" => 0xDE,
        _ => return None,
    })
}

/// `VkKeyScanW` of a character: the virtual key in the low byte, the shift state in the high
/// byte, -1 if no key of the layout types it.
type VkScan<'a> = &'a dyn Fn(char) -> Result<i32, String>;
/// `MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)`: the scan code of a virtual key, 0 if no key makes it.
type VkToScan<'a> = &'a dyn Fn(u8) -> Result<u32, String>;

/// Virtual-key code for `key`, or `None` if no key of the layout produces it.
///
/// Punctuation lives on different keys in different layouts (on a Turkish Q keyboard "." is
/// where "/" is on a US one), so the active layout is asked first with `vk_scan`
/// (`VkKeyScanW`). A character that the layout only types with Shift or AltGr (German '/',
/// Turkish '[') has no key of its own: the user asked for the unshifted key, and the US virtual
/// key would be a different key (Turkish 'ğ' for '[') or no key at all, so `None` is returned.
/// The US table is only a fallback for characters the layout lacks entirely (Cyrillic, Greek),
/// where the US position is the sensible guess, and even then only if a physical key produces
/// that virtual key (`vk_to_scan`, i.e. `MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)`, is non-zero).
fn win_virtual_key(key: &str, vk_scan: Option<VkScan>, vk_to_scan: Option<VkToScan>) -> Option<u8> {
    let Some(ch) = punctuation(key) else {
        return win_vk(key);
    };
    if let Some(vk_scan) = vk_scan {
        let result = vk_scan(ch).unwrap_or(-1);
        if result != -1 {
            if (result >> 8) & 0xFF != 0 {
                log::debug!("{ch:?} needs Shift or AltGr on the current keyboard layout");
                return None;
            }
            return Some((result & 0xFF) as u8);
        }
    }
    let vk = win_vk(key)?;
    if let Some(vk_to_scan) = vk_to_scan {
        match vk_to_scan(vk) {
            Ok(0) => return None,
            Ok(_) => {}
            Err(err) => log::debug!("MapVirtualKey({vk:#x}) failed: {err}"),
        }
    }
    Some(vk)
}

/// [`win_virtual_key`] on the active keyboard layout (user32 is only asked for punctuation).
fn win_layout_virtual_key(key: &str) -> Option<u8> {
    if punctuation(key).is_none() {
        return win_virtual_key(key, None, None); // layout independent
    }
    #[cfg(windows)]
    {
        win32::virtual_key_on_layout(key)
    }
    #[cfg(not(windows))]
    {
        win_virtual_key(key, None, None)
    }
}

#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const MAPVK_VK_TO_VSC: u32 = 0;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const TOUNICODE_KEEP_STATE: u32 = 0x4; // Windows 10 1607+: do not change the keyboard state
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const LOCALE_SENGLISHDISPLAYNAME: u32 = 0x72;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_SHIFT: usize = 0x10;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_CONTROL: usize = 0x11;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_MENU: usize = 0x12;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_LSHIFT: usize = 0xA0;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_LCONTROL: usize = 0xA2;
#[cfg_attr(not(windows), allow(dead_code))] // used by the Windows manager and the tests
const VK_RMENU: usize = 0xA5;

/// Read-only questions about the installed Windows keyboard layouts.
pub(crate) trait WinLayoutProbe: Send + Sync {
    /// Handles (HKLs) of every keyboard layout the user has installed.
    fn layouts(&self) -> Vec<isize>;

    /// What Ctrl+Alt(+Shift)+`vk` types on layout `hkl`: `(text, is_dead_key)`.
    fn altgr_text(&self, vk: u8, shift: bool, hkl: isize) -> Option<(String, bool)>;

    /// Display name of layout `hkl` such as `"Polish (Programmers)"`.
    fn layout_name(&self, hkl: isize) -> String;
}

/// Why a Ctrl+Alt hotkey would eat a character typed with AltGr, or `None`.
///
/// Windows reports AltGr as LCtrl+RAlt, and `RegisterHotKey` cannot tell left from right
/// modifiers, so a Ctrl+Alt(+Shift) hotkey fires on AltGr(+Shift) as well and the character
/// never reaches the application (Ctrl+Alt+T is AltGr+T, '₺', on Turkish Q). Hotkeys are matched
/// by virtual key whatever layout is active later, so every installed layout is checked. With
/// the Win key held AltGr cannot be meant: no Windows layout uses the Win key as a character
/// modifier.
fn win_altgr_conflict(hotkey: &Hotkey, vk: u8, probe: &dyn WinLayoutProbe) -> Option<String> {
    if !(hotkey.has("ctrl") && hotkey.has("alt")) || hotkey.has("meta") {
        return None;
    }
    let shift = hotkey.has("shift");
    for hkl in probe.layouts() {
        let Some((text, dead)) = probe.altgr_text(vk, shift, hkl) else {
            continue;
        };
        let what = if dead {
            format!("the dead key {}", describe_typed(&text))
        } else {
            describe_typed(&text)
        };
        let mut altgr = vec!["AltGr".to_string()];
        if shift {
            altgr.push("Shift".to_string());
        }
        altgr.push(key_label(hotkey.key, false));
        // "is AltGr+, and types" rather than "is AltGr+,, which types": a punctuation key
        // label must not run into the sentence's own punctuation.
        return Some(format!(
            "{} is {} and types {what} on the {} keyboard layout",
            format_hotkey(hotkey, Some(false)),
            altgr.join("+"),
            probe.layout_name(hkl)
        ));
    }
    None
}

/// `WindowsHotkeyManager::layout_conflict`: advisory only, so a failing layout query never
/// breaks a settings dialog.
#[cfg_attr(not(windows), allow(dead_code))]
fn win_layout_conflict(hotkey: &Hotkey, probe: &dyn WinLayoutProbe) -> Option<String> {
    let checked = catch_unwind(AssertUnwindSafe(|| {
        win_layout_virtual_key(hotkey.key).and_then(|vk| win_altgr_conflict(hotkey, vk, probe))
    }));
    checked.unwrap_or_else(|payload| {
        log::debug!(
            "Keyboard layout check for {hotkey} failed: {}",
            panic_message(payload.as_ref())
        );
        None
    })
}

/// The user32/kernel32 calls behind [`Win32LayoutProbe`] (a fake in tests).
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) trait Win32LayoutApi: Send + Sync {
    /// `GetKeyboardLayoutList`: the count with `None`, else fills `handles`.
    fn get_keyboard_layout_list(&self, handles: Option<&mut [isize]>) -> i32;

    fn map_virtual_key_ex(&self, code: u32, map_type: u32, hkl: isize) -> u32;

    fn to_unicode_ex(
        &self,
        vk: u32,
        scan: u32,
        state: &[u8; 256],
        buf: &mut [u16],
        flags: u32,
        hkl: isize,
    ) -> i32;

    fn get_locale_info(&self, lcid: u32, lctype: u32, buf: &mut [u16]) -> i32;

    /// The registry's "Layout Text" of keyboard layout `hkl`.
    fn layout_text(&self, hkl: isize) -> Option<String>;
}

/// [`WinLayoutProbe`] backed by user32 (safe to use from any thread). None of the queries loads
/// or activates a layout; `ToUnicodeEx` is always called with flag 0x4 so that it leaves the
/// keyboard state (including a pending dead key) untouched.
#[cfg_attr(not(windows), allow(dead_code))]
struct Win32LayoutProbe<A> {
    api: A,
}

impl<A: Win32LayoutApi> WinLayoutProbe for Win32LayoutProbe<A> {
    fn layouts(&self) -> Vec<isize> {
        let count = self.api.get_keyboard_layout_list(None);
        if count <= 0 {
            return Vec::new();
        }
        let mut handles = vec![0isize; count as usize];
        let count = self.api.get_keyboard_layout_list(Some(&mut handles));
        handles.truncate(count.max(0) as usize);
        handles
    }

    fn altgr_text(&self, vk: u8, shift: bool, hkl: isize) -> Option<(String, bool)> {
        let scan = self
            .api
            .map_virtual_key_ex(u32::from(vk), MAPVK_VK_TO_VSC, hkl);
        if scan == 0 {
            return None; // no key produces this virtual key on that layout
        }
        let mut state = [0u8; 256];
        for vk_mod in [VK_CONTROL, VK_LCONTROL, VK_MENU, VK_RMENU] {
            state[vk_mod] = 0x80;
        }
        if shift {
            state[VK_SHIFT] = 0x80;
            state[VK_LSHIFT] = 0x80;
        }
        let mut buf = [0u16; 8];
        let count = self.api.to_unicode_ex(
            u32::from(vk),
            scan,
            &state,
            &mut buf,
            TOUNICODE_KEEP_STATE,
            hkl,
        );
        if count == 0 {
            return None;
        }
        let dead = count < 0;
        let raw = if dead {
            &buf[..1]
        } else {
            &buf[..(count as usize).min(buf.len())]
        };
        // Control characters are not "typed text" (Ctrl+Alt+Enter may report "\r").
        let text: String = String::from_utf16_lossy(raw)
            .chars()
            .filter(|c| !c.is_control())
            .collect();
        (!text.is_empty()).then_some((text, dead))
    }

    fn layout_name(&self, hkl: isize) -> String {
        if let Some(name) = self.api.layout_text(hkl).filter(|n| !n.is_empty()) {
            return name;
        }
        let mut buf = [0u16; 128];
        if self
            .api
            .get_locale_info((hkl & 0xFFFF) as u32, LOCALE_SENGLISHDISPLAYNAME, &mut buf)
            != 0
        {
            let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
            return String::from_utf16_lossy(&buf[..end]);
        }
        format!("0x{:08X}", hkl & 0xFFFF_FFFF)
    }
}

#[cfg(windows)]
mod win32 {
    //! The Win32 calls of the Windows manager.

    use windows::Win32::Foundation::{ERROR_SUCCESS, LPARAM, WPARAM};
    use windows::Win32::Globalization::GetLocaleInfoW;
    use windows::Win32::System::Registry::{
        HKEY, HKEY_LOCAL_MACHINE, KEY_READ, RRF_RT_REG_SZ, RegCloseKey, RegEnumKeyExW,
        RegGetValueW, RegOpenKeyExW,
    };
    use windows::Win32::System::Threading::GetCurrentThreadId;
    use windows::Win32::UI::Input::KeyboardAndMouse::{
        GetKeyboardLayoutList, HKL, HOT_KEY_MODIFIERS, MAP_VIRTUAL_KEY_TYPE, MapVirtualKeyExW,
        MapVirtualKeyW, RegisterHotKey, ToUnicodeEx, UnregisterHotKey, VkKeyScanW,
    };
    use windows::Win32::UI::WindowsAndMessaging::{
        GetMessageW, MSG, PM_NOREMOVE, PeekMessageW, PostThreadMessageW,
    };
    use windows::core::{HSTRING, PCWSTR, PWSTR, w};

    use super::{MAPVK_VK_TO_VSC, Win32LayoutApi, win_virtual_key};

    const KEYBOARD_LAYOUTS_KEY: PCWSTR = w!(r"SYSTEM\CurrentControlSet\Control\Keyboard Layouts");

    pub(super) fn current_thread_id() -> u32 {
        // SAFETY: no arguments or preconditions.
        unsafe { GetCurrentThreadId() }
    }

    /// PostThreadMessageW fails until the thread owns a message queue; any Peek/GetMessage call
    /// creates it.
    pub(super) fn create_message_queue() {
        let mut msg = MSG::default();
        // SAFETY: `msg` is a valid, writable MSG for the call.
        let _ =
            unsafe { PeekMessageW(&mut msg, None, super::WM_USER, super::WM_USER, PM_NOREMOVE) };
    }

    /// `GetMessageW`: 0 on WM_QUIT, -1 on failure.
    pub(super) fn get_message(msg: &mut MSG) -> i32 {
        // SAFETY: `msg` is a valid, writable MSG; no window filter.
        unsafe { GetMessageW(msg, None, 0, 0) }.0
    }

    pub(super) fn last_error() -> u32 {
        windows::core::Error::from_thread().code().0 as u32 & 0xFFFF
    }

    pub(super) fn post_thread_message(thread_id: u32, message: u32, wparam: usize) -> bool {
        // SAFETY: plain value arguments; an invalid thread id only makes the call fail.
        unsafe { PostThreadMessageW(thread_id, message, WPARAM(wparam), LPARAM(0)) }.is_ok()
    }

    /// `RegisterHotKey` with a NULL window; the Win32 error code on failure.
    pub(super) fn register_hotkey(id: u32, modifiers: u32, vk: u8) -> Result<(), u32> {
        // SAFETY: NULL window (thread queue); plain value arguments.
        unsafe { RegisterHotKey(None, id as i32, HOT_KEY_MODIFIERS(modifiers), u32::from(vk)) }
            .map_err(|err| err.code().0 as u32 & 0xFFFF)
    }

    pub(super) fn unregister_hotkey(id: u32) {
        // SAFETY: plain value arguments; an unknown id only makes the call fail.
        let _ = unsafe { UnregisterHotKey(None, id as i32) };
    }

    /// [`win_virtual_key`] asking the active layout with `VkKeyScanW` and `MapVirtualKeyW`.
    pub(super) fn virtual_key_on_layout(key: &str) -> Option<u8> {
        let vk_scan = |ch: char| -> Result<i32, String> {
            let mut units = [0u16; 2];
            match ch.encode_utf16(&mut units) {
                // SAFETY: plain value argument.
                [unit] => Ok(i32::from(unsafe { VkKeyScanW(*unit) })),
                _ => Ok(-1),
            }
        };
        let vk_to_scan = |vk: u8| -> Result<u32, String> {
            // SAFETY: plain value arguments.
            Ok(unsafe { MapVirtualKeyW(u32::from(vk), MAP_VIRTUAL_KEY_TYPE(MAPVK_VK_TO_VSC)) })
        };
        win_virtual_key(key, Some(&vk_scan), Some(&vk_to_scan))
    }

    /// user32 and kernel32 as [`Win32LayoutApi`].
    pub(super) struct User32;

    fn hkl(handle: isize) -> Option<HKL> {
        Some(HKL(handle as *mut core::ffi::c_void))
    }

    impl Win32LayoutApi for User32 {
        fn get_keyboard_layout_list(&self, handles: Option<&mut [isize]>) -> i32 {
            match handles {
                // SAFETY: no buffer; the call only returns the number of layouts.
                None => unsafe { GetKeyboardLayoutList(None) },
                Some(handles) => {
                    let mut raw = vec![HKL::default(); handles.len()];
                    // SAFETY: `raw` is a writable HKL buffer; its length is passed with it.
                    let count = unsafe { GetKeyboardLayoutList(Some(&mut raw)) };
                    for (out, h) in handles.iter_mut().zip(&raw) {
                        *out = h.0 as isize;
                    }
                    count
                }
            }
        }

        fn map_virtual_key_ex(&self, code: u32, map_type: u32, handle: isize) -> u32 {
            // SAFETY: plain value arguments; a stale layout handle only fails the call.
            unsafe { MapVirtualKeyExW(code, MAP_VIRTUAL_KEY_TYPE(map_type), hkl(handle)) }
        }

        fn to_unicode_ex(
            &self,
            vk: u32,
            scan: u32,
            state: &[u8; 256],
            buf: &mut [u16],
            flags: u32,
            handle: isize,
        ) -> i32 {
            // SAFETY: `state` is the 256 bytes of key state and `buf` a writable buffer whose
            // length is passed with it; a stale layout handle only fails the call.
            unsafe { ToUnicodeEx(vk, scan, state, buf, flags, hkl(handle)) }
        }

        fn get_locale_info(&self, lcid: u32, lctype: u32, buf: &mut [u16]) -> i32 {
            // SAFETY: `buf` is a writable buffer whose length is passed with it.
            unsafe { GetLocaleInfoW(lcid, lctype, Some(buf)) }
        }

        fn layout_text(&self, handle: isize) -> Option<String> {
            layout_text(handle)
        }
    }

    /// A string value of `subkey` under `key`, or `None`.
    fn reg_string(key: HKEY, subkey: &str, value: PCWSTR) -> Option<String> {
        let subkey = HSTRING::from(subkey);
        let mut size = 0u32;
        // SAFETY: `subkey` (an HSTRING) and `value` (a `w!` literal) are NUL-terminated and
        // outlive the call; only the size is requested, so there is no data buffer.
        let status = unsafe {
            RegGetValueW(
                key,
                &subkey,
                value,
                RRF_RT_REG_SZ,
                None,
                None,
                Some(&mut size),
            )
        };
        if status != ERROR_SUCCESS || size == 0 {
            return None;
        }
        let mut buf = vec![0u16; (size as usize).div_ceil(2)];
        // SAFETY: `buf` holds at least `size` bytes (rounded up to whole u16 units), the size
        // the first call reported; the strings are as above.
        let status = unsafe {
            RegGetValueW(
                key,
                &subkey,
                value,
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr().cast()),
                Some(&mut size),
            )
        };
        if status != ERROR_SUCCESS {
            return None;
        }
        let end = buf.iter().position(|c| *c == 0).unwrap_or(buf.len());
        Some(String::from_utf16_lossy(&buf[..end]))
    }

    /// The registry's "Layout Text" of keyboard layout `hkl` (e.g. `"Turkish Q"`).
    ///
    /// The high word of an HKL is either the low word of the layout id (KLID `0000xxxx`), an IME
    /// KLID (`Exxxxxxx`) or `Fnnn` where `nnn` is the layout's "Layout Id" value (variants such
    /// as Polish (214) or US-Dvorak). Only reads the registry.
    pub(super) fn layout_text(hkl: isize) -> Option<String> {
        let hkl = (hkl as u64 & 0xFFFF_FFFF) as u32;
        let device = hkl >> 16;
        let mut layouts = HKEY::default();
        // SAFETY: `KEYBOARD_LAYOUTS_KEY` is a NUL-terminated literal; `layouts` receives the
        // opened key, which is closed below.
        let status = unsafe {
            RegOpenKeyExW(
                HKEY_LOCAL_MACHINE,
                KEYBOARD_LAYOUTS_KEY,
                None,
                KEY_READ,
                &mut layouts,
            )
        };
        if status != ERROR_SUCCESS {
            return None;
        }
        let text = if device & 0xF000 == 0xF000 {
            let wanted = device & 0x0FFF;
            let mut found = None;
            for index in 0.. {
                let mut name = [0u16; 256];
                let mut len = name.len() as u32;
                // SAFETY: `layouts` is the open key; `name` is a writable buffer of `len` units
                // and `len` is updated to the name's length; the optional outputs are omitted.
                let status = unsafe {
                    RegEnumKeyExW(
                        layouts,
                        index,
                        Some(PWSTR(name.as_mut_ptr())),
                        &mut len,
                        None,
                        None,
                        None,
                        None,
                    )
                };
                if status != ERROR_SUCCESS {
                    break;
                }
                let klid = String::from_utf16_lossy(&name[..len as usize]);
                let id = reg_string(layouts, &klid, w!("Layout Id"))
                    .and_then(|id| u32::from_str_radix(id.trim(), 16).ok());
                if id == Some(wanted) {
                    found = reg_string(layouts, &klid, w!("Layout Text"));
                    break;
                }
            }
            found
        } else {
            let klid = if device & 0xF000 == 0xE000 {
                format!("{hkl:08X}")
            } else {
                format!("{device:08X}")
            };
            reg_string(layouts, &klid, w!("Layout Text")).filter(|t| !t.is_empty())
        };
        // SAFETY: `layouts` was opened above and is not used afterwards.
        let _ = unsafe { RegCloseKey(layouts) };
        text
    }
}

#[cfg(windows)]
/// The Windows manager's state (see [`WindowsHotkeyManager`]).
struct WindowsBackend {
    core: Core,
    threaded: Threaded,
    thread_id: std::sync::atomic::AtomicU32,
    os_ids: Mutex<HashSet<u32>>, // touched only on the hotkey thread
    layout_probe: Mutex<Option<Arc<dyn WinLayoutProbe>>>,
}

#[cfg(windows)]
native_manager!(
    /// `RegisterHotKey` with `MOD_NOREPEAT` on a dedicated message-loop thread.
    ///
    /// Hotkeys registered with a `NULL` window belong to the registering thread, so
    /// registrations are posted to that thread (`PostThreadMessageW` with a private `WM_APP`
    /// message) and executed there. `GetMessageW` blocks, so the thread costs nothing while idle.
    ///
    /// Ctrl+Alt(+Shift) combinations that type a character with AltGr on any installed keyboard
    /// layout are refused (see `win_altgr_conflict`); a layout probe replaces the
    /// user32-backed layout queries in tests.
    WindowsHotkeyManager(WindowsBackend)
);

#[cfg(windows)]
impl WindowsHotkeyManager {
    pub fn new() -> WindowsHotkeyManager {
        WindowsHotkeyManager::with_layout_probe(None)
    }

    pub(crate) fn with_layout_probe(
        layout_probe: Option<Arc<dyn WinLayoutProbe>>,
    ) -> WindowsHotkeyManager {
        WindowsHotkeyManager(Arc::new(WindowsBackend {
            core: Core::new("win32", None),
            threaded: Threaded::new(),
            thread_id: std::sync::atomic::AtomicU32::new(0),
            os_ids: Mutex::new(HashSet::new()),
            layout_probe: Mutex::new(layout_probe),
        }))
    }
}

#[cfg(windows)]
impl Default for WindowsHotkeyManager {
    fn default() -> Self {
        WindowsHotkeyManager::new()
    }
}

#[cfg(windows)]
impl WindowsBackend {
    fn probe(&self) -> Arc<dyn WinLayoutProbe> {
        self.layout_probe
            .lock()
            .get_or_insert_with(|| Arc::new(Win32LayoutProbe { api: win32::User32 }))
            .clone()
    }

    fn post(&self, message: u32) -> bool {
        let thread_id = self.thread_id.load(Ordering::SeqCst);
        thread_id != 0 && win32::post_thread_message(thread_id, message, 0)
    }

    fn register_on_thread(&self, binding: &Binding) -> bool {
        let hotkey = &binding.hotkey;
        let label = format_hotkey(hotkey, Some(false));
        let Some(vk) = win_layout_virtual_key(hotkey.key) else {
            let typed = punctuation(hotkey.key).map_or(hotkey.key.to_string(), String::from);
            return self.core.reject(
                binding,
                format!(
                    "{label}: no key types {} without Shift or AltGr on the current keyboard \
                     layout",
                    describe_typed(&typed)
                ),
            );
        };
        let probe = self.probe();
        let conflict = catch_unwind(AssertUnwindSafe(|| win_altgr_conflict(hotkey, vk, &*probe)))
            .unwrap_or_else(|payload| {
                // The check must never make hotkeys unusable.
                log::warn!(
                    "Could not check {label} against the keyboard layouts: {}",
                    panic_message(payload.as_ref())
                );
                None
            });
        if let Some(conflict) = conflict {
            return self.core.reject(binding, conflict);
        }
        let mods = win_modifiers(hotkey.modifiers().iter().copied()) | MOD_NOREPEAT;
        if let Err(error) = win32::register_hotkey(binding.id, mods, vk) {
            if error == ERROR_HOTKEY_ALREADY_REGISTERED {
                return self.core.reject(
                    binding,
                    format!("{label} is already in use by another application"),
                );
            }
            return self.core.reject(
                binding,
                format!("RegisterHotKey({label}) failed (error {error})"),
            );
        }
        self.os_ids.lock().insert(binding.id);
        true
    }

    fn unregister_on_thread(&self, hotkey_id: u32) {
        if self.os_ids.lock().remove(&hotkey_id) {
            win32::unregister_hotkey(hotkey_id);
        }
    }
}

#[cfg(windows)]
impl Native for WindowsBackend {
    fn core(&self) -> &Core {
        &self.core
    }

    fn backend_start(self: &Arc<Self>) -> bool {
        threaded_backend_start(self)
    }

    fn backend_stop(self: &Arc<Self>) {
        threaded_backend_stop(self)
    }

    fn native_register(self: &Arc<Self>, binding: &Arc<Binding>) -> bool {
        let (this, binding) = (self.clone(), binding.clone());
        call_on_thread(self, move || this.register_on_thread(&binding), false)
    }

    fn native_unregister(self: &Arc<Self>, binding: &Arc<Binding>) {
        let (this, id) = (self.clone(), binding.id);
        call_on_thread(self, move || this.unregister_on_thread(id), ())
    }

    fn native_layout_conflict(self: &Arc<Self>, hotkey: &Hotkey) -> Option<String> {
        win_layout_conflict(hotkey, &*self.probe())
    }
}

#[cfg(windows)]
impl ThreadedNative for WindowsBackend {
    fn threaded(&self) -> &Threaded {
        &self.threaded
    }

    fn thread_setup(self: &Arc<Self>) -> bool {
        self.thread_id
            .store(win32::current_thread_id(), Ordering::SeqCst);
        // PostThreadMessageW fails until the thread owns a message queue; create it before
        // announcing readiness.
        win32::create_message_queue();
        true
    }

    fn thread_loop(self: &Arc<Self>) {
        let mut msg = windows::Win32::UI::WindowsAndMessaging::MSG::default();
        while !self.threaded.is_stopping() {
            match win32::get_message(&mut msg) {
                0 => break, // WM_QUIT
                -1 => {
                    log::error!("GetMessageW failed (error {})", win32::last_error());
                    break;
                }
                _ => {}
            }
            if msg.message == WM_HOTKEY {
                self.core.dispatch(msg.wParam.0 as u32);
            } else if msg.message == WM_APP_CALL {
                self.threaded.run_pending_calls();
            }
        }
    }

    fn thread_teardown(self: &Arc<Self>) {
        for hotkey_id in self.os_ids.lock().drain() {
            win32::unregister_hotkey(hotkey_id);
        }
        // Thread ids are recycled; never post to this one again.
        self.thread_id.store(0, Ordering::SeqCst);
    }

    fn wake(&self) -> bool {
        self.post(WM_APP_CALL)
    }

    fn wake_for_stop(&self) {
        self.post(WM_QUIT);
    }
}

// --------------------------------------------------------------------------- macOS
// The manager is plain logic over `CarbonApi`, so it runs against a fake on every OS; only the
// real Carbon binding needs macOS.

const MAC_CMD_KEY: u32 = 0x0100;
const MAC_SHIFT_KEY: u32 = 0x0200;
const MAC_OPTION_KEY: u32 = 0x0800;
const MAC_CONTROL_KEY: u32 = 0x1000;
const NO_ERR: i32 = 0;
const EVENT_NOT_HANDLED_ERR: i32 = -9874;
const EVENT_HOTKEY_EXISTS_ERR: i32 = -9878;
#[cfg(target_os = "macos")]
const EVENT_HOTKEY_PRESSED: u32 = 5;
/// kEventHotKeyExclusive: without it Carbon lets several applications register the same
/// combination and one press fires all of them, so a combination another app owns could never
/// be reported. With it, registering fails with eventHotKeyExistsErr while any other
/// registration of the combination exists (reported as "already in use by another
/// application"), and while we hold it, registrations that other apps make later stay silent
/// until we release it. The defaults (⌃⌥⌘T/P/C) are therefore chosen to be nobody else's default.
#[cfg(target_os = "macos")]
const EVENT_HOTKEY_EXCLUSIVE: u32 = 1 << 0;
#[cfg(target_os = "macos")]
const UC_KEY_ACTION_DOWN: u16 = 0;
#[cfg(target_os = "macos")]
const UC_KEY_TRANSLATE_NO_DEAD_KEYS: u32 = 1 << 0; // kUCKeyTranslateNoDeadKeysMask
#[cfg(target_os = "macos")]
const CF_NOTIFICATION_DELIVER_IMMEDIATELY: isize = 4;

/// Key codes of the character keys whose meaning follows the keyboard layout: the main block
/// (0x00-0x32, which includes the ISO section key 0x0A) plus the JIS Yen and underscore keys.
/// Keypad keys are left out on purpose: they are separate keys.
#[cfg(target_os = "macos")]
fn mac_character_keycodes() -> impl Iterator<Item = u16> {
    (0x00..0x33).chain([0x5D, 0x5E])
}

/// Four-character Carbon `OSType` as an integer (`b"keyb"` → 0x6B657962).
const fn fourcc(code: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*code)
}

#[cfg(target_os = "macos")]
const EVENT_CLASS_KEYBOARD: u32 = fourcc(b"keyb");
#[cfg(target_os = "macos")]
const EVENT_PARAM_DIRECT_OBJECT: u32 = fourcc(b"----");
#[cfg(target_os = "macos")]
const TYPE_EVENT_HOTKEY_ID: u32 = fourcc(b"hkid");
const HOTKEY_SIGNATURE: u32 = fourcc(b"EyTk");

/// kVK_* virtual key codes from HIToolbox/Events.h. They identify physical key positions on an
/// ANSI (US) keyboard; macOS has no F21–F24.
#[rustfmt::skip]
const MAC_KEYCODES: [(&str, u32); 82] = [
    ("a", 0x00), ("s", 0x01), ("d", 0x02), ("f", 0x03), ("h", 0x04), ("g", 0x05), ("z", 0x06),
    ("x", 0x07), ("c", 0x08), ("v", 0x09), ("b", 0x0B), ("q", 0x0C), ("w", 0x0D), ("e", 0x0E),
    ("r", 0x0F), ("y", 0x10), ("t", 0x11), ("1", 0x12), ("2", 0x13), ("3", 0x14), ("4", 0x15),
    ("6", 0x16), ("5", 0x17), ("equal", 0x18), ("9", 0x19), ("7", 0x1A), ("minus", 0x1B),
    ("8", 0x1C), ("0", 0x1D), ("bracketright", 0x1E), ("o", 0x1F), ("u", 0x20),
    ("bracketleft", 0x21), ("i", 0x22), ("p", 0x23), ("enter", 0x24), ("l", 0x25), ("j", 0x26),
    ("quote", 0x27), ("k", 0x28), ("semicolon", 0x29), ("backslash", 0x2A), ("comma", 0x2B),
    ("slash", 0x2C), ("n", 0x2D), ("m", 0x2E), ("period", 0x2F), ("tab", 0x30), ("space", 0x31),
    ("grave", 0x32),
    ("backspace", 0x33), // kVK_Delete
    ("escape", 0x35), ("f17", 0x40), ("f18", 0x4F), ("f19", 0x50), ("f20", 0x5A), ("f5", 0x60),
    ("f6", 0x61), ("f7", 0x62), ("f3", 0x63), ("f8", 0x64), ("f9", 0x65), ("f11", 0x67),
    ("f13", 0x69), ("f16", 0x6A), ("f14", 0x6B), ("f10", 0x6D), ("f12", 0x6F), ("f15", 0x71),
    ("insert", 0x72), // kVK_Help
    ("home", 0x73), ("pageup", 0x74),
    ("delete", 0x75), // kVK_ForwardDelete
    ("f4", 0x76), ("end", 0x77), ("f2", 0x78), ("pagedown", 0x79), ("f1", 0x7A), ("left", 0x7B),
    ("right", 0x7C), ("down", 0x7D), ("up", 0x7E),
];

fn mac_us_keycode(key: &str) -> Option<u32> {
    MAC_KEYCODES
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, c)| *c)
}

fn mac_modifiers<'a>(modifiers: impl IntoIterator<Item = &'a str>) -> u32 {
    modifiers.into_iter().fold(0, |mask, m| {
        mask | match m {
            "ctrl" => MAC_CONTROL_KEY,
            "alt" => MAC_OPTION_KEY,
            "shift" => MAC_SHIFT_KEY,
            _ => MAC_CMD_KEY,
        }
    })
}

/// The character a canonical key stands for: punctuation, letters and digits.
fn key_char(key: &str) -> Option<char> {
    if let Some(ch) = punctuation(key) {
        return Some(ch);
    }
    let mut chars = key.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => Some(ch),
        _ => None,
    }
}

/// Character → key code of letters, digits and punctuation at their US (ANSI) position.
fn mac_ansi_chars() -> impl Iterator<Item = (char, u32)> {
    MAC_KEYCODES
        .iter()
        .filter_map(|(key, code)| key_char(key).map(|ch| (ch, *code)))
}

/// Invert `{keycode: typed text}` into `{character: keycode}`.
///
/// Only single, non-blank characters are kept (lower-cased). If several keys type the same
/// character, the key at that character's US position wins, otherwise the lowest key code.
fn mac_char_keymap(chars: &BTreeMap<u32, String>) -> HashMap<char, u32> {
    let mut keymap = HashMap::new();
    for (code, text) in chars {
        let text = text.to_lowercase();
        let mut it = text.chars();
        let (Some(ch), None) = (it.next(), it.next()) else {
            continue;
        };
        if ch.is_whitespace() {
            continue;
        }
        let us = mac_ansi_chars().find(|(c, _)| *c == ch).map(|(_, k)| k);
        if !keymap.contains_key(&ch) || us == Some(*code) {
            keymap.insert(ch, *code);
        }
    }
    keymap
}

/// Key code for canonical `key` on the layout described by `keymap`.
///
/// `keymap` (from [`mac_char_keymap`]) makes letters, digits and punctuation follow the layout,
/// so ⌃⌥A is the key labelled A on AZERTY and Dvorak too. Letters and digits that the layout
/// does not type without modifiers (Cyrillic letters, the AZERTY number row) keep their US
/// position, which is where their label is; punctuation that needs Shift/Option on this layout
/// has no key of its own and yields `None`. Without a `keymap` the US positions are used.
fn mac_keycode(key: &str, keymap: Option<&HashMap<char, u32>>) -> Option<u32> {
    if let (Some(ch), Some(keymap)) = (key_char(key), keymap) {
        if let Some(code) = keymap.get(&ch) {
            return Some(*code);
        }
        if punctuation(key).is_some() {
            return None;
        }
    }
    mac_us_keycode(key)
}

/// Whether `hotkey` uses Option without Control/Command: the macOS typing layer.
fn mac_option_types(hotkey: &Hotkey) -> bool {
    hotkey.has("alt") && !hotkey.has("ctrl") && !hotkey.has("meta")
}

/// Handles a Carbon hot-key event (its `EventRef`); returns the OSStatus for Carbon.
type CarbonHandler = Box<dyn Fn(usize) -> i32 + Send + Sync>;
/// Called when the user selects another keyboard layout.
type LayoutCallback = Box<dyn Fn() + Send + Sync>;

/// The handful of Carbon (HIToolbox) hot-key functions, plus the Text Input Sources /
/// `UCKeyTranslate` calls that read the keyboard layout.
///
/// Every method takes and returns plain values (references are opaque `usize` handles) so
/// [`MacHotkeyManager`] can be exercised with a fake on any OS.
pub(crate) trait CarbonApi: Send + Sync {
    /// `InstallEventHandler` for hot-key presses on the application target: `(status, ref)`.
    fn install_handler(&self, handler: CarbonHandler) -> (i32, usize);

    fn remove_handler(&self, handler_ref: usize) -> i32;

    /// `RegisterEventHotKey` (exclusive): `(status, ref)`.
    fn register_hotkey(
        &self,
        keycode: u32,
        modifiers: u32,
        signature: u32,
        hotkey_id: u32,
    ) -> (i32, usize);

    fn unregister_hotkey(&self, hotkey_ref: usize) -> i32;

    /// `(status, signature, id)` of the hot key that produced `event`.
    fn hotkey_id(&self, event: usize) -> (i32, u32, u32);

    /// What each character key types on the current keyboard layout.
    ///
    /// `modifiers` are Carbon modifier masks (`shiftKey`, `optionKey`). `ascii_capable` reads
    /// the ASCII-capable layout that macOS uses for shortcuts while a Cyrillic layout or an
    /// input method is selected. Returns `{keycode: text}` (dead keys give their spacing
    /// character), or `None` if the layout cannot be read. Main thread only, like all TIS calls.
    fn layout_characters(
        &self,
        modifiers: u32,
        ascii_capable: bool,
    ) -> Option<BTreeMap<u32, String>>;

    /// Call `callback` whenever the user selects another keyboard layout (on the main thread).
    /// Returns success.
    fn observe_layout_changes(&self, callback: LayoutCallback) -> bool;

    fn stop_observing_layout_changes(&self);
}

/// The real Carbon binding, or why it cannot be loaded.
fn load_carbon_api() -> Result<Arc<dyn CarbonApi>, String> {
    #[cfg(target_os = "macos")]
    {
        Ok(Arc::new(carbon::Carbon::default()))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err("Carbon exists on macOS only".to_string())
    }
}

/// Whether the calling thread is the process's main thread.
fn on_process_main_thread() -> bool {
    #[cfg(target_os = "macos")]
    {
        // SAFETY: pthread_main_np has no preconditions.
        unsafe { libc::pthread_main_np() != 0 }
    }
    #[cfg(not(target_os = "macos"))]
    {
        thread::current().name() == Some("main")
    }
}

#[derive(Default)]
struct MacState {
    handler_ref: Option<usize>,
    refs: HashMap<u32, usize>,
    /// binding id → key code it is registered at
    keycodes: HashMap<u32, u32>,
    /// character → key code (current layout)
    keymap: Option<HashMap<char, u32>>,
    observing: bool,
}

/// The macOS manager's state (see [`MacHotkeyManager`]).
struct MacBackend {
    core: Core,
    carbon: Mutex<Option<Arc<dyn CarbonApi>>>, // injectable for tests; loaded lazily otherwise
    state: Mutex<MacState>,
    /// Tests: the thread that counts as the main thread (`None`: the process's main thread).
    main_thread: Option<ThreadId>,
}

native_manager!(
    /// Carbon `RegisterEventHotKey` hotkeys.
    ///
    /// Unlike an event tap this needs neither Accessibility nor Input Monitoring permission.
    /// Carbon is not thread-safe, so every call must come from the main thread (calls from other
    /// threads are refused with a warning); callbacks run on the main thread, dispatched by the
    /// Cocoa run loop the UI is already running.
    ///
    /// Carbon matches virtual key codes, which are physical key positions. Letters, digits and
    /// punctuation are therefore looked up in the current keyboard layout (see `mac_keycode`),
    /// so ⌃⌥A is the key labelled A on AZERTY and Dvorak as it is on Windows and X11, and the
    /// registrations move when the user switches layouts. Hotkeys are registered exclusively,
    /// so a combination that another application already owns is reported as unavailable, and
    /// Option(+Shift) combinations that type a character are refused.
    MacHotkeyManager(MacBackend)
);

impl MacHotkeyManager {
    pub fn new() -> MacHotkeyManager {
        MacHotkeyManager::build(None, None)
    }

    /// A manager on `carbon` whose main thread is the calling thread (tests).
    #[cfg(test)]
    pub(crate) fn with_carbon(carbon: Option<Arc<dyn CarbonApi>>) -> MacHotkeyManager {
        MacHotkeyManager::build(carbon, Some(thread::current().id()))
    }

    fn build(
        carbon: Option<Arc<dyn CarbonApi>>,
        main_thread: Option<ThreadId>,
    ) -> MacHotkeyManager {
        MacHotkeyManager(Arc::new(MacBackend {
            core: Core::new("carbon", None),
            carbon: Mutex::new(carbon),
            state: Mutex::new(MacState::default()),
            main_thread,
        }))
    }
}

impl Default for MacHotkeyManager {
    fn default() -> Self {
        MacHotkeyManager::new()
    }
}

impl MacBackend {
    fn is_main_thread(&self) -> bool {
        match self.main_thread {
            Some(id) => thread::current().id() == id,
            None => on_process_main_thread(),
        }
    }

    fn on_main_thread(&self, what: &str) -> bool {
        if self.is_main_thread() {
            return true;
        }
        log::warn!("macOS hotkeys must be {what} from the main thread");
        false
    }

    /// The Carbon binding, loaded on first use (`None` if it cannot be).
    fn load_carbon(&self) -> Option<Arc<dyn CarbonApi>> {
        let mut carbon = self.carbon.lock();
        if carbon.is_none() {
            match load_carbon_api() {
                Ok(api) => *carbon = Some(api),
                Err(err) => {
                    log::warn!("Carbon hot-key API unavailable: {err}");
                    self.core.set_note(Some(
                        "The Carbon hot-key API could not be loaded.".to_string(),
                    ));
                }
            }
        }
        carbon.clone()
    }

    fn carbon(&self) -> Option<Arc<dyn CarbonApi>> {
        self.carbon.lock().clone()
    }

    /// Character → key code of the current layout, completed by the ASCII-capable one.
    fn read_keymap(&self, carbon: &dyn CarbonApi) -> Option<HashMap<char, u32>> {
        let current = carbon.layout_characters(0, false);
        let ascii_capable = carbon.layout_characters(0, true);
        if current.is_none() && ascii_capable.is_none() {
            return None;
        }
        let mut keymap = mac_char_keymap(&ascii_capable.unwrap_or_default());
        keymap.extend(mac_char_keymap(&current.unwrap_or_default()));
        Some(keymap)
    }

    /// Why an Option(+Shift) hotkey would eat a character typed with Option, or `None`.
    fn option_conflict(
        &self,
        carbon: &dyn CarbonApi,
        hotkey: &Hotkey,
        keycode: u32,
    ) -> Option<String> {
        if !mac_option_types(hotkey) {
            return None;
        }
        let chars =
            carbon.layout_characters(mac_modifiers(hotkey.modifiers().iter().copied()), false);
        let text = chars.unwrap_or_default().remove(&keycode)?;
        if text.is_empty() {
            return None;
        }
        Some(format!(
            "{} types {} on the current keyboard layout",
            format_hotkey(hotkey, Some(true)),
            describe_typed(&text)
        ))
    }

    fn register_carbon(&self, binding: &Binding) -> bool {
        let Some(carbon) = self.carbon() else {
            return false;
        };
        let hotkey = &binding.hotkey;
        let label = format_hotkey(hotkey, Some(true));
        let keymap = self.state.lock().keymap.clone();
        let Some(keycode) = mac_keycode(hotkey.key, keymap.as_ref()) else {
            if let Some(ch) = punctuation(hotkey.key) {
                return self.core.reject(
                    binding,
                    format!(
                        "{label}: no key types {} without modifiers on the current keyboard \
                         layout",
                        describe_typed(&ch.to_string())
                    ),
                );
            }
            return self.core.reject(
                binding,
                format!(
                    "{label}: Mac keyboards have no {}",
                    key_label(hotkey.key, false)
                ),
            );
        };
        let conflict = catch_unwind(AssertUnwindSafe(|| {
            self.option_conflict(&*carbon, hotkey, keycode)
        }))
        .unwrap_or_else(|payload| {
            // The check must never make hotkeys unusable.
            log::warn!(
                "Could not check {label} against the keyboard layout: {}",
                panic_message(payload.as_ref())
            );
            None
        });
        if let Some(conflict) = conflict {
            return self.core.reject(binding, conflict);
        }
        let (status, hotkey_ref) = carbon.register_hotkey(
            keycode,
            mac_modifiers(hotkey.modifiers().iter().copied()),
            HOTKEY_SIGNATURE,
            binding.id,
        );
        if status != NO_ERR {
            if status == EVENT_HOTKEY_EXISTS_ERR {
                return self.core.reject(
                    binding,
                    format!("{label} is already in use by another application"),
                );
            }
            return self.core.reject(
                binding,
                format!("RegisterEventHotKey({label}) failed (OSStatus {status})"),
            );
        }
        let mut state = self.state.lock();
        state.refs.insert(binding.id, hotkey_ref);
        state.keycodes.insert(binding.id, keycode);
        true
    }

    /// Move every registration to the key that types its character on the new layout.
    ///
    /// Bindings that cannot be registered on the new layout stay known (inactive) and are
    /// retried on the next layout change.
    fn on_layout_changed(&self) {
        if self.state.lock().handler_ref.is_none() || !self.on_main_thread("updated") {
            return;
        }
        let Some(carbon) = self.carbon() else {
            return;
        };
        let _api = self.core.api_lock.lock();
        let keymap = self.read_keymap(&*carbon);
        self.state.lock().keymap = keymap.clone();
        for binding in self.core.bindings() {
            let keycode = mac_keycode(binding.hotkey.key, keymap.as_ref());
            let old_ref = {
                let mut state = self.state.lock();
                if state.refs.contains_key(&binding.id)
                    && keycode == state.keycodes.get(&binding.id).copied()
                {
                    continue;
                }
                state.keycodes.remove(&binding.id);
                state.refs.remove(&binding.id)
            };
            if let Some(old_ref) = old_ref {
                carbon.unregister_hotkey(old_ref);
            }
            let ok = self.register_carbon(&binding);
            self.core.set_active(&binding, ok);
        }
    }

    fn on_event(&self, event: usize) -> i32 {
        let Some(carbon) = self.carbon() else {
            return EVENT_NOT_HANDLED_ERR;
        };
        let (status, signature, hotkey_id) = carbon.hotkey_id(event);
        if status != NO_ERR || signature != HOTKEY_SIGNATURE {
            return EVENT_NOT_HANDLED_ERR; // someone else's hot key: let it propagate
        }
        self.core.dispatch(hotkey_id);
        NO_ERR
    }
}

impl Native for MacBackend {
    fn core(&self) -> &Core {
        &self.core
    }

    fn backend_start(self: &Arc<Self>) -> bool {
        if self.state.lock().handler_ref.is_some() {
            return true;
        }
        if !self.on_main_thread("registered") {
            return false;
        }
        let Some(carbon) = self.load_carbon() else {
            return false;
        };
        let weak = Arc::downgrade(self);
        let (status, handler_ref) = carbon.install_handler(Box::new(move |event| {
            weak.upgrade()
                .map_or(EVENT_NOT_HANDLED_ERR, |this| this.on_event(event))
        }));
        if status != NO_ERR {
            log::warn!("InstallEventHandler failed (OSStatus {status})");
            return false;
        }
        let keymap = self.read_keymap(&*carbon);
        {
            let mut state = self.state.lock();
            state.handler_ref = Some(handler_ref);
            state.keymap = keymap;
        }
        let weak = Arc::downgrade(self);
        let observing = carbon.observe_layout_changes(Box::new(move || {
            if let Some(this) = weak.upgrade() {
                this.on_layout_changed();
            }
        }));
        self.state.lock().observing = observing;
        true
    }

    fn backend_stop(self: &Arc<Self>) {
        if self.state.lock().handler_ref.is_none() || !self.on_main_thread("stopped") {
            return;
        }
        let Some(carbon) = self.carbon() else {
            return;
        };
        let (observing, refs, handler_ref) = {
            let mut state = self.state.lock();
            let refs: Vec<usize> = state.refs.drain().map(|(_, r)| r).collect();
            state.keycodes.clear();
            state.keymap = None;
            (
                std::mem::take(&mut state.observing),
                refs,
                state.handler_ref.take(),
            )
        };
        if observing {
            carbon.stop_observing_layout_changes();
        }
        for hotkey_ref in refs {
            // normally none: stop() unregisters first
            carbon.unregister_hotkey(hotkey_ref);
        }
        if let Some(handler_ref) = handler_ref {
            carbon.remove_handler(handler_ref);
        }
    }

    fn native_register(self: &Arc<Self>, binding: &Arc<Binding>) -> bool {
        if !self.on_main_thread("registered") {
            return self.core.reject(
                binding,
                "macOS hotkeys can only be registered on the main thread",
            );
        }
        self.register_carbon(binding)
    }

    fn native_unregister(self: &Arc<Self>, binding: &Arc<Binding>) {
        let hotkey_ref = {
            let mut state = self.state.lock();
            state.keycodes.remove(&binding.id);
            state.refs.remove(&binding.id)
        };
        if let Some(hotkey_ref) = hotkey_ref
            && self.on_main_thread("unregistered")
            && let Some(carbon) = self.carbon()
        {
            let status = carbon.unregister_hotkey(hotkey_ref);
            if status != NO_ERR {
                log::debug!("UnregisterEventHotKey failed (OSStatus {status})");
            }
        }
    }

    fn native_layout_conflict(self: &Arc<Self>, hotkey: &Hotkey) -> Option<String> {
        if !mac_option_types(hotkey) || !self.is_main_thread() {
            return None;
        }
        let checked = catch_unwind(AssertUnwindSafe(|| {
            let carbon = self.load_carbon()?;
            let started = self.state.lock().handler_ref.is_some();
            let keymap = if started {
                self.state.lock().keymap.clone()
            } else {
                self.read_keymap(&*carbon)
            };
            let keycode = mac_keycode(hotkey.key, keymap.as_ref())?;
            self.option_conflict(&*carbon, hotkey, keycode)
        }));
        // Advisory only: never break a settings dialog.
        checked.unwrap_or_else(|payload| {
            log::debug!(
                "Keyboard layout check for {hotkey} failed: {}",
                panic_message(payload.as_ref())
            );
            None
        })
    }
}

#[cfg(target_os = "macos")]
mod carbon {
    //! The real Carbon binding (HIToolbox hot keys, Text Input Sources, UCKeyTranslate).

    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr;

    use parking_lot::Mutex;

    use super::{
        CF_NOTIFICATION_DELIVER_IMMEDIATELY, CarbonApi, CarbonHandler, EVENT_CLASS_KEYBOARD,
        EVENT_HOTKEY_EXCLUSIVE, EVENT_HOTKEY_PRESSED, EVENT_NOT_HANDLED_ERR,
        EVENT_PARAM_DIRECT_OBJECT, LayoutCallback, NO_ERR, TYPE_EVENT_HOTKEY_ID,
        UC_KEY_ACTION_DOWN, UC_KEY_TRANSLATE_NO_DEAD_KEYS, mac_character_keycodes, panic_message,
    };

    type OSStatus = i32;

    #[repr(C)]
    struct EventTypeSpec {
        event_class: u32,
        event_kind: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct EventHotKeyID {
        signature: u32,
        id: u32,
    }

    type EventHandlerProc = extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> OSStatus;
    type NotificationCallback =
        extern "C" fn(*const c_void, *mut c_void, *const c_void, *const c_void, *const c_void);

    #[link(name = "Carbon", kind = "framework")]
    unsafe extern "C" {
        fn GetApplicationEventTarget() -> *mut c_void;
        fn InstallEventHandler(
            target: *mut c_void,
            handler: EventHandlerProc,
            num_types: usize,
            list: *const EventTypeSpec,
            user_data: *mut c_void,
            out_ref: *mut *mut c_void,
        ) -> OSStatus;
        fn RemoveEventHandler(handler_ref: *mut c_void) -> OSStatus;
        fn RegisterEventHotKey(
            key_code: u32,
            modifiers: u32,
            id: EventHotKeyID,
            target: *mut c_void,
            options: u32,
            out_ref: *mut *mut c_void,
        ) -> OSStatus;
        fn UnregisterEventHotKey(hotkey_ref: *mut c_void) -> OSStatus;
        fn GetEventParameter(
            event: *mut c_void,
            name: u32,
            desired_type: u32,
            actual_type: *mut u32,
            buffer_size: usize,
            actual_size: *mut usize,
            data: *mut c_void,
        ) -> OSStatus;
        fn TISCopyCurrentKeyboardLayoutInputSource() -> *mut c_void;
        fn TISCopyCurrentASCIICapableKeyboardLayoutInputSource() -> *mut c_void;
        fn TISGetInputSourceProperty(source: *mut c_void, key: *const c_void) -> *const c_void;
        fn LMGetKbdType() -> u8;
        static kTISPropertyUnicodeKeyLayoutData: *const c_void;
        static kTISNotifySelectedKeyboardInputSourceChanged: *const c_void;
    }

    #[link(name = "CoreServices", kind = "framework")]
    unsafe extern "C" {
        fn UCKeyTranslate(
            layout: *const c_void,
            virtual_key_code: u16,
            key_action: u16,
            modifier_key_state: u32,
            keyboard_type: u32,
            options: u32,
            dead_key_state: *mut u32,
            max_length: usize,
            actual_length: *mut usize,
            unicode_string: *mut u16,
        ) -> OSStatus;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
        fn CFRelease(object: *const c_void);
        fn CFNotificationCenterGetDistributedCenter() -> *mut c_void;
        fn CFNotificationCenterAddObserver(
            center: *mut c_void,
            observer: *const c_void,
            callback: NotificationCallback,
            name: *const c_void,
            object: *const c_void,
            suspension_behavior: isize,
        );
        fn CFNotificationCenterRemoveObserver(
            center: *mut c_void,
            observer: *const c_void,
            name: *const c_void,
            object: *const c_void,
        );
    }

    /// The C side only holds raw pointers: the boxed handler and layout callback are owned here
    /// (as addresses, so the binding stays `Send`) until they are removed again.
    #[derive(Default)]
    pub(super) struct Carbon {
        handler: Mutex<usize>,
        /// (distributed center, boxed callback = observer token)
        observer: Mutex<Option<(usize, usize)>>,
    }

    extern "C" fn handler_trampoline(
        _call_ref: *mut c_void,
        event: *mut c_void,
        user_data: *mut c_void,
    ) -> OSStatus {
        // SAFETY: user_data is the boxed handler installed with this trampoline; it lives until
        // remove_handler, which also removes the trampoline.
        let handler = unsafe { &*(user_data as *const CarbonHandler) };
        // A panic must never unwind into Carbon.
        catch_unwind(AssertUnwindSafe(|| handler(event as usize))).unwrap_or_else(|payload| {
            log::error!(
                "Hotkey event handler failed: {}",
                panic_message(payload.as_ref())
            );
            EVENT_NOT_HANDLED_ERR
        })
    }

    extern "C" fn layout_trampoline(
        _center: *const c_void,
        observer: *mut c_void,
        _name: *const c_void,
        _object: *const c_void,
        _user_info: *const c_void,
    ) {
        // SAFETY: the observer token is the boxed callback; it lives until the observer is
        // removed.
        let callback = unsafe { &*(observer as *const LayoutCallback) };
        // A panic must never unwind into CoreFoundation.
        if let Err(payload) = catch_unwind(AssertUnwindSafe(callback)) {
            log::error!(
                "Keyboard layout change handler failed: {}",
                panic_message(payload.as_ref())
            );
        }
    }

    impl CarbonApi for Carbon {
        fn install_handler(&self, handler: CarbonHandler) -> (i32, usize) {
            let user_data = Box::into_raw(Box::new(handler));
            let spec = EventTypeSpec {
                event_class: EVENT_CLASS_KEYBOARD,
                event_kind: EVENT_HOTKEY_PRESSED,
            };
            let mut handler_ref: *mut c_void = ptr::null_mut();
            // SAFETY: plain Carbon call with valid pointers.
            let status = unsafe {
                InstallEventHandler(
                    GetApplicationEventTarget(),
                    handler_trampoline,
                    1,
                    &spec,
                    user_data.cast(),
                    &mut handler_ref,
                )
            };
            if status == NO_ERR {
                *self.handler.lock() = user_data as usize;
            } else {
                // SAFETY: not installed, so nothing else refers to it.
                drop(unsafe { Box::from_raw(user_data) });
            }
            (status, handler_ref as usize)
        }

        fn remove_handler(&self, handler_ref: usize) -> i32 {
            // SAFETY: handler_ref came from InstallEventHandler.
            let status = unsafe { RemoveEventHandler(handler_ref as *mut c_void) };
            let user_data = std::mem::take(&mut *self.handler.lock());
            if user_data != 0 {
                // SAFETY: the handler is removed, so Carbon no longer calls the trampoline.
                drop(unsafe { Box::from_raw(user_data as *mut CarbonHandler) });
            }
            status
        }

        fn register_hotkey(
            &self,
            keycode: u32,
            modifiers: u32,
            signature: u32,
            hotkey_id: u32,
        ) -> (i32, usize) {
            let mut hotkey_ref: *mut c_void = ptr::null_mut();
            // SAFETY: plain Carbon call with valid pointers.
            let status = unsafe {
                RegisterEventHotKey(
                    keycode,
                    modifiers,
                    EventHotKeyID {
                        signature,
                        id: hotkey_id,
                    },
                    GetApplicationEventTarget(),
                    EVENT_HOTKEY_EXCLUSIVE,
                    &mut hotkey_ref,
                )
            };
            (status, hotkey_ref as usize)
        }

        fn unregister_hotkey(&self, hotkey_ref: usize) -> i32 {
            // SAFETY: hotkey_ref came from RegisterEventHotKey.
            unsafe { UnregisterEventHotKey(hotkey_ref as *mut c_void) }
        }

        fn hotkey_id(&self, event: usize) -> (i32, u32, u32) {
            let mut hk = EventHotKeyID::default();
            // SAFETY: event is the EventRef Carbon passed to the handler; hk is large enough.
            let status = unsafe {
                GetEventParameter(
                    event as *mut c_void,
                    EVENT_PARAM_DIRECT_OBJECT,
                    TYPE_EVENT_HOTKEY_ID,
                    ptr::null_mut(),
                    size_of::<EventHotKeyID>(),
                    ptr::null_mut(),
                    (&mut hk as *mut EventHotKeyID).cast(),
                )
            };
            (status, hk.signature, hk.id)
        }

        fn layout_characters(
            &self,
            modifiers: u32,
            ascii_capable: bool,
        ) -> Option<BTreeMap<u32, String>> {
            // SAFETY: TIS copy functions return an owned reference (released below) or NULL.
            let source = unsafe {
                if ascii_capable {
                    TISCopyCurrentASCIICapableKeyboardLayoutInputSource()
                } else {
                    TISCopyCurrentKeyboardLayoutInputSource()
                }
            };
            if source.is_null() {
                return None;
            }
            // SAFETY: source is valid until CFRelease; the layout data belongs to it.
            let chars = unsafe {
                let data = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData);
                let layout = if data.is_null() {
                    ptr::null()
                } else {
                    CFDataGetBytePtr(data)
                };
                if layout.is_null() {
                    None // input methods have no Unicode key layout
                } else {
                    Some(translate_keys(layout.cast(), modifiers))
                }
            };
            // SAFETY: source was copied above.
            unsafe { CFRelease(source) };
            chars
        }

        fn observe_layout_changes(&self, callback: LayoutCallback) -> bool {
            let mut observer = self.observer.lock();
            if observer.is_some() {
                return true;
            }
            // SAFETY: plain CoreFoundation call.
            let center = unsafe { CFNotificationCenterGetDistributedCenter() };
            if center.is_null() {
                return false;
            }
            let token = Box::into_raw(Box::new(callback));
            // SAFETY: the token stays valid until stop_observing_layout_changes removes it.
            unsafe {
                CFNotificationCenterAddObserver(
                    center,
                    token as *const c_void,
                    layout_trampoline,
                    kTISNotifySelectedKeyboardInputSourceChanged,
                    ptr::null(),
                    CF_NOTIFICATION_DELIVER_IMMEDIATELY,
                );
            }
            *observer = Some((center as usize, token as usize));
            true
        }

        fn stop_observing_layout_changes(&self) {
            let Some((center, token)) = self.observer.lock().take() else {
                return;
            };
            // SAFETY: removes the observer added above, then frees its callback.
            unsafe {
                CFNotificationCenterRemoveObserver(
                    center as *mut c_void,
                    token as *const c_void,
                    kTISNotifySelectedKeyboardInputSourceChanged,
                    ptr::null(),
                );
                drop(Box::from_raw(token as *mut LayoutCallback));
            }
        }
    }

    /// What each character key types on `layout` (a `UCKeyboardLayout`) with `modifiers`.
    ///
    /// # Safety
    /// `layout` must point to valid `UCKeyboardLayout` data.
    unsafe fn translate_keys(layout: *const c_void, modifiers: u32) -> BTreeMap<u32, String> {
        // SAFETY: plain HIToolbox call.
        let kbd_type = u32::from(unsafe { LMGetKbdType() });
        let state = (modifiers >> 8) & 0xFF;
        let mut chars = BTreeMap::new();
        for keycode in mac_character_keycodes() {
            let mut dead = 0u32;
            let mut length = 0usize;
            let mut buf = [0u16; 4];
            // SAFETY: the out-pointers are valid and buf holds max_length units.
            let status = unsafe {
                UCKeyTranslate(
                    layout,
                    keycode,
                    UC_KEY_ACTION_DOWN,
                    state,
                    kbd_type,
                    UC_KEY_TRANSLATE_NO_DEAD_KEYS,
                    &mut dead,
                    buf.len(),
                    &mut length,
                    buf.as_mut_ptr(),
                )
            };
            let count = length.min(buf.len());
            if status != NO_ERR || count == 0 {
                continue;
            }
            let text: String = String::from_utf16_lossy(&buf[..count])
                .chars()
                .filter(|c| !c.is_control())
                .collect();
            if !text.is_empty() {
                chars.insert(u32::from(keycode), text);
            }
        }
        chars
    }
}

// ----------------------------------------------------------------------------- X11
// The manager is plain logic over `XDisplay` (a fake in tests) plus POSIX pipes, so it is built
// on every Unix; only the x11rb display needs Linux.

#[cfg(unix)]
pub use x11::X11HotkeyManager;

#[cfg(unix)]
mod x11 {
    use std::os::fd::RawFd;

    use super::*;

    // Values from X.h.
    pub(super) const X_SHIFT_MASK: u16 = 1 << 0;
    pub(super) const X_LOCK_MASK: u16 = 1 << 1;
    pub(super) const X_CONTROL_MASK: u16 = 1 << 2;
    pub(super) const X_MOD1_MASK: u16 = 1 << 3; // Alt
    pub(super) const X_MOD2_MASK: u16 = 1 << 4; // NumLock on virtually every keymap
    pub(super) const X_MOD4_MASK: u16 = 1 << 6; // Super
    pub(super) const X_MAPPING_MODIFIER: u8 = 0;
    pub(super) const X_MAPPING_KEYBOARD: u8 = 1;
    pub(super) const X_GRAB_MODE_ASYNC: u8 = 1;
    pub(super) const X_PROPERTY_CHANGE_MASK: u32 = 1 << 22;
    /// Root window property in which setxkbmap and the desktops record the XKB rules they
    /// applied: "rules\0model\0layout\0variant\0options" (options comma-separated).
    pub(super) const XKB_RULES_NAMES: &str = "_XKB_RULES_NAMES";

    /// [`X11HotkeyManager::note`](HotkeyManager::note) while another program grabs a Super key
    /// on its own (see `X11Backend::check_super_alone`).
    pub(super) const SUPER_TAKEN_NOTE: &str = "Another program uses the Super key on its own \
        (Xfce opens its menu with it), so shortcuts with Super work only when Ctrl or Alt is \
        pressed before Super.";

    /// How an XKB option repurposes keys of a hotkey (see [`X_KEY_OPTIONS`]).
    pub(super) struct XkbKeyOption {
        /// The modifiers that trigger it when held together.
        pub(super) modifiers: &'static [&'static str],
        /// The key pressed with them, or `None` when the modifiers alone trigger it.
        pub(super) key: Option<&'static str>,
        /// What it does, completing "…, which <effect>" in the user-facing message.
        pub(super) effect: &'static str,
    }

    const SWITCHES_LAYOUT: &str = "switches the keyboard layout";
    const ACTS_AS_ALTGR: &str = "acts as AltGr";
    const ACTS_AS_LEVEL5: &str = "acts as a level-5 key";
    const WIN_AS_CTRL: &str = "turns the Win keys into Ctrl";
    const WIN_AS_ALT: &str = "turns the Win keys into Alt";

    /// Table entry: `modifiers` (+ `key`) do `effect` instead of their usual job.
    const fn xkb_option(
        effect: &'static str,
        modifiers: &'static [&'static str],
        key: Option<&'static str>,
    ) -> XkbKeyOption {
        XkbKeyOption {
            modifiers,
            key,
            effect,
        }
    }

    /// XKB options (`setxkbmap -option`, or a desktop's keyboard settings) that take modifiers
    /// of a hotkey for something else, read from xkeyboard-config's `symbols/group`,
    /// `symbols/level3` and `rules/base`. A grab of such a hotkey succeeds, yet pressing it does
    /// not work as intended, so it is refused with the reason given here:
    ///
    /// * Pair toggles: with `grp:alt_shift_toggle` the second key pressed of Alt and Shift sends
    ///   `ISO_Next_Group` instead of its modifier, in either order (Shift is `PC_ALT_LEVEL2`
    ///   [Shift_L, ISO_Next_Group], Alt `TWO_LEVEL` [NoSymbol, ISO_Next_Group]); Ctrl+Shift,
    ///   Ctrl+Alt and Ctrl+Win toggles work alike. The chord cannot be formed: the layout
    ///   changes and the rest reaches other applications (Ctrl+Alt+Shift+T arrives as
    ///   Ctrl+Alt+T, which opens a terminal).
    /// * Single-key switches (`grp:lalt_toggle`, `grp:lwin_switch`, the Win and Alt `lv3`
    ///   switches, …) turn the (left) key into a layout or AltGr key, so it no longer adds its
    ///   modifier at all. So do the layout *selectors* (`grp:win_menu_select`,
    ///   `grp:ctrl_select`; `grp:win_menu_switch` and `grp:lctrl_rctrl_switch` are their older
    ///   names, still accepted), the level-5 Win keys (`lv5:lwin_switch_lock`) and
    ///   `altwin:ctrl_win` / `altwin:alt_win`, which make both Win keys Ctrl or Alt: no Super is
    ///   left.
    /// * Space toggles: the hotkey would fire, but every press also switches the layout.
    ///
    /// Right-hand-only variants (`grp:ralt_rshift_toggle`, `grp:rctrl_rshift_toggle`,
    /// `grp:rctrl_ralt_toggle`, `grp:toggle`, `grp:rwin_*`, `lv3:ralt_*`) and
    /// both-keys-of-a-kind toggles (`grp:alts_toggle`, `grp:ctrls_toggle`, `grp:shifts_toggle`)
    /// leave the left-hand keys that chords are pressed with working, so they are not listed.
    pub(super) const X_KEY_OPTIONS: [(&str, XkbKeyOption); 33] = [
        // Pair toggles: the chord cannot be formed.
        (
            "grp:alt_shift_toggle",
            xkb_option(SWITCHES_LAYOUT, &["alt", "shift"], None),
        ),
        (
            "grp:alt_shift_toggle_bidir",
            xkb_option(SWITCHES_LAYOUT, &["alt", "shift"], None),
        ),
        (
            "grp:lalt_lshift_toggle",
            xkb_option(SWITCHES_LAYOUT, &["alt", "shift"], None),
        ),
        (
            "grp:ctrl_shift_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "shift"], None),
        ),
        (
            "grp:ctrl_shift_toggle_bidir",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "shift"], None),
        ),
        (
            "grp:lctrl_lshift_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "shift"], None),
        ),
        (
            "grp:ctrl_alt_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "alt"], None),
        ),
        (
            "grp:ctrl_alt_toggle_bidir",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "alt"], None),
        ),
        (
            "grp:lctrl_lalt_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "alt"], None),
        ),
        (
            "grp:lctrl_lwin_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "meta"], None),
        ),
        (
            "grp:lctrl_lwin_rctrl_menu",
            xkb_option(SWITCHES_LAYOUT, &["ctrl", "meta"], None),
        ),
        // Single (left) keys that stop being modifiers.
        (
            "grp:lalt_toggle",
            xkb_option(SWITCHES_LAYOUT, &["alt"], None),
        ),
        ("grp:lswitch", xkb_option(SWITCHES_LAYOUT, &["alt"], None)),
        (
            "grp:lctrl_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl"], None),
        ),
        (
            "grp:lshift_toggle",
            xkb_option(SWITCHES_LAYOUT, &["shift"], None),
        ),
        (
            "grp:lwin_toggle",
            xkb_option(SWITCHES_LAYOUT, &["meta"], None),
        ),
        (
            "grp:lwin_switch",
            xkb_option(SWITCHES_LAYOUT, &["meta"], None),
        ),
        (
            "grp:win_switch",
            xkb_option(SWITCHES_LAYOUT, &["meta"], None),
        ),
        (
            "grp:win_menu_select",
            xkb_option(SWITCHES_LAYOUT, &["meta"], None),
        ),
        (
            "grp:win_menu_switch",
            xkb_option(SWITCHES_LAYOUT, &["meta"], None),
        ),
        (
            "grp:ctrl_select",
            xkb_option(SWITCHES_LAYOUT, &["ctrl"], None),
        ),
        (
            "grp:lctrl_rctrl_switch",
            xkb_option(SWITCHES_LAYOUT, &["ctrl"], None),
        ),
        ("lv3:lalt_switch", xkb_option(ACTS_AS_ALTGR, &["alt"], None)),
        ("lv3:alt_switch", xkb_option(ACTS_AS_ALTGR, &["alt"], None)),
        (
            "lv3:lwin_switch",
            xkb_option(ACTS_AS_ALTGR, &["meta"], None),
        ),
        ("lv3:win_switch", xkb_option(ACTS_AS_ALTGR, &["meta"], None)),
        (
            "lv5:lwin_switch_lock",
            xkb_option(ACTS_AS_LEVEL5, &["meta"], None),
        ),
        (
            "lv5:lwin_switch_lock_cancel",
            xkb_option(ACTS_AS_LEVEL5, &["meta"], None),
        ),
        ("altwin:ctrl_win", xkb_option(WIN_AS_CTRL, &["meta"], None)),
        ("altwin:alt_win", xkb_option(WIN_AS_ALT, &["meta"], None)),
        // Space toggles: every press of the hotkey would also switch the layout.
        (
            "grp:alt_space_toggle",
            xkb_option(SWITCHES_LAYOUT, &["alt"], Some("space")),
        ),
        (
            "grp:ctrl_space_toggle",
            xkb_option(SWITCHES_LAYOUT, &["ctrl"], Some("space")),
        ),
        (
            "grp:win_space_toggle",
            xkb_option(SWITCHES_LAYOUT, &["meta"], Some("space")),
        ),
    ];

    pub(super) const X_RELEVANT_MASK: u16 =
        X_SHIFT_MASK | X_CONTROL_MASK | X_MOD1_MASK | X_MOD4_MASK;

    const F_KEYSYM_NAMES: [&str; 24] = [
        "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14",
        "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24",
    ];

    /// The X11 keysym name of canonical `key`.
    pub(super) fn x11_keysym_name(key: &'static str) -> &'static str {
        if let Some(n) = key.strip_prefix('f').and_then(|n| n.parse::<usize>().ok()) {
            return F_KEYSYM_NAMES[n - 1];
        }
        match key {
            "enter" => "Return",
            "tab" => "Tab",
            "escape" => "Escape",
            "backspace" => "BackSpace",
            "insert" => "Insert",
            "delete" => "Delete",
            "home" => "Home",
            "end" => "End",
            "pageup" => "Prior",
            "pagedown" => "Next",
            "left" => "Left",
            "right" => "Right",
            "up" => "Up",
            "down" => "Down",
            "quote" => "apostrophe",
            // letters, digits, space and the other punctuation keep their names
            other => other,
        }
    }

    /// The keysym called `name` (XStringToKeysym for the names this module uses), 0 if unknown.
    pub(super) fn x11_string_to_keysym(name: &str) -> u32 {
        let mut chars = name.chars();
        if let (Some(ch), None) = (chars.next(), chars.next())
            && ch.is_ascii_alphanumeric()
        {
            return u32::from(ch); // Latin-1 keysyms equal their code points
        }
        if let Some(n) = name.strip_prefix('F').and_then(|n| n.parse::<u32>().ok())
            && (1..=35).contains(&n)
        {
            return 0xFFBE + n - 1;
        }
        match name {
            "space" => 0x20,
            "apostrophe" => 0x27,
            "comma" => 0x2C,
            "minus" => 0x2D,
            "period" => 0x2E,
            "slash" => 0x2F,
            "semicolon" => 0x3B,
            "equal" => 0x3D,
            "bracketleft" => 0x5B,
            "backslash" => 0x5C,
            "bracketright" => 0x5D,
            "grave" => 0x60,
            "BackSpace" => 0xFF08,
            "Tab" => 0xFF09,
            "Return" => 0xFF0D,
            "Escape" => 0xFF1B,
            "Home" => 0xFF50,
            "Left" => 0xFF51,
            "Up" => 0xFF52,
            "Right" => 0xFF53,
            "Down" => 0xFF54,
            "Prior" => 0xFF55,
            "Next" => 0xFF56,
            "End" => 0xFF57,
            "Insert" => 0xFF63,
            "Num_Lock" => 0xFF7F,
            "Super_L" => 0xFFEB,
            "Super_R" => 0xFFEC,
            "Delete" => 0xFFFF,
            _ => 0,
        }
    }

    pub(super) fn x11_modifiers<'a>(modifiers: impl IntoIterator<Item = &'a str>) -> u16 {
        modifiers.into_iter().fold(0, |mask, m| {
            mask | match m {
                "ctrl" => X_CONTROL_MASK,
                "alt" => X_MOD1_MASK,
                "shift" => X_SHIFT_MASK,
                _ => X_MOD4_MASK,
            }
        })
    }

    /// Modifier masks to grab in addition to the real ones.
    ///
    /// A passive grab matches the modifier state exactly, so without these variants a hotkey
    /// would silently stop working whenever Caps Lock or Num Lock is on.
    pub(super) fn x11_lock_variants(numlock_mask: u16) -> Vec<u16> {
        let mut variants = Vec::new();
        for mask in [0, X_LOCK_MASK, numlock_mask, X_LOCK_MASK | numlock_mask] {
            if !variants.contains(&mask) {
                variants.push(mask);
            }
        }
        variants
    }

    /// The XKB options recorded in an `_XKB_RULES_NAMES` property value.
    ///
    /// The value is five NUL-separated fields (rules, model, layout, variant, options).
    /// Anything malformed yields no options.
    pub(super) fn x11_parse_xkb_options(value: Option<&[u8]>) -> Vec<String> {
        let Some(value) = value else {
            return Vec::new();
        };
        // ICCCM STRING (Latin-1); option names are ASCII.
        let raw: String = value.iter().map(|b| char::from(*b)).collect();
        let fields: Vec<&str> = raw.split('\0').collect();
        if fields.len() < 5 {
            return Vec::new();
        }
        fields[4]
            .split(',')
            .map(str::trim)
            .filter(|o| !o.is_empty())
            .map(String::from)
            .collect()
    }

    /// The XKB options of the X server (read-only `GetProperty` on the root window).
    fn x11_read_xkb_options(display: &mut dyn XDisplay, atom: u32) -> Result<Vec<String>, String> {
        let value = display.get_root_property(atom)?;
        Ok(x11_parse_xkb_options(value.as_deref()))
    }

    /// Why keys of `hotkey` are taken by one of the XKB `options`, or `None`.
    ///
    /// See [`X_KEY_OPTIONS`]; the first matching option is named, for example
    /// `"Ctrl+Alt+Shift+T includes Alt+Shift, which switches the keyboard layout (XKB option
    /// grp:alt_shift_toggle)"`.
    pub(super) fn x11_layout_switch_conflict<S: AsRef<str>>(
        hotkey: &Hotkey,
        options: &[S],
    ) -> Option<String> {
        for option in options {
            let option = option.as_ref();
            let Some((_, taken)) = X_KEY_OPTIONS.iter().find(|(name, _)| *name == option) else {
                continue;
            };
            if !taken.modifiers.iter().all(|m| hotkey.has(m))
                || taken.key.is_some_and(|key| key != hotkey.key)
            {
                continue;
            }
            let label = format_hotkey(hotkey, Some(false));
            let mut chord = modifier_labels(taken.modifiers.iter().copied(), Platform::CURRENT);
            if let Some(key) = taken.key {
                chord.push(key_label(key, false));
            }
            let chord = chord.join("+");
            // "Alt+Space switches …" rather than "Alt+Space includes Alt+Space, which …".
            let subject = if chord == label {
                label
            } else {
                format!("{label} includes {chord}, which")
            };
            return Some(format!("{subject} {} (XKB option {option})", taken.effect));
        }
        None
    }

    /// An event of the hotkey connection.
    #[derive(Debug, Clone)]
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))] // made by the x11rb display and the tests
    pub(super) enum XEvent {
        KeyPress { detail: u8, state: u16, time: u32 },
        // No state: releases are matched by keycode alone (see `handle_event_in`).
        KeyRelease { detail: u8, time: u32 },
        MappingNotify { request: u8 },
        PropertyNotify { atom: u32 },
        Other,
    }

    /// Records the first X error of a series of requests (Python's `_XErrorCatcher`), so a
    /// failed grab (`BadAccess`: another client owns the combination) is noticed.
    #[derive(Default)]
    pub(super) struct XErrorCatcher {
        pub(super) error: Option<String>,
    }

    impl XErrorCatcher {
        pub(super) fn catch(&mut self, error: impl Into<String>) {
            if self.error.is_none() {
                self.error = Some(error.into());
            }
        }
    }

    /// The X11 connection of the hotkey thread (the requests this module makes on the root
    /// window of the default screen).
    pub(super) trait XDisplay: Send {
        fn string_to_keysym(&self, name: &str) -> u32 {
            x11_string_to_keysym(name)
        }

        /// The primary keycode bound to `keysym` (lowest column, then lowest code), or 0.
        fn keysym_to_keycode(&mut self, keysym: u32) -> u8;

        /// The keycodes of the eight modifiers.
        fn get_modifier_mapping(&mut self) -> Result<Vec<Vec<u8>>, String>;

        fn grab_key(
            &mut self,
            keycode: u8,
            modifiers: u16,
            owner_events: bool,
            pointer_mode: u8,
            keyboard_mode: u8,
            onerror: &mut XErrorCatcher,
        );

        fn ungrab_key(&mut self, keycode: u8, modifiers: u16, onerror: &mut XErrorCatcher);

        /// A round trip: every error of the requests before it has arrived.
        fn sync(&mut self) -> Result<(), String>;

        fn flush(&mut self);

        /// Update the cached keyboard mapping after a `MappingNotify`.
        fn refresh_keyboard_mapping(&mut self, event: &XEvent);

        /// The atom called `name`; 0 if `only_if_exists` and there is none.
        fn intern_atom(&mut self, name: &str, only_if_exists: bool) -> Result<u32, String>;

        /// The value of root window property `atom` (`None` if it does not exist).
        fn get_root_property(&mut self, atom: u32) -> Result<Option<Vec<u8>>, String>;

        /// Select `event_mask` on the root window.
        fn select_root_events(&mut self, event_mask: u32) -> Result<(), String>;

        /// The next event already received (or readable without blocking), if any.
        fn poll_event(&mut self) -> Result<Option<XEvent>, String>;

        fn fileno(&self) -> RawFd;

        fn close(&mut self);
    }

    /// Opens an X11 connection (`None`: the `DISPLAY` one).
    pub(super) type DisplayFactory =
        Box<dyn Fn(Option<&str>) -> Result<Box<dyn XDisplay>, String> + Send + Sync>;

    fn open_display(name: Option<&str>) -> Result<Box<dyn XDisplay>, String> {
        #[cfg(target_os = "linux")]
        {
            xlib::open(name)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = name;
            Err("X11 is not supported on this system".to_string())
        }
    }

    /// Only touched on the hotkey thread (tests drive it synchronously).
    pub(super) struct X11State {
        pub(super) display: Option<Box<dyn XDisplay>>,
        pub(super) numlock_mask: u16,
        /// The _XKB_RULES_NAMES atom (0: options not followed).
        pub(super) rules_atom: u32,
        /// Another client grabs Super alone (`check_super_alone`).
        pub(super) super_taken: bool,
        /// binding id → (keycode, modifier mask)
        pub(super) grabs: HashMap<u32, (u8, u16)>,
        /// (keycode, modifier mask) → binding id
        pub(super) lookup: HashMap<(u8, u16), u32>,
        pub(super) held: HashSet<u32>,
        pub(super) last_release: HashMap<u32, u32>,
    }

    impl Default for X11State {
        fn default() -> Self {
            X11State {
                display: None,
                numlock_mask: X_MOD2_MASK,
                rules_atom: 0,
                super_taken: false,
                grabs: HashMap::new(),
                lookup: HashMap::new(),
                held: HashSet::new(),
                last_release: HashMap::new(),
            }
        }
    }

    /// Block until an event or a wake-up: no periodic wake-ups while no key is pressed.
    pub(super) const SELECT_TIMEOUT: Option<Duration> = None;

    /// The X11 manager's state (see [`X11HotkeyManager`]).
    pub(super) struct X11Backend {
        pub(super) core: Core,
        pub(super) threaded: Threaded,
        /// The note this manager was created with; the Super-key remark is added to it.
        base_note: Option<String>,
        display_name: Option<String>,
        display_factory: DisplayFactory,
        /// The wake-up pipe (read end, write end); -1 when closed.
        pipe: Mutex<(RawFd, RawFd)>,
        /// XKB options of the server, written by the hotkey thread only (`None` while it is not
        /// running); other threads may read them.
        pub(super) xkb_options: Mutex<Option<Vec<String>>>,
        pub(super) state: Mutex<X11State>,
    }

    native_manager!(
        /// Passive key grabs on the X11 root window, served by a dedicated thread.
        ///
        /// The thread owns its own connection. Its loop blocks in `poll` on that connection and
        /// a wake-up pipe, without a timeout: every cross-thread call and `stop` write to the
        /// pipe after queuing their work, and events that the connection already buffered are
        /// drained before waiting. The thread therefore never wakes up while no key is pressed,
        /// and `stop` still returns promptly.
        ///
        /// After a keyboard mapping change every binding is grabbed again at its new keycode;
        /// one whose key vanished (`setxkbmap ru`) stays known, is reported as inactive and is
        /// grabbed again by the next mapping change that restores it. A hotkey refused when it
        /// was registered, because its key was missing or a layout switch took its keys, is
        /// kept and retried the same way.
        ///
        /// A Super key that another client grabs on its own (Xubuntu opens its menu with it)
        /// takes the whole keyboard while it is held, so a chord pressed Super first never
        /// reaches these grabs. That cannot be fixed from here; it is detected and explained in
        /// the note.
        ///
        /// A hotkey that contains a configured keyboard-layout switch (Alt+Shift with
        /// `grp:alt_shift_toggle`, see `X_KEY_OPTIONS`) is refused, because `XGrabKey` would
        /// accept it although it can never be pressed. The options are read from the root
        /// window's `_XKB_RULES_NAMES` and followed through `PropertyNotify` (root properties
        /// change on focus changes at most, which the loop ignores after comparing one atom)
        /// and mapping changes.
        X11HotkeyManager(X11Backend)
    );

    impl X11HotkeyManager {
        /// A manager for X display `display_name` (`None`: `$DISPLAY`). Connects only when the
        /// first hotkey is registered.
        pub fn new(display_name: Option<String>, note: Option<String>) -> X11HotkeyManager {
            X11HotkeyManager::with_display_factory(display_name, note, Box::new(open_display))
        }

        pub(super) fn with_display_factory(
            display_name: Option<String>,
            note: Option<String>,
            display_factory: DisplayFactory,
        ) -> X11HotkeyManager {
            X11HotkeyManager(Arc::new(X11Backend {
                core: Core::new("x11", note.clone()),
                threaded: Threaded::new(),
                base_note: note,
                display_name,
                display_factory,
                pipe: Mutex::new((-1, -1)),
                xkb_options: Mutex::new(None),
                state: Mutex::new(X11State::default()),
            }))
        }
    }

    /// `poll(2)` on `fds` until one is readable; which ones are.
    fn wait_readable(fds: &[RawFd], timeout: Option<Duration>) -> std::io::Result<Vec<bool>> {
        let mut polled: Vec<libc::pollfd> = fds
            .iter()
            .map(|fd| libc::pollfd {
                fd: *fd,
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        let timeout_ms = timeout.map_or(-1, |t| t.as_millis().min(i32::MAX as u128) as i32);
        loop {
            // SAFETY: polled is a valid array of pollfd for the duration of the call.
            let result = unsafe {
                libc::poll(
                    polled.as_mut_ptr(),
                    polled.len() as libc::nfds_t,
                    timeout_ms,
                )
            };
            if result >= 0 {
                break;
            }
            let err = std::io::Error::last_os_error();
            if err.kind() != std::io::ErrorKind::Interrupted {
                return Err(err);
            }
        }
        Ok(polled.iter().map(|p| p.revents != 0).collect())
    }

    impl X11Backend {
        fn drain_wake_pipe(&self, read_fd: RawFd) {
            // Non-blocking: read until empty.
            let mut buf = [0u8; 4096];
            // SAFETY: buf is valid for its length.
            while unsafe { libc::read(read_fd, buf.as_mut_ptr().cast(), buf.len()) } > 0 {}
        }

        /// The server's XKB options, from any thread.
        ///
        /// The hotkey thread keeps them up to date while it runs; otherwise (`doctor`, a
        /// settings dialog before the first registration) a short-lived connection reads them.
        fn current_xkb_options(&self) -> Result<Vec<String>, String> {
            if let Some(options) = self.xkb_options.lock().clone() {
                return Ok(options);
            }
            let mut display = (self.display_factory)(self.display_name.as_deref())?;
            let options = display.intern_atom(XKB_RULES_NAMES, true).and_then(|atom| {
                if atom == 0 {
                    Ok(Vec::new())
                } else {
                    x11_read_xkb_options(&mut *display, atom)
                }
            });
            display.close();
            options
        }

        /// Read the XKB options and ask for `PropertyNotify` when they change.
        ///
        /// Hotkey thread only. Desktops may apply the user's layout switch after this app
        /// started (at login), and `setxkbmap` writes the property only after the keymap change
        /// that sends `MappingNotify`, so the property itself is watched. Failing that, options
        /// are simply not checked.
        pub(super) fn follow_xkb_options(&self, st: &mut X11State) {
            if let Some(display) = st.display.as_mut() {
                let followed = display
                    .intern_atom(XKB_RULES_NAMES, false)
                    .and_then(|atom| {
                        st.rules_atom = atom;
                        display.select_root_events(X_PROPERTY_CHANGE_MASK)
                    });
                if let Err(err) = followed {
                    log::debug!("Cannot follow the XKB layout options: {err}");
                    st.rules_atom = 0;
                }
            }
            self.refresh_xkb_options(st);
        }

        /// Re-read the XKB options (hotkey thread only); returns whether they changed.
        fn refresh_xkb_options(&self, st: &mut X11State) -> bool {
            let old = self.xkb_options.lock().clone();
            let mut new = old.clone().unwrap_or_default();
            if st.rules_atom != 0
                && let Some(display) = st.display.as_mut()
            {
                match x11_read_xkb_options(&mut **display, st.rules_atom) {
                    Ok(options) => new = options,
                    // Keep what was known: never flip-flop on a failed read.
                    Err(err) => log::debug!("Reading the XKB layout options failed: {err}"),
                }
            }
            let changed = old.as_ref() != Some(&new);
            if changed {
                let shown = if new.is_empty() {
                    "(none)".to_string()
                } else {
                    new.join(",")
                };
                log::debug!("XKB options: {shown}");
            }
            *self.xkb_options.lock() = Some(new);
            changed
        }

        fn keycode_for(st: &mut X11State, key: &'static str) -> u8 {
            let Some(display) = st.display.as_mut() else {
                return 0;
            };
            let keysym = display.string_to_keysym(x11_keysym_name(key));
            if keysym == 0 {
                return 0;
            }
            display.keysym_to_keycode(keysym)
        }

        pub(super) fn find_numlock_mask(st: &mut X11State) -> u16 {
            if let Some(display) = st.display.as_mut() {
                let keysym = display.string_to_keysym("Num_Lock");
                let keycode = display.keysym_to_keycode(keysym);
                if keycode != 0 {
                    match display.get_modifier_mapping() {
                        Ok(mapping) => {
                            for (index, keycodes) in mapping.iter().enumerate() {
                                if keycodes.contains(&keycode) {
                                    return 1 << index;
                                }
                            }
                        }
                        Err(err) => log::debug!("Could not determine the NumLock modifier: {err}"),
                    }
                }
            }
            X_MOD2_MASK
        }

        pub(super) fn grab_binding(&self, binding: &Binding) -> bool {
            let mut st = self.state.lock();
            self.grab_binding_in(&mut st, binding)
        }

        fn grab_binding_in(&self, st: &mut X11State, binding: &Binding) -> bool {
            let label = format_hotkey(&binding.hotkey, Some(false));
            // XGrabKey accepts a chord that the layout switch makes impossible to press;
            // without this check the hotkey would silently never fire.
            let options = self.xkb_options.lock().clone().unwrap_or_default();
            if let Some(switch) = x11_layout_switch_conflict(&binding.hotkey, &options) {
                // Grabbed as soon as the user moves the switch elsewhere (regrab_all).
                binding.retry_on_layout_change.store(true, Ordering::SeqCst);
                return self.core.reject(binding, switch);
            }
            let keycode = Self::keycode_for(st, binding.hotkey.key);
            if keycode == 0 {
                binding.retry_on_layout_change.store(true, Ordering::SeqCst);
                return self.core.reject(
                    binding,
                    format!("the key of {label} is not on the current keyboard layout"),
                );
            }
            let mods = x11_modifiers(binding.hotkey.modifiers().iter().copied());
            if !Self::grab(st, keycode, mods) {
                return self.core.reject(
                    binding,
                    format!("{label} is already in use by another application"),
                );
            }
            st.grabs.insert(binding.id, (keycode, mods));
            st.lookup.insert((keycode, mods), binding.id);
            true
        }

        pub(super) fn ungrab_binding(&self, binding_id: u32) {
            let mut st = self.state.lock();
            Self::ungrab(&mut st, binding_id);
            if let Some(display) = st.display.as_mut() {
                display.flush();
            }
        }

        fn grab(st: &mut X11State, keycode: u8, mods: u16) -> bool {
            let variants = x11_lock_variants(st.numlock_mask);
            let Some(display) = st.display.as_mut() else {
                return false;
            };
            let mut catcher = XErrorCatcher::default();
            for variant in &variants {
                display.grab_key(
                    keycode,
                    mods | variant,
                    true,
                    X_GRAB_MODE_ASYNC,
                    X_GRAB_MODE_ASYNC,
                    &mut catcher,
                );
            }
            // Grab errors arrive asynchronously; a round trip guarantees they were seen.
            if let Err(err) = display.sync() {
                catcher.catch(err);
            }
            if catcher.error.is_none() {
                return true;
            }
            // Release whichever variants did succeed so nothing half-registered lingers.
            let mut quiet = XErrorCatcher::default();
            for variant in &variants {
                display.ungrab_key(keycode, mods | variant, &mut quiet);
            }
            let _ = display.sync();
            false
        }

        /// Note whether another client grabs a Super key on its own (hotkey thread only).
        ///
        /// Xubuntu binds Super_L and Super_R without modifiers to its menu, and xfsettingsd
        /// grabs them asynchronously and acts on the release. X activates a passive grab only
        /// while the keyboard is not grabbed, so with Super pressed first the whole keyboard
        /// belongs to that client until Super is released, and Ctrl+Alt+Super+T never reaches
        /// this manager's grab (Ctrl or Alt first works). Such a grab is found by asking for the
        /// same one, which X refuses with BadAccess; a grab that is granted is released at once.
        /// The result is explained in the note.
        pub(super) fn check_super_alone(&self, st: &mut X11State) {
            let mut taken = false;
            for name in ["Super_L", "Super_R"] {
                let variants = x11_lock_variants(st.numlock_mask);
                let Some(display) = st.display.as_mut() else {
                    return;
                };
                let keysym = display.string_to_keysym(name);
                let keycode = if keysym == 0 {
                    0
                } else {
                    display.keysym_to_keycode(keysym)
                };
                if keycode == 0 {
                    continue;
                }
                if !Self::grab(st, keycode, 0) {
                    taken = true;
                    break;
                }
                let Some(display) = st.display.as_mut() else {
                    return;
                };
                let mut quiet = XErrorCatcher::default();
                for variant in variants {
                    display.ungrab_key(keycode, variant, &mut quiet);
                }
                if let Err(err) = display.sync() {
                    log::debug!("Checking the Super key grabs failed: {err}");
                    return;
                }
            }
            if taken != st.super_taken {
                st.super_taken = taken;
                if taken {
                    log::warn!(
                        "Another program grabs the Super key on its own: {SUPER_TAKEN_NOTE}"
                    );
                }
            }
            let parts: Vec<&str> = [self.base_note.as_deref(), taken.then_some(SUPER_TAKEN_NOTE)]
                .into_iter()
                .flatten()
                .filter(|p| !p.is_empty())
                .collect();
            self.core
                .set_note((!parts.is_empty()).then(|| parts.join(" ")));
        }

        fn ungrab(st: &mut X11State, binding_id: u32) {
            let grab = st.grabs.remove(&binding_id);
            st.held.remove(&binding_id);
            st.last_release.remove(&binding_id);
            let Some(grab) = grab else {
                return;
            };
            st.lookup.remove(&grab);
            let (keycode, mods) = grab;
            let variants = x11_lock_variants(st.numlock_mask);
            if let Some(display) = st.display.as_mut() {
                let mut quiet = XErrorCatcher::default();
                for variant in variants {
                    display.ungrab_key(keycode, mods | variant, &mut quiet);
                }
            }
        }

        /// Re-establish grabs after a keyboard layout or modifier mapping change.
        ///
        /// The wanted bindings come from the live table, not from `grabs`: a binding whose grab
        /// failed on an earlier change (its key was missing) must be retried now that the key
        /// may be back. A registration in flight on this thread has already grabbed but is not
        /// yet in `bindings`, so grabbed `by_id` entries are included too. Bindings being
        /// removed have already left both tables, so they are not grabbed again.
        fn regrab_all(&self, st: &mut X11State) {
            let wanted: Vec<Arc<Binding>> = {
                let table = self.core.table.lock();
                let mut wanted: Vec<Arc<Binding>> = table.bindings.values().cloned().collect();
                for (id, binding) in &table.by_id {
                    if st.grabs.contains_key(id) && !wanted.iter().any(|b| b.id == *id) {
                        wanted.push(binding.clone());
                    }
                }
                wanted
            };
            let grabbed: Vec<u32> = st.grabs.keys().copied().collect();
            for binding_id in grabbed {
                Self::ungrab(st, binding_id);
            }
            st.numlock_mask = Self::find_numlock_mask(st);
            self.check_super_alone(st); // the Super keys may have new keycodes
            for binding in wanted {
                let ok = self.grab_binding_in(st, &binding); // logs why it failed
                if !ok {
                    log::info!(
                        "Hotkey {:?} is retried on the next keyboard mapping change",
                        binding.name
                    );
                }
                self.core.set_active(&binding, ok);
            }
        }

        /// Handle one event; a matching key press dispatches its callback (after the state
        /// lock is released, so the callback may call back into the manager).
        pub(super) fn handle_event(&self, event: XEvent) {
            let fire = self.handle_event_in(&mut self.state.lock(), &event);
            if let Some(binding_id) = fire {
                self.core.dispatch(binding_id);
            }
        }

        fn handle_event_in(&self, st: &mut X11State, event: &XEvent) -> Option<u32> {
            match *event {
                XEvent::MappingNotify { request } => {
                    if request == X_MAPPING_KEYBOARD || request == X_MAPPING_MODIFIER {
                        if let Some(display) = st.display.as_mut() {
                            display.refresh_keyboard_mapping(event);
                        }
                        self.refresh_xkb_options(st);
                        self.regrab_all(st);
                    }
                    None
                }
                XEvent::PropertyNotify { atom } => {
                    // Any root property change arrives here (focus changes, window lists); only
                    // a change of the XKB options matters.
                    if st.rules_atom != 0 && atom == st.rules_atom && self.refresh_xkb_options(st) {
                        self.regrab_all(st);
                    }
                    None
                }
                XEvent::KeyRelease { detail, time } => {
                    // A release carries the modifiers held just before it, and users often lift
                    // Ctrl/Alt before the key: match releases by keycode alone, or the binding
                    // would stay "held" and swallow every later press.
                    let released: Vec<u32> = st
                        .grabs
                        .iter()
                        .filter(|(_, (code, _))| *code == detail)
                        .map(|(id, _)| *id)
                        .collect();
                    for id in released {
                        st.held.remove(&id);
                        st.last_release.insert(id, time);
                    }
                    None
                }
                XEvent::KeyPress {
                    detail,
                    state,
                    time,
                } => {
                    let binding_id = *st.lookup.get(&(detail, state & X_RELEVANT_MASK))?;
                    // X auto-repeat sends Release+Press pairs with identical timestamps while a
                    // key is held; fire once per physical press (the MOD_NOREPEAT equivalent).
                    if st.held.contains(&binding_id) {
                        return None;
                    }
                    if let Some(last_release) = st.last_release.get(&binding_id)
                        && time.wrapping_sub(*last_release) <= 1
                    {
                        return None;
                    }
                    st.held.insert(binding_id);
                    Some(binding_id)
                }
                XEvent::Other => None,
            }
        }
    }

    impl Native for X11Backend {
        fn core(&self) -> &Core {
            &self.core
        }

        fn backend_start(self: &Arc<Self>) -> bool {
            threaded_backend_start(self)
        }

        fn backend_stop(self: &Arc<Self>) {
            threaded_backend_stop(self)
        }

        fn native_register(self: &Arc<Self>, binding: &Arc<Binding>) -> bool {
            let (this, binding) = (self.clone(), binding.clone());
            call_on_thread(self, move || this.grab_binding(&binding), false)
        }

        fn native_unregister(self: &Arc<Self>, binding: &Arc<Binding>) {
            let (this, id) = (self.clone(), binding.id);
            call_on_thread(self, move || this.ungrab_binding(id), ())
        }

        fn native_layout_conflict(self: &Arc<Self>, hotkey: &Hotkey) -> Option<String> {
            // Advisory only: never break a settings dialog.
            match self.current_xkb_options() {
                Ok(options) => x11_layout_switch_conflict(hotkey, &options),
                Err(err) => {
                    log::debug!("Keyboard layout check for {hotkey} failed: {err}");
                    None
                }
            }
        }
    }

    impl ThreadedNative for X11Backend {
        fn threaded(&self) -> &Threaded {
            &self.threaded
        }

        fn before_thread_start(&self) -> bool {
            let mut pipe = self.pipe.lock();
            if pipe.0 < 0 {
                let mut fds = [-1 as RawFd; 2];
                // SAFETY: fds has room for the two descriptors.
                if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
                    log::warn!(
                        "Cannot create the hotkey wake-up pipe: {}",
                        std::io::Error::last_os_error()
                    );
                    return false;
                }
                for fd in fds {
                    // SAFETY: fd is a descriptor just created.
                    let configured = unsafe {
                        let flags = libc::fcntl(fd, libc::F_GETFL);
                        flags >= 0 && libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) == 0
                    };
                    if !configured {
                        let err = std::io::Error::last_os_error();
                        // SAFETY: closing the descriptors created above.
                        unsafe {
                            libc::close(fds[0]);
                            libc::close(fds[1]);
                        }
                        log::warn!("Cannot configure the hotkey wake-up pipe: {err}");
                        return false;
                    }
                }
                *pipe = (fds[0], fds[1]);
            }
            true
        }

        fn thread_setup(self: &Arc<Self>) -> bool {
            let display = match (self.display_factory)(self.display_name.as_deref()) {
                Ok(display) => display,
                Err(err) => {
                    log::warn!("Cannot connect to the X server for hotkeys: {err}");
                    self.core
                        .set_note(Some(format!("Could not connect to the X server: {err}")));
                    return false;
                }
            };
            let mut st = self.state.lock();
            st.display = Some(display);
            st.numlock_mask = Self::find_numlock_mask(&mut st);
            self.follow_xkb_options(&mut st);
            self.check_super_alone(&mut st);
            true
        }

        fn thread_loop(self: &Arc<Self>) {
            let wake_r = self.pipe.lock().0;
            while !self.threaded.is_stopping() {
                self.threaded.run_pending_calls();
                // The connection may already have buffered events (e.g. read during a round
                // trip), which poll() would not report, so drain before waiting.
                let display_fd = loop {
                    let polled = {
                        let mut st = self.state.lock();
                        let Some(display) = st.display.as_mut() else {
                            return;
                        };
                        display.poll_event().map(|event| (event, display.fileno()))
                    };
                    match polled {
                        Ok((Some(event), _)) => self.handle_event(event),
                        Ok((None, fd)) => break fd,
                        Err(err) => {
                            log::warn!("Hotkey thread stopping: {err}");
                            return;
                        }
                    }
                };
                match wait_readable(&[display_fd, wake_r], SELECT_TIMEOUT) {
                    Ok(readable) => {
                        if readable[1] {
                            self.drain_wake_pipe(wake_r);
                        }
                    }
                    Err(err) => {
                        log::warn!("Hotkey thread stopping: {err}");
                        break;
                    }
                }
            }
        }

        fn thread_teardown(self: &Arc<Self>) {
            {
                let mut st = self.state.lock();
                let grabbed: Vec<u32> = st.grabs.keys().copied().collect();
                for binding_id in grabbed {
                    Self::ungrab(&mut st, binding_id);
                }
                if let Some(mut display) = st.display.take() {
                    if let Err(err) = display.sync() {
                        log::debug!("Ungrabbing hotkeys failed: {err}");
                    }
                    display.close();
                }
                st.rules_atom = 0;
                if st.super_taken {
                    // checked again when the thread starts again
                    st.super_taken = false;
                    self.core.set_note(self.base_note.clone());
                }
                st.grabs.clear();
                st.lookup.clear();
                st.held.clear();
                st.last_release.clear();
            }
            *self.xkb_options.lock() = None;
            let mut pipe = self.pipe.lock();
            for fd in [pipe.0, pipe.1] {
                if fd >= 0 {
                    // SAFETY: closing a descriptor this manager owns.
                    unsafe { libc::close(fd) };
                }
            }
            *pipe = (-1, -1);
        }

        fn wake(&self) -> bool {
            let pipe = self.pipe.lock();
            if pipe.1 < 0 {
                return false;
            }
            // SAFETY: writing one byte from a valid buffer.
            if unsafe { libc::write(pipe.1, [0u8].as_ptr().cast(), 1) } == 1 {
                return true;
            }
            // Pipe full: the thread is due to wake up anyway.
            std::io::Error::last_os_error().kind() == std::io::ErrorKind::WouldBlock
        }
    }

    #[cfg(target_os = "linux")]
    mod xlib {
        //! The x11rb connection behind [`XDisplay`].

        use std::os::fd::{AsRawFd, RawFd};

        use x11rb::connection::Connection;
        use x11rb::protocol::Event;
        use x11rb::protocol::xproto::{
            AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, EventMask, GrabMode, ModMask,
        };
        use x11rb::rust_connection::RustConnection;

        use super::{XDisplay, XErrorCatcher, XEvent};

        pub(super) struct X11rbDisplay {
            conn: RustConnection,
            root: u32,
            min_keycode: u8,
            keysyms_per_keycode: usize,
            keysyms: Vec<u32>,
        }

        pub(super) fn open(name: Option<&str>) -> Result<Box<dyn XDisplay>, String> {
            let (conn, screen) = x11rb::connect(name).map_err(|e| e.to_string())?;
            let root = conn
                .setup()
                .roots
                .get(screen)
                .map(|s| s.root)
                .ok_or_else(|| format!("no screen {screen}"))?;
            let min_keycode = conn.setup().min_keycode;
            let mut display = X11rbDisplay {
                conn,
                root,
                min_keycode,
                keysyms_per_keycode: 0,
                keysyms: Vec::new(),
            };
            display.load_keymap()?;
            Ok(Box::new(display))
        }

        impl X11rbDisplay {
            fn load_keymap(&mut self) -> Result<(), String> {
                let setup = self.conn.setup();
                let (min, max) = (setup.min_keycode, setup.max_keycode);
                let reply = self
                    .conn
                    .get_keyboard_mapping(min, max - min + 1)
                    .map_err(|e| e.to_string())?
                    .reply()
                    .map_err(|e| e.to_string())?;
                self.min_keycode = min;
                self.keysyms_per_keycode = usize::from(reply.keysyms_per_keycode);
                self.keysyms = reply.keysyms;
                Ok(())
            }
        }

        impl XDisplay for X11rbDisplay {
            fn keysym_to_keycode(&mut self, keysym: u32) -> u8 {
                let per = self.keysyms_per_keycode;
                if per == 0 {
                    return 0;
                }
                for column in 0..per {
                    for (index, syms) in self.keysyms.chunks(per).enumerate() {
                        if syms.get(column) == Some(&keysym) {
                            return self.min_keycode.saturating_add(index as u8);
                        }
                    }
                }
                0
            }

            fn get_modifier_mapping(&mut self) -> Result<Vec<Vec<u8>>, String> {
                let reply = self
                    .conn
                    .get_modifier_mapping()
                    .map_err(|e| e.to_string())?
                    .reply()
                    .map_err(|e| e.to_string())?;
                let per = usize::from(reply.keycodes_per_modifier());
                if per == 0 {
                    return Ok(vec![Vec::new(); 8]);
                }
                Ok(reply.keycodes.chunks(per).map(<[u8]>::to_vec).collect())
            }

            fn grab_key(
                &mut self,
                keycode: u8,
                modifiers: u16,
                owner_events: bool,
                pointer_mode: u8,
                keyboard_mode: u8,
                onerror: &mut XErrorCatcher,
            ) {
                let checked = self
                    .conn
                    .grab_key(
                        owner_events,
                        self.root,
                        ModMask::from(modifiers),
                        keycode,
                        GrabMode::from(pointer_mode),
                        GrabMode::from(keyboard_mode),
                    )
                    .map_err(|e| e.to_string())
                    .and_then(|cookie| cookie.check().map_err(|e| format!("{e:?}")));
                if let Err(err) = checked {
                    onerror.catch(err);
                }
            }

            fn ungrab_key(&mut self, keycode: u8, modifiers: u16, onerror: &mut XErrorCatcher) {
                let checked = self
                    .conn
                    .ungrab_key(keycode, self.root, ModMask::from(modifiers))
                    .map_err(|e| e.to_string())
                    .and_then(|cookie| cookie.check().map_err(|e| format!("{e:?}")));
                if let Err(err) = checked {
                    onerror.catch(err);
                }
            }

            fn sync(&mut self) -> Result<(), String> {
                self.conn
                    .get_input_focus()
                    .map_err(|e| e.to_string())?
                    .reply()
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            }

            fn flush(&mut self) {
                let _ = self.conn.flush();
            }

            fn refresh_keyboard_mapping(&mut self, _event: &XEvent) {
                if let Err(err) = self.load_keymap() {
                    log::debug!("Reading the keyboard mapping failed: {err}");
                }
            }

            fn intern_atom(&mut self, name: &str, only_if_exists: bool) -> Result<u32, String> {
                Ok(self
                    .conn
                    .intern_atom(only_if_exists, name.as_bytes())
                    .map_err(|e| e.to_string())?
                    .reply()
                    .map_err(|e| e.to_string())?
                    .atom)
            }

            fn get_root_property(&mut self, atom: u32) -> Result<Option<Vec<u8>>, String> {
                let reply = self
                    .conn
                    .get_property(false, self.root, atom, AtomEnum::ANY, 0, u32::MAX / 4)
                    .map_err(|e| e.to_string())?
                    .reply()
                    .map_err(|e| e.to_string())?;
                Ok((reply.type_ != u32::from(AtomEnum::NONE)).then_some(reply.value))
            }

            fn select_root_events(&mut self, event_mask: u32) -> Result<(), String> {
                let aux = ChangeWindowAttributesAux::new().event_mask(EventMask::from(event_mask));
                self.conn
                    .change_window_attributes(self.root, &aux)
                    .map_err(|e| e.to_string())?
                    .check()
                    .map_err(|e| format!("{e:?}"))
            }

            fn poll_event(&mut self) -> Result<Option<XEvent>, String> {
                let event = self.conn.poll_for_event().map_err(|e| e.to_string())?;
                Ok(event.map(|event| match event {
                    Event::KeyPress(e) => XEvent::KeyPress {
                        detail: e.detail,
                        state: u16::from(e.state),
                        time: e.time,
                    },
                    Event::KeyRelease(e) => XEvent::KeyRelease {
                        detail: e.detail,
                        time: e.time,
                    },
                    Event::MappingNotify(e) => XEvent::MappingNotify {
                        request: u8::from(e.request),
                    },
                    Event::PropertyNotify(e) => XEvent::PropertyNotify { atom: e.atom },
                    Event::Error(e) => {
                        log::debug!("X error in hotkey connection: {e:?}");
                        XEvent::Other
                    }
                    _ => XEvent::Other,
                }))
            }

            fn fileno(&self) -> RawFd {
                self.conn.stream().as_raw_fd()
            }

            fn close(&mut self) {
                // The connection closes when this display is dropped.
            }
        }
    }
}

// --------------------------------------------------------------------------- factory

fn is_wayland_session(env: &dyn Fn(&str) -> Option<String>) -> bool {
    env("XDG_SESSION_TYPE").is_some_and(|t| t.to_lowercase() == "wayland")
        || env("WAYLAND_DISPLAY").is_some_and(|d| !d.is_empty())
}

/// The `ctl` commands to bind where global hotkeys are unavailable.
const CTL_SHORTCUT_ACTIONS: [&str; 3] = ["toggle", "privacy-toggle", "calibrate"];

/// `'… ctl toggle', '… ctl privacy-toggle' or '… ctl calibrate'`, spelled as this copy is run
/// (`command_text` is [`crate::cli::cli_command_text`]): the release packages put no
/// `eye-tracker` command on the PATH (an AppImage is called by its file name, the macOS app has
/// `eye-tracker-cli` inside it), and a desktop shortcut bound to a command that does not exist
/// fails without a word. Built when a manager is created, because a `--config-dir` profile
/// becomes part of the command.
fn ctl_commands(command_text: &dyn Fn(&[&str]) -> String) -> String {
    let commands: Vec<String> = CTL_SHORTCUT_ACTIONS
        .iter()
        .map(|action| format!("'{}'", command_text(&["ctl", action])))
        .collect();
    format!(
        "{} or {}",
        commands[..commands.len() - 1].join(", "),
        commands[commands.len() - 1]
    )
}

fn wayland_note(commands: &str) -> String {
    format!(
        "Wayland does not allow applications to register global hotkeys. Bind {commands} to \
         shortcuts in your desktop's keyboard settings."
    )
}

#[cfg_attr(not(unix), allow(dead_code))]
fn xwayland_note(commands: &str) -> String {
    format!(
        "Running under Wayland: hotkeys are grabbed through XWayland and may only fire while an \
         X11 window has focus. For reliable shortcuts bind {commands} in your desktop's \
         keyboard settings."
    )
}

/// Return the best hotkey manager for this system (never fails).
pub fn create_hotkey_manager() -> Box<dyn HotkeyManager> {
    create_for(
        std::env::consts::OS,
        &|name| std::env::var(name).ok(),
        &crate::cli::cli_command_text,
    )
}

/// [`create_hotkey_manager`] for `os` (a `std::env::consts::OS` value) and environment `env`.
fn create_for(
    os: &str,
    env: &dyn Fn(&str) -> Option<String>,
    command_text: &dyn Fn(&[&str]) -> String,
) -> Box<dyn HotkeyManager> {
    #[cfg(windows)]
    if os == "windows" {
        return Box::new(WindowsHotkeyManager::new());
    }
    if os == "macos" {
        return Box::new(MacHotkeyManager::new());
    }
    if env("DISPLAY").is_none_or(|d| d.is_empty()) {
        if is_wayland_session(env) {
            return Box::new(UnsupportedHotkeyManager::new(Some(wayland_note(
                &ctl_commands(command_text),
            ))));
        }
        return Box::new(UnsupportedHotkeyManager::new(Some(
            "No X11 display is available for global hotkeys.".to_string(),
        )));
    }
    #[cfg(unix)]
    {
        let note = is_wayland_session(env).then(|| xwayland_note(&ctl_commands(command_text)));
        Box::new(X11HotkeyManager::new(None, note))
    }
    #[cfg(not(unix))]
    {
        Box::new(UnsupportedHotkeyManager::new(Some(
            "Global hotkeys are not supported on this system.".to_string(),
        )))
    }
}

#[cfg(test)]
#[path = "hotkeys_tests.rs"]
mod tests;
