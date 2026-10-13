// Ported from tests/test_panes_chromium_apps.py (the desktop-app pane provider, its place behind
// the deny-list and the tree walk of the UI Automation binding).
//
// Everything runs against a fake control view shaped like the measured tree of the Claude desktop
// app; no test asks a real window anything.
//
// Changed: the tree is an arena of nodes addressed by index (`UiaNode(i)`), since Rust nodes cannot
// point at their parents; the fake client is shared (a clone goes to the provider) and
// the tests change the tree through `FakeClient::update`.
//
// Not ported:
// - test_the_vtable_slots_match_uiautomationclient_h, test_control_view_releases_everything_it_handed_out,
//   test_control_view_releases_the_walker_when_creating_fails, test_control_view_refuses_other_platforms
//   and test_smoke_reads_and_frees_a_bstr: tests of the ctypes COM layer; the equivalents that
//   still mean something are in platform/win_uia_tests.rs (see the reasons there).
// - `worker._failures == {}` in test_a_tree_not_built_yet_is_no_panes_not_a_failure: the worker's
//   counters are private; `disabled_providers` is checked.
// - test_default_providers_offer_desktop_apps_only_when_on_and_on_windows: the real factory only has
//   the optional providers on Windows, so the full list is asserted there and the shorter one
//   elsewhere (the registry tests inject a factory to see them on every OS).

use std::collections::VecDeque;
use std::sync::Arc;

use parking_lot::Mutex;

use super::*;
use crate::config::{PaneSettings, Settings, describe_settings};
use crate::panes::registry::{PaneRegistry, default_providers, is_denied};
use crate::panes::worker::{PaneEvent, PaneWorker, PaneWorkerOptions};
use crate::platform::win_uia::{
    ComError, UIA_DOCUMENT_CONTROL_TYPE_ID as DOCUMENT, UIA_EDIT_CONTROL_TYPE_ID as EDIT,
    UIA_GROUP_CONTROL_TYPE_ID as GROUP, UIA_PANE_CONTROL_TYPE_ID as NATIVE,
};
use crate::testing::panes_fakes::{Call, RecordingProvider, two_panes};
use crate::types::WindowHandle;

const WINDOW_TYPE: i32 = 50032; // UIA_WindowControlTypeId
const TEXT: i32 = 50020; // UIA_TextControlTypeId
const BUTTON: i32 = 50000; // UIA_ButtonControlTypeId

const HWND: isize = 0x5E0A1C;
const PID: u32 = 31337;
const AREA: Rect = Rect::new(1920, 0, 1920, 1040);

const PRIMARY_RECT: Rect = Rect::new(2200, 36, 820, 1004);
const SECONDARY_RECT: Rect = Rect::new(3020, 36, 820, 1004);
const PRIMARY_CLASS: &str = "dframe-pane dframe-pane-primary min-w-0 relative flex flex-col";
const SECONDARY_CLASS: &str = "dframe-pane dframe-pane-extra min-w-0 relative flex flex-col";

const MAIN_RECT: Rect = Rect::new(2261, 96, 884, 940);
const SIDE_RECT: Rect = Rect::new(3145, 96, 691, 940);

fn primary_id() -> PaneId {
    PaneId::Runtime(vec![42, 7, 101])
}

fn secondary_id() -> PaneId {
    PaneId::Runtime(vec![42, 7, 102])
}

fn main_id() -> PaneId {
    PaneId::Runtime(vec![42, 9, 1])
}

fn side_id() -> PaneId {
    PaneId::Runtime(vec![42, 9, 2])
}

fn window() -> WindowRef {
    WindowRef {
        handle: WindowHandle::Hwnd(HWND),
        pid: Some(PID),
        rect: Some(AREA),
    }
}

fn identity(process: &str, app_id: &str) -> AppIdentity {
    AppIdentity {
        process: process.into(),
        app_id: app_id.into(),
    }
}

fn app() -> AppIdentity {
    identity("claude", "Chrome_WidgetWin_1")
}

fn gpt_app() -> AppIdentity {
    identity("chatgpt", "Chrome_WidgetWin_1")
}

// ------------------------------------------------------------------ fake tree
/// One element of the fake control view.
#[derive(Clone)]
struct Node {
    kind: i32,
    cls: String,
    aid: String,
    rect: Option<Rect>,
    rid: Option<Vec<i32>>,
    offscreen: bool,
    focusable: bool,
    pid: u32,
    children: Vec<usize>,
    parent: Option<usize>,
}

impl Node {
    fn new(kind: i32) -> Self {
        Self {
            kind,
            cls: String::new(),
            aid: String::new(),
            rect: None,
            rid: None,
            offscreen: false,
            focusable: false,
            pid: PID,
            children: Vec::new(),
            parent: None,
        }
    }

    fn cls(mut self, cls: &str) -> Self {
        self.cls = cls.to_string();
        self
    }

    fn aid(mut self, aid: &str) -> Self {
        self.aid = aid.to_string();
        self
    }

    fn rect(mut self, rect: Rect) -> Self {
        self.rect = Some(rect);
        self
    }

    fn rid(mut self, rid: &[i32]) -> Self {
        self.rid = Some(rid.to_vec());
        self
    }

    fn focusable(mut self) -> Self {
        self.focusable = true;
        self
    }
}

/// All the nodes of a tree; a node is its index.
#[derive(Clone, Default)]
struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn new_node(&mut self, node: Node) -> usize {
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    /// `parent.add(*children)`; returns `parent`.
    fn add(&mut self, parent: usize, children: &[usize]) -> usize {
        for &child in children {
            self.nodes[child].parent = Some(parent);
            self.nodes[parent].children.push(child);
        }
        parent
    }

    fn insert_first(&mut self, parent: usize, child: usize) {
        self.nodes[child].parent = Some(parent);
        self.nodes[parent].children.insert(0, child);
    }

    fn last_child(&self, node: usize) -> usize {
        *self.nodes[node].children.last().expect("a child")
    }

    /// `node` and its descendants, depth first.
    fn walk(&self, node: usize) -> Vec<usize> {
        let mut found = vec![node];
        for &child in &self.nodes[node].children {
            found.extend(self.walk(child));
        }
        found
    }

    fn descendants(&self, node: usize) -> Vec<usize> {
        self.walk(node)[1..].to_vec()
    }
}

#[derive(Clone, Copy)]
struct Ids {
    window: usize,
    document: usize,
    primary: usize,
    secondary: usize,
    primary_composer: usize,
    secondary_composer: usize,
    sidebar: usize,
}

#[derive(Clone)]
struct ClaudeTree {
    arena: Arena,
    ids: Ids,
}

impl ClaudeTree {
    fn node(&mut self, index: usize) -> &mut Node {
        &mut self.arena.nodes[index]
    }
}

/// A transcript: one group per message, each with text inside.
fn conversation(arena: &mut Arena, messages: usize) -> usize {
    let transcript = arena.new_node(Node::new(GROUP).cls("flex-1 overflow-y-auto"));
    for i in 0..messages {
        let message = arena.new_node(Node::new(GROUP).cls("font-claude-message"));
        let text = arena.new_node(Node::new(TEXT));
        let inner = arena.new_node(Node::new(GROUP).cls(&format!("m-{i}")));
        let inner_text = arena.new_node(Node::new(TEXT));
        arena.add(inner, &[inner_text]);
        arena.add(message, &[text, inner]);
        arena.add(transcript, &[message]);
    }
    transcript
}

fn session(
    arena: &mut Arena,
    cls: &str,
    rect: Rect,
    rid: &[i32],
    messages: usize,
) -> (usize, usize) {
    let composer = arena.new_node(
        Node::new(EDIT)
            .cls("tiptap ProseMirror break-words")
            .rect(Rect::new(rect.x + 53, 975, 688, 20))
            .focusable(),
    );
    let pane = arena.new_node(Node::new(GROUP).cls(cls).rect(rect).rid(rid));
    let sticky = arena.new_node(Node::new(GROUP).cls("sticky top-0"));
    let sticky_button = arena.new_node(Node::new(BUTTON));
    arena.add(sticky, &[sticky_button]);
    let transcript = conversation(arena, messages);
    let composer_area = arena.new_node(Node::new(GROUP).cls("composer-area"));
    let relative = arena.new_node(Node::new(GROUP).cls("relative"));
    arena.add(relative, &[composer]);
    let toolbar = arena.new_node(Node::new(GROUP).cls("toolbar"));
    let toolbar_button = arena.new_node(Node::new(BUTTON));
    arena.add(toolbar, &[toolbar_button]);
    arena.add(composer_area, &[relative, toolbar]);
    arena.add(pane, &[sticky, transcript, composer_area]);
    (pane, composer)
}

/// The keyword arguments of the Python `claude_tree`.
#[derive(Clone, Copy)]
struct ClaudeOptions {
    built: bool,
    secondary: bool,
    messages: usize,
    sidebar_chats: usize,
    /// A second, nearly empty `RootWebArea` before the one with the sessions, as newer builds of
    /// the app have.
    empty_document_first: bool,
}

impl Default for ClaudeOptions {
    fn default() -> Self {
        Self {
            built: true,
            secondary: true,
            messages: 3,
            sidebar_chats: 50,
            empty_document_first: false,
        }
    }
}

fn claude_tree() -> ClaudeTree {
    claude_tree_with(ClaudeOptions::default())
}

/// The Claude window as measured (native views down to the web content).
fn claude_tree_with(options: ClaudeOptions) -> ClaudeTree {
    let mut a = Arena::default();
    let (primary, primary_composer) = session(
        &mut a,
        PRIMARY_CLASS,
        PRIMARY_RECT,
        &[42, 7, 101],
        options.messages,
    );
    let (extra, extra_composer) = session(
        &mut a,
        SECONDARY_CLASS,
        SECONDARY_RECT,
        &[42, 7, 102],
        options.messages,
    );
    let sidebar = a.new_node(
        Node::new(GROUP)
            .cls("dframe-sidebar df-hub-rail")
            .rect(Rect::new(1920, 0, 280, 1040)),
    );
    for _ in 0..options.sidebar_chats {
        let item = a.new_node(Node::new(GROUP).cls("chat-item"));
        let text = a.new_node(Node::new(TEXT));
        a.add(item, &[text]);
        a.add(sidebar, &[item]);
    }
    let content = a.new_node(Node::new(GROUP).cls("dframe-content flex"));
    a.add(content, &[primary]);
    if options.secondary {
        a.add(content, &[extra]);
    }
    let document = a.new_node(
        Node::new(DOCUMENT)
            .aid("RootWebArea")
            .rect(AREA)
            .focusable(),
    );
    if options.built {
        let page =
            a.new_node(Node::new(GROUP).cls("bg-surface-1 text-primary font-sans min-h-screen"));
        let root = a.new_node(Node::new(GROUP).aid("root"));
        a.add(root, &[sidebar, content]);
        a.add(page, &[root]);
        a.add(document, &[page]);
    }
    let innermost = a.new_node(Node::new(NATIVE).cls("View"));
    a.add(innermost, &[document]);
    let mut views = innermost;
    for _ in 0..3 {
        let outer = a.new_node(Node::new(NATIVE).cls("View"));
        a.add(outer, &[views]);
        views = outer;
    }
    let client_view = a.new_node(Node::new(NATIVE).cls("ClientView"));
    if options.empty_document_first {
        let empty = a.new_node(
            Node::new(DOCUMENT)
                .aid("RootWebArea")
                .rect(AREA)
                .focusable(),
        );
        let empty_root = a.new_node(Node::new(GROUP).aid("root"));
        a.add(empty, &[empty_root]);
        let empty_view = a.new_node(Node::new(NATIVE).cls("View"));
        a.add(empty_view, &[empty]);
        a.add(client_view, &[empty_view]);
    }
    a.add(client_view, &[views]);
    let window = a.new_node(Node::new(WINDOW_TYPE).cls("Chrome_WidgetWin_1").rect(AREA));
    let root_view = a.new_node(Node::new(NATIVE).cls("RootView"));
    let non_client = a.new_node(Node::new(NATIVE).cls("NonClientView"));
    let frame = a.new_node(Node::new(NATIVE).cls("WinFrameView"));
    let title_bar = a.new_node(Node::new(NATIVE).cls("TitleBar"));
    for _ in 0..3 {
        let button = a.new_node(Node::new(BUTTON));
        a.add(title_bar, &[button]);
    }
    a.add(frame, &[title_bar, client_view]);
    a.add(non_client, &[frame]);
    a.add(root_view, &[non_client]);
    a.add(window, &[root_view]);
    ClaudeTree {
        arena: a,
        ids: Ids {
            window,
            document,
            primary,
            secondary: extra,
            primary_composer,
            secondary_composer: extra_composer,
            sidebar,
        },
    }
}

/// The keyword arguments of the Python `chatgpt_tree`.
struct ChatgptOptions {
    side: bool,
    overlay: bool,
    build_hash: &'static str,
    messages: usize,
}

impl Default for ChatgptOptions {
    fn default() -> Self {
        Self {
            side: true,
            overlay: true,
            build_hash: "bo1ta_2",
            messages: 3,
        }
    }
}

fn chatgpt_tree() -> ClaudeTree {
    chatgpt_tree_with(ChatgptOptions::default())
}

/// The ChatGPT window as measured: main conversation and side chat.
fn chatgpt_tree_with(options: ChatgptOptions) -> ClaudeTree {
    let mut a = Arena::default();
    let main_composer = a.new_node(
        Node::new(EDIT)
            .cls("ProseMirror")
            .rect(Rect::new(2347, 936, 712, 44))
            .focusable(),
    );
    let main = a.new_node(
        Node::new(GROUP)
            .cls("relative flex h-full flex-col min-h-0")
            .rect(MAIN_RECT)
            .rid(&[42, 9, 1]),
    );
    let layout =
        a.new_node(Node::new(GROUP).cls("group/thread-scroll-layout has-[[data-composer]]"));
    let transcript = conversation(&mut a, options.messages);
    let composer_group = a.new_node(Node::new(GROUP).cls("composer"));
    a.add(composer_group, &[main_composer]);
    a.add(layout, &[transcript, composer_group]);
    a.add(main, &[layout]);
    let side_composer = a.new_node(
        Node::new(EDIT)
            .cls("ProseMirror ProseMirror-focused")
            .rect(Rect::new(3174, 936, 634, 44))
            .focusable(),
    );
    let side_pane = a.new_node(
        Node::new(GROUP)
            .cls("relative z-[41] h-full min-h-0 min-w-0 shrink-0")
            .rect(SIDE_RECT)
            .rid(&[42, 9, 2]),
    );
    let side_inner = a.new_node(Node::new(GROUP).cls("min-h-0 flex-1"));
    let tab_panel = a.new_node(
        Node::new(NATIVE)
            .cls("min-h-0 min-w-0 flex-1 outline-none relative")
            .aid("app-shell-tab-panel-app-shell-tab:4")
            .rect(Rect::new(3146, 96, 690, 940)),
    );
    let side_transcript = conversation(&mut a, options.messages);
    a.add(tab_panel, &[side_transcript, side_composer]);
    a.add(side_inner, &[tab_panel]);
    a.add(side_pane, &[side_inner]);
    let container = a.new_node(
        Node::new(GROUP)
            .cls(&format!(
                "outline-none _MainContentSurface_{}",
                options.build_hash
            ))
            .aid("_r_bs_")
            .rect(Rect::new(2260, 44, 1576, 992)),
    );
    a.add(container, &[main]);
    if options.side {
        a.add(container, &[side_pane]);
    }
    if options.overlay {
        let toast = a.new_node(
            Node::new(GROUP)
                .cls("absolute toast")
                .rect(Rect::new(3000, 100, 180, 40))
                .rid(&[42, 9, 3]),
        );
        a.add(container, &[toast]);
    }
    let sidebar = a.new_node(
        Node::new(GROUP)
            .cls("app-shell-left-panel pointer-events-auto")
            .rect(Rect::new(1920, 44, 340, 992)),
    );
    for _ in 0..100 {
        let item = a.new_node(Node::new(GROUP).cls("chat-item"));
        let text = a.new_node(Node::new(TEXT));
        a.add(item, &[text]);
        a.add(sidebar, &[item]);
    }
    let document = a.new_node(
        Node::new(DOCUMENT)
            .aid("RootWebArea")
            .rect(AREA)
            .focusable(),
    );
    let outer = a.new_node(Node::new(GROUP));
    let root = a.new_node(Node::new(GROUP).aid("root"));
    a.add(root, &[sidebar, container]);
    a.add(outer, &[root]);
    a.add(document, &[outer]);
    let widget = a.new_node(Node::new(NATIVE).cls("Chrome_WidgetWin_1"));
    a.add(widget, &[document]);
    let view = a.new_node(Node::new(NATIVE).cls("View"));
    a.add(view, &[widget]);
    let client_view = a.new_node(Node::new(NATIVE).cls("ChromeNodeClientView"));
    a.add(client_view, &[view]);
    let frame = a.new_node(Node::new(NATIVE).cls("ChromeNodeFrameView"));
    a.add(frame, &[client_view]);
    let non_client = a.new_node(Node::new(NATIVE).cls("NonClientView"));
    a.add(non_client, &[frame]);
    let root_view = a.new_node(Node::new(NATIVE).cls("RootView"));
    a.add(root_view, &[non_client]);
    let window = a.new_node(Node::new(WINDOW_TYPE).cls("Chrome_WidgetWin_1").rect(AREA));
    a.add(window, &[root_view]);
    ClaudeTree {
        arena: a,
        ids: Ids {
            window,
            document,
            primary: main,
            secondary: side_pane,
            primary_composer: main_composer,
            secondary_composer: side_composer,
            sidebar,
        },
    }
}

// ---------------------------------------------------------- fake client/view
/// What a view recorded (counts steps, records property reads).
#[derive(Default)]
struct ViewRecord {
    steps: usize,
    reads: Vec<(&'static str, usize)>,
    closed: bool,
}

struct ClientState {
    tree: ClaudeTree,
    foreground: Option<isize>,
    focused: usize,
    views: Vec<Arc<Mutex<ViewRecord>>>,
    calls: Vec<String>,
    focus_set: Vec<usize>,
    fail: Option<UiaError>,
    /// Foreground windows answered first, one per call (then `foreground`).
    foreground_sequence: VecDeque<Option<isize>>,
    closed: u32,
}

/// Answers like `win_uia::UiAutomation`; records every view it opened. Clones share the state.
#[derive(Clone)]
struct FakeClient(Arc<Mutex<ClientState>>);

impl FakeClient {
    fn new(tree: ClaudeTree) -> Self {
        let focused = tree.ids.primary_composer;
        Self::with_focus(tree, focused)
    }

    fn with_focus(tree: ClaudeTree, focused: usize) -> Self {
        Self(Arc::new(Mutex::new(ClientState {
            tree,
            foreground: Some(HWND),
            focused,
            views: Vec::new(),
            calls: Vec::new(),
            focus_set: Vec::new(),
            fail: None,
            foreground_sequence: VecDeque::new(),
            closed: 0,
        })))
    }

    fn foreground(self, foreground: isize) -> Self {
        self.0.lock().foreground = Some(foreground);
        self
    }

    fn update<T>(&self, change: impl FnOnce(&mut ClaudeTree) -> T) -> T {
        change(&mut self.0.lock().tree)
    }

    fn set_tree(&self, tree: ClaudeTree) {
        let mut state = self.0.lock();
        state.focused = tree.ids.primary_composer;
        state.tree = tree;
    }

    fn set_focused(&self, focused: usize) {
        self.0.lock().focused = focused;
    }

    fn set_fail(&self, error: UiaError) {
        self.0.lock().fail = Some(error);
    }

    fn ids(&self) -> Ids {
        self.0.lock().tree.ids
    }

    fn calls(&self) -> Vec<String> {
        self.0.lock().calls.clone()
    }

    fn clear_calls(&self) {
        self.0.lock().calls.clear();
    }

    fn focus_set(&self) -> Vec<usize> {
        self.0.lock().focus_set.clone()
    }

    fn closed(&self) -> u32 {
        self.0.lock().closed
    }

    fn set_foreground_sequence(&self, sequence: &[Option<isize>]) {
        self.0.lock().foreground_sequence = sequence.iter().copied().collect();
    }

    fn views(&self) -> Vec<Arc<Mutex<ViewRecord>>> {
        self.0.lock().views.clone()
    }

    fn all_closed(&self) -> bool {
        self.views().iter().all(|v| v.lock().closed)
    }

    /// The reads of view `index`.
    fn reads(&self, index: usize) -> Vec<(&'static str, usize)> {
        self.views()[index].lock().reads.clone()
    }

    fn steps(&self, index: usize) -> usize {
        self.views()[index].lock().steps
    }
}

impl UiaTreeClient for FakeClient {
    fn control_view(&self, hwnd: isize) -> Result<Box<dyn ElementView>, UiaError> {
        let mut state = self.0.lock();
        state.calls.push("control_view".to_string());
        assert_eq!(hwnd, HWND);
        let record = Arc::new(Mutex::new(ViewRecord::default()));
        state.views.push(record.clone());
        Ok(Box::new(FakeView {
            client: self.clone(),
            root: state.tree.ids.window,
            record,
        }))
    }

    fn foreground_window(&self) -> Result<Option<isize>, UiaError> {
        let mut state = self.0.lock();
        state.calls.push("foreground".to_string());
        Ok(match state.foreground_sequence.pop_front() {
            Some(answer) => answer,
            None => state.foreground,
        })
    }

    fn close(&self) {
        self.0.lock().closed += 1;
    }
}

/// A control view over a fake tree.
struct FakeView {
    client: FakeClient,
    root: usize,
    record: Arc<Mutex<ViewRecord>>,
}

impl FakeView {
    fn step(&self, node: Option<usize>) -> Result<Option<UiaNode>, UiaError> {
        let mut record = self.record.lock();
        assert!(!record.closed);
        record.steps += 1;
        if let Some(error) = &self.client.0.lock().fail {
            return Err(error.clone());
        }
        Ok(node.map(UiaNode))
    }

    fn read<T>(
        &self,
        what: &'static str,
        node: UiaNode,
        value: impl FnOnce(&Node) -> T,
    ) -> Result<T, UiaError> {
        let mut record = self.record.lock();
        assert!(!record.closed);
        record.reads.push((what, node.0));
        Ok(value(&self.client.0.lock().tree.arena.nodes[node.0]))
    }

    fn sibling(&self, node: UiaNode, offset: isize) -> Result<Option<UiaNode>, UiaError> {
        let found = {
            let state = self.client.0.lock();
            let nodes = &state.tree.arena.nodes;
            match nodes[node.0].parent {
                None => None, // `[node]`: no neighbours
                Some(parent) => {
                    let siblings = &nodes[parent].children;
                    let position = siblings.iter().position(|&s| s == node.0).expect("a child");
                    position
                        .checked_add_signed(offset)
                        .and_then(|i| siblings.get(i).copied())
                }
            }
        };
        self.step(found)
    }
}

impl ElementView for FakeView {
    fn root(&self) -> Option<UiaNode> {
        Some(UiaNode(self.root))
    }

    fn first_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
        let child = self.client.0.lock().tree.arena.nodes[node.0]
            .children
            .first()
            .copied();
        self.step(child)
    }

    fn last_child(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
        let child = self.client.0.lock().tree.arena.nodes[node.0]
            .children
            .last()
            .copied();
        self.step(child)
    }

    fn next_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
        self.sibling(node, 1)
    }

    fn previous_sibling(&mut self, node: UiaNode) -> Result<Option<UiaNode>, UiaError> {
        self.sibling(node, -1)
    }

    fn focused(&mut self) -> Result<Option<UiaNode>, UiaError> {
        assert!(!self.record.lock().closed);
        Ok(Some(UiaNode(self.client.0.lock().focused)))
    }

    fn control_type(&self, node: UiaNode) -> Result<i32, UiaError> {
        self.read("control_type", node, |n| n.kind)
    }

    fn process_id(&self, node: UiaNode) -> Result<u32, UiaError> {
        self.read("process_id", node, |n| n.pid)
    }

    fn class_name(&self, node: UiaNode) -> Result<String, UiaError> {
        self.read("class_name", node, |n| n.cls.clone())
    }

    fn automation_id(&self, node: UiaNode) -> Result<String, UiaError> {
        self.read("automation_id", node, |n| n.aid.clone())
    }

    fn is_keyboard_focusable(&self, node: UiaNode) -> Result<bool, UiaError> {
        self.read("is_keyboard_focusable", node, |n| n.focusable)
    }

    fn is_offscreen(&self, node: UiaNode) -> Result<bool, UiaError> {
        self.read("is_offscreen", node, |n| n.offscreen)
    }

    fn rect(&self, node: UiaNode) -> Result<Option<Rect>, UiaError> {
        self.read("rect", node, |n| n.rect)
    }

    fn runtime_id(&self, node: UiaNode) -> Result<Option<Vec<i32>>, UiaError> {
        self.read("runtime_id", node, |n| n.rid.clone())
    }

    fn set_focus(&self, node: UiaNode) -> Result<(), UiaError> {
        assert!(!self.record.lock().closed);
        let mut state = self.client.0.lock();
        state.focus_set.push(node.0);
        state.focused = node.0;
        Ok(())
    }

    fn close(&mut self) {
        self.record.lock().closed = true;
    }
}

fn provider(client: &FakeClient) -> ChromiumAppProvider {
    ChromiumAppProvider::new(Arc::new(|window: &WindowRef| {
        (window.handle == WindowHandle::Hwnd(HWND)).then_some(AREA)
    }))
    .with_client(Box::new(client.clone()))
}

fn detect(p: &mut ChromiumAppProvider, app: &AppIdentity) -> Option<PaneSnapshot> {
    p.detect(&window(), app).expect("no UI Automation error")
}

fn focused_id(snapshot: &PaneSnapshot) -> Option<PaneId> {
    snapshot.focused().map(|p| p.id.clone())
}

fn ids(snapshot: &PaneSnapshot) -> Vec<PaneId> {
    snapshot.panes.iter().map(|p| p.id.clone()).collect()
}

fn rects(snapshot: &PaneSnapshot) -> Vec<Rect> {
    snapshot.panes.iter().map(|p| p.rect).collect()
}

fn pane_of(id: PaneId, rect: Rect) -> Pane {
    Pane::new(id, rect, false, PROVIDER_NAME)
}

// ---------------------------------------------------------------- profiles
#[test]
fn test_only_the_claude_and_chatgpt_apps_have_a_profile() {
    assert_eq!(PROFILES, [&CLAUDE, &CHATGPT]);
    assert_eq!(profile_for(Some(&app())), Some(&CLAUDE));
    assert_eq!(
        profile_for(Some(&identity("Claude", "Chrome_WidgetWin_1"))),
        Some(&CLAUDE)
    );
    assert_eq!(profile_for(Some(&gpt_app())), Some(&CHATGPT));
    assert_eq!(
        profile_for(Some(&identity("ChatGPT", "Chrome_WidgetWin_1"))),
        Some(&CHATGPT)
    );
    // Process name and window class must both match.
    assert_eq!(
        profile_for(Some(&identity("claude", "Chrome_WidgetWin_0"))),
        None
    );
    assert_eq!(
        profile_for(Some(&identity("claude", "com.anthropic.claudefordesktop"))),
        None
    );
    assert_eq!(
        profile_for(Some(&identity("chatgpt", "com.openai.chat"))),
        None
    );
    assert_eq!(
        profile_for(Some(&identity("code", "Chrome_WidgetWin_1"))),
        None
    );
    assert_eq!(profile_for(None), None);
}

#[test]
fn test_a_profile_finds_panes_one_way() {
    let profile = |pane_token, container_prefix| AppProfile {
        name: "x",
        processes: &["x"],
        window_class: "c",
        composer_token: "ProseMirror",
        pane_token,
        container_prefix,
        skip_tokens: &[],
        min_pane: DEFAULT_MIN_PANE,
    };
    for invalid in [profile(None, None), profile(Some("a"), Some("b"))] {
        let error = invalid.validated().expect_err("an invalid profile");
        assert!(error.to_string().contains("exactly one"), "{error}");
    }
    assert!(profile(Some("a"), None).validated().is_ok());
    assert!(profile(None, Some("b")).validated().is_ok());
    assert!(CLAUDE.validated().is_ok() && CHATGPT.validated().is_ok());
}

#[test]
fn test_the_container_is_matched_by_token_prefix() {
    let is_container =
        |profile: &AppProfile, class: &str| profile.is_container(&class_tokens(class));
    assert!(is_container(
        &CHATGPT,
        "outline-none _MainContentSurface_bo1ta_2"
    ));
    assert!(is_container(&CHATGPT, "_MainContentSurface_zz9_7")); // another build
    assert!(!is_container(&CHATGPT, "x_MainContentSurface_bo1ta_2"));
    assert!(!is_container(&CHATGPT, "MainContentSurface outline-none"));
    assert!(!is_container(&CLAUDE, "_MainContentSurface_bo1ta_2"));
}

#[test]
fn test_creating_the_provider_touches_nothing() {
    let mut p = ChromiumAppProvider::new(Arc::new(|_| None));
    assert_eq!(p.name(), PROVIDER_NAME);
    assert_eq!(PROVIDER_NAME, "desktop_apps");
    assert!(p.applies(&app()));
    assert!(!p.applies(&identity("cursor", "Chrome_WidgetWin_1")));
    assert!(p.client.is_none());
}

#[test]
fn test_class_tokens_are_whole_words() {
    let tokens = class_tokens(PRIMARY_CLASS);
    assert!(tokens.contains("dframe-pane") && tokens.contains("dframe-pane-primary"));
    assert!(!class_tokens("dframe-pane-primary min-w-0").contains("dframe-pane"));
    assert!(!class_tokens("xdframe-pane").contains("dframe-pane"));
    assert_eq!(class_tokens("  a\tb\n c "), HashSet::from(["a", "b", "c"]));
    assert_eq!(class_tokens(""), HashSet::new());
}

// ------------------------------------------------------------------- detect
#[test]
fn test_finds_the_sessions_behind_an_empty_first_document() {
    let client = FakeClient::new(claude_tree_with(ClaudeOptions {
        empty_document_first: true,
        ..ClaudeOptions::default()
    }));
    let snapshot = detect(&mut provider(&client), &app()).expect("a snapshot");
    assert_eq!(rects(&snapshot), vec![PRIMARY_RECT, SECONDARY_RECT]);
    assert!(client.all_closed());
}

#[test]
fn test_focuses_a_session_behind_an_empty_first_document() {
    let client = FakeClient::new(claude_tree_with(ClaudeOptions {
        empty_document_first: true,
        ..ClaudeOptions::default()
    }));
    let mut p = provider(&client);
    let snapshot = detect(&mut p, &app()).expect("a snapshot");
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(client.focus_set(), vec![client.ids().secondary_composer]);
}

#[test]
fn test_detects_the_two_sessions_and_the_focused_one() {
    let client = FakeClient::new(claude_tree());
    let snapshot = detect(&mut provider(&client), &app()).expect("a snapshot");
    assert_eq!(snapshot.window_handle, WindowHandle::Hwnd(HWND));
    assert_eq!(rects(&snapshot), vec![PRIMARY_RECT, SECONDARY_RECT]);
    assert_eq!(ids(&snapshot), vec![primary_id(), secondary_id()]);
    assert!(snapshot.panes.iter().all(|p| p.provider == "desktop_apps"));
    assert_eq!(focused_id(&snapshot), Some(primary_id()));
    assert_eq!(client.calls(), vec!["foreground", "control_view"]);
    assert!(client.all_closed());

    client.set_focused(client.ids().secondary_composer);
    let snapshot = detect(&mut provider(&client), &app()).expect("a snapshot");
    assert_eq!(focused_id(&snapshot), Some(secondary_id()));
}

#[test]
fn test_the_walk_is_short_and_reads_no_more_than_it_needs() {
    let client = FakeClient::new(claude_tree_with(ClaudeOptions {
        messages: 200,
        sidebar_chats: 300,
        ..ClaudeOptions::default()
    }));
    assert!(detect(&mut provider(&client), &app()).is_some());
    assert_eq!(client.views().len(), 1);
    // Native views, the document, three groups down to the panes: a few dozen steps.
    let steps = client.steps(0);
    assert!(steps <= 40, "{steps}");
    let ids = client.ids();
    let reads = client.reads(0);
    // Of the focused element only its process and rectangle are read.
    let focused_reads: HashSet<&str> = reads
        .iter()
        .filter(|(_, n)| *n == ids.primary_composer)
        .map(|(what, _)| *what)
        .collect();
    assert_eq!(focused_reads, HashSet::from(["process_id", "rect"]));
    let touched: HashSet<usize> = reads
        .iter()
        .filter(|(_, n)| *n != ids.primary_composer)
        .map(|(_, n)| *n)
        .collect();
    let arena = client.update(|t| t.arena.clone());
    let inside: Vec<usize> = [ids.primary, ids.secondary]
        .into_iter()
        .flat_map(|pane| arena.descendants(pane))
        .collect();
    assert!(!inside.iter().any(|n| touched.contains(n))); // pane contents are not walked
    let sidebar = arena.descendants(ids.sidebar);
    assert!(!sidebar.iter().any(|n| touched.contains(n))); // nor the sidebar
    let kinds: HashSet<&str> = reads.iter().map(|(what, _)| *what).collect();
    let allowed = HashSet::from([
        "control_type",
        "class_name",
        "automation_id",
        "is_offscreen",
        "rect",
        "runtime_id",
        "process_id",
    ]);
    assert!(kinds.is_subset(&allowed), "{kinds:?}");
    // Class names only of groups, automation ids only of documents.
    let kind_of = |n: usize| arena.nodes[n].kind;
    assert!(
        reads
            .iter()
            .filter(|(what, _)| *what == "class_name")
            .all(|(_, n)| kind_of(*n) == GROUP)
    );
    assert!(
        reads
            .iter()
            .filter(|(what, _)| *what == "automation_id")
            .all(|(_, n)| kind_of(*n) == DOCUMENT)
    );
}

#[test]
fn test_one_session_is_not_a_split() {
    let client = FakeClient::new(claude_tree_with(ClaudeOptions {
        secondary: false,
        ..ClaudeOptions::default()
    }));
    assert_eq!(detect(&mut provider(&client), &app()), None);
}

#[test]
fn test_hidden_and_empty_sessions_do_not_count() {
    let detects_nothing = |change: &dyn Fn(&mut Node)| {
        let mut tree = claude_tree();
        let secondary = tree.ids.secondary;
        change(tree.node(secondary));
        detect(&mut provider(&FakeClient::new(tree)), &app()).is_none()
    };
    assert!(detects_nothing(&|n| n.offscreen = true));
    assert!(detects_nothing(&|n| n.rect = None));
    // Outside the window.
    assert!(detects_nothing(
        &|n| n.rect = Some(Rect::new(5000, 36, 820, 1004))
    ));
}

#[test]
fn test_only_whole_class_tokens_make_a_pane() {
    let mut tree = claude_tree();
    let secondary = tree.ids.secondary;
    tree.node(secondary).cls = "dframe-pane-extra min-w-0".into(); // no "dframe-pane" token
    assert_eq!(
        detect(&mut provider(&FakeClient::new(tree.clone())), &app()),
        None
    );
    tree.node(secondary).cls = "xdframe-pane".into();
    assert_eq!(
        detect(&mut provider(&FakeClient::new(tree.clone())), &app()),
        None
    );
    tree.node(secondary).cls = "min-w-0 dframe-pane".into(); // the token anywhere in the list
    assert!(detect(&mut provider(&FakeClient::new(tree)), &app()).is_some());
}

#[test]
fn test_a_pane_that_is_not_a_group_does_not_count() {
    let mut tree = claude_tree();
    let secondary = tree.ids.secondary;
    tree.node(secondary).kind = NATIVE;
    assert_eq!(detect(&mut provider(&FakeClient::new(tree)), &app()), None);
}

#[test]
fn test_no_pane_is_focused_unless_the_focus_is_inside_one() {
    let tree = claude_tree();
    let ids = tree.ids;
    let focused_with = |tree: ClaudeTree, focused: usize| {
        let client = FakeClient::with_focus(tree, focused);
        let snapshot = detect(&mut provider(&client), &app()).expect("a snapshot");
        focused_id(&snapshot)
    };
    // The whole page has the focus: larger than a pane, so no pane has it.
    assert_eq!(focused_with(tree.clone(), ids.document), None);
    // The sidebar has it.
    assert_eq!(focused_with(tree.clone(), ids.sidebar), None);
    // The pane itself has it.
    assert_eq!(
        focused_with(tree.clone(), ids.secondary),
        Some(secondary_id())
    );
    // An element of another process at the same place does not count.
    let mut tree = tree;
    let mut stranger = Node::new(EDIT).rect(Rect::new(3073, 975, 688, 20));
    stranger.pid = PID + 1;
    let stranger = tree.arena.new_node(stranger);
    assert_eq!(focused_with(tree, stranger), None);
}

#[test]
fn test_without_a_known_pid_the_windows_process_is_compared() {
    let mut tree = claude_tree();
    let reference = WindowRef {
        handle: WindowHandle::Hwnd(HWND),
        pid: None,
        rect: Some(AREA),
    };
    let snapshot = provider(&FakeClient::new(tree.clone()))
        .detect(&reference, &app())
        .unwrap()
        .expect("a snapshot");
    assert_eq!(focused_id(&snapshot), Some(primary_id()));
    let window = tree.ids.window;
    tree.node(window).pid = PID + 1;
    let snapshot = provider(&FakeClient::new(tree))
        .detect(&reference, &app())
        .unwrap()
        .expect("a snapshot");
    assert_eq!(focused_id(&snapshot), None);
}

#[test]
fn test_without_runtime_ids_the_rectangles_are_the_ids() {
    let mut tree = claude_tree();
    let (primary, secondary) = (tree.ids.primary, tree.ids.secondary);
    tree.node(primary).rid = None;
    tree.node(secondary).rid = Some(vec![]);
    let snapshot = detect(&mut provider(&FakeClient::new(tree)), &app()).expect("a snapshot");
    assert_eq!(
        ids(&snapshot),
        vec![
            PaneId::Rect(Rect::new(2200, 36, 820, 1004)),
            PaneId::Rect(Rect::new(3020, 36, 820, 1004)),
        ]
    );
}

fn registry_worker(client: &FakeClient) -> PaneWorker {
    PaneWorker::new(
        PaneRegistry::new(vec![Box::new(provider(client))]).with_desktop_apps(true),
        PaneWorkerOptions {
            synchronous: true,
            ..PaneWorkerOptions::default()
        },
    )
}

fn detected(worker: &PaneWorker) -> Vec<Option<PaneSnapshot>> {
    worker
        .drain_events()
        .into_iter()
        .filter_map(|event| match event {
            PaneEvent::Detected(result) => Some(result.snapshot),
            PaneEvent::Focused(_) => None,
        })
        .collect()
}

#[test]
fn test_a_tree_not_built_yet_is_no_panes_not_a_failure() {
    // Chromium builds its accessibility tree once asked: first only native views,
    // then web content without panes, then everything.
    let mut tree = claude_tree_with(ClaudeOptions {
        built: false,
        ..ClaudeOptions::default()
    });
    let document = tree.ids.document;
    tree.node(document).kind = NATIVE; // not even the document yet
    let client = FakeClient::new(tree);
    let worker = registry_worker(&client);
    let mut results = Vec::new();
    for _ in 0..5 {
        worker.request_detect(&window(), &app());
    }
    results.extend(detected(&worker));
    client.set_tree(claude_tree_with(ClaudeOptions {
        built: false,
        ..ClaudeOptions::default()
    }));
    for _ in 0..5 {
        worker.request_detect(&window(), &app());
    }
    results.extend(detected(&worker));
    assert_eq!(results.len(), 10);
    assert!(results.iter().all(Option::is_none));
    assert!(worker.disabled_providers().is_empty());
    client.set_tree(claude_tree());
    worker.request_detect(&window(), &app());
    let results = detected(&worker);
    let snapshot = results
        .last()
        .expect("a result")
        .as_ref()
        .expect("a snapshot");
    assert_eq!(snapshot.provider(), Some("desktop_apps"));
    assert!(client.all_closed());
}

#[test]
fn test_another_document_is_never_entered() {
    let mut tree = claude_tree();
    let other = tree.arena.new_node(Node::new(DOCUMENT).aid("DevTools"));
    let x = tree.arena.new_node(Node::new(GROUP).cls("x"));
    let dev_pane = tree.arena.new_node(
        Node::new(GROUP)
            .cls("dframe-pane")
            .rect(PRIMARY_RECT)
            .rid(&[9]),
    );
    tree.arena.add(x, &[dev_pane]);
    tree.arena.add(other, &[x]);
    let root_view = tree.arena.nodes[tree.ids.window].children[0];
    tree.arena.insert_first(root_view, other);
    let client = FakeClient::new(tree.clone());
    let snapshot = detect(&mut provider(&client), &app()).expect("a snapshot");
    assert_eq!(ids(&snapshot), vec![primary_id(), secondary_id()]);
    let touched: HashSet<usize> = client.reads(0).iter().map(|(_, n)| *n).collect();
    assert!(
        !tree
            .arena
            .descendants(other)
            .iter()
            .any(|n| touched.contains(n))
    );
}

#[test]
fn test_the_walk_stops_at_its_budget() {
    let mut tree = claude_tree();
    // Thousands of groups before the content, without a pane among them.
    let wide = tree.arena.new_node(Node::new(GROUP).cls("virtual-list"));
    for _ in 0..5000 {
        let row = tree.arena.new_node(Node::new(GROUP).cls("row"));
        let cell = tree.arena.new_node(Node::new(GROUP).cls("cell"));
        tree.arena.add(row, &[cell]);
        tree.arena.add(wide, &[row]);
    }
    let page = tree.arena.nodes[tree.ids.document].children[0];
    let root_group = tree.arena.nodes[page].children[0];
    tree.arena.insert_first(root_group, wide);
    let client = FakeClient::new(tree);
    assert_eq!(detect(&mut provider(&client), &app()), None);
    assert!(client.steps(0) <= NODE_BUDGET);
    assert!(client.all_closed());

    // A deep chain of groups: at most WEB_DEPTH levels below the document.
    let mut tree = claude_tree();
    let mut deep = tree.ids.document;
    for _ in 0..50 {
        let child = tree.arena.new_node(Node::new(GROUP).cls("nest"));
        tree.arena.insert_first(deep, child);
        deep = child;
    }
    let leaf = tree.arena.new_node(
        Node::new(GROUP)
            .cls("dframe-pane")
            .rect(PRIMARY_RECT)
            .rid(&[1]),
    );
    tree.arena.add(deep, &[leaf]);
    let client = FakeClient::new(tree);
    let _ = provider(&client).detect(&window(), &app());
    let reached: HashSet<usize> = client.reads(0).iter().map(|(_, n)| *n).collect();
    assert!(!reached.contains(&deep));
}

#[test]
fn test_a_huge_native_tree_stops_at_the_budget_too() {
    let mut tree = claude_tree();
    let window_node = tree.arena.new_node(Node::new(WINDOW_TYPE));
    for _ in 0..2000 {
        let outer = tree.arena.new_node(Node::new(NATIVE).cls("View"));
        let inner = tree.arena.new_node(Node::new(NATIVE).cls("View"));
        tree.arena.add(outer, &[inner]);
        tree.arena.add(window_node, &[outer]);
    }
    tree.ids.window = window_node;
    let client = FakeClient::new(tree);
    assert_eq!(detect(&mut provider(&client), &app()), None);
    assert!(client.steps(0) <= NODE_BUDGET);
}

#[test]
fn test_nothing_is_asked_unless_the_window_is_in_the_foreground() {
    let client = FakeClient::new(claude_tree()).foreground(0x1234);
    let mut p = provider(&client);
    assert_eq!(detect(&mut p, &app()), None);
    assert_eq!(
        p.focus(&window(), &pane_of(secondary_id(), SECONDARY_RECT)),
        Ok(false)
    );
    assert_eq!(client.calls(), vec!["foreground"]); // focus() has not seen this window yet
    let no_handle = WindowRef {
        handle: WindowHandle::X11(0),
        pid: None,
        rect: None,
    };
    assert_eq!(p.detect(&no_handle, &app()), Ok(None));
    assert_eq!(
        detect(&mut p, &identity("code", "Chrome_WidgetWin_1")),
        None
    );
    assert!(client.views().is_empty());
}

#[test]
fn test_com_errors_become_pane_errors_and_release_the_view() {
    let client = FakeClient::new(claude_tree());
    let mut p = provider(&client);
    assert!(detect(&mut p, &app()).is_some());
    client.set_fail(
        ComError::new("IUIAutomationTreeWalker::GetFirstChildElement", 0x8004_0201).into(),
    );
    let error = p.detect(&window(), &app()).expect_err("a pane error");
    assert!(error.0.contains("0x80040201"), "{error}");
    assert!(
        p.focus(&window(), &pane_of(secondary_id(), SECONDARY_RECT))
            .is_err()
    );
    assert!(client.all_closed());
}

// -------------------------------------------------------------------- focus
fn detected_provider(client: &FakeClient) -> (ChromiumAppProvider, PaneSnapshot) {
    let mut p = provider(client);
    let snapshot = detect(&mut p, &app()).expect("a snapshot");
    (p, snapshot)
}

#[test]
fn test_focus_goes_to_the_sessions_message_box() {
    let client = FakeClient::new(claude_tree());
    let (mut p, snapshot) = detected_provider(&client);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(client.focus_set(), vec![client.ids().secondary_composer]);
    assert!(client.all_closed());
    // And the next detection sees that session focused.
    let again = detect(&mut p, &app()).expect("a snapshot");
    assert_eq!(focused_id(&again), Some(secondary_id()));
}

#[test]
fn test_focus_finds_the_message_box_after_a_long_conversation() {
    let client = FakeClient::new(claude_tree_with(ClaudeOptions {
        messages: 2000,
        ..ClaudeOptions::default()
    }));
    let (mut p, snapshot) = detected_provider(&client);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(client.focus_set(), vec![client.ids().secondary_composer]);
    let last = client.views().len() - 1;
    assert!(client.steps(last) <= NODE_BUDGET + COMPOSER_BUDGET);
}

#[test]
fn test_focus_prefers_the_prosemirror_editor() {
    let mut tree = claude_tree();
    let composer_area = tree.arena.last_child(tree.ids.secondary);
    let toolbar = tree.arena.last_child(composer_area);
    let search = tree.arena.new_node(
        Node::new(EDIT)
            .cls("search-input")
            .rect(Rect::new(3500, 990, 100, 20))
            .focusable(),
    );
    tree.arena.add(toolbar, &[search]); // walked before the message box (bottom up)
    let composer = tree.ids.secondary_composer;
    let client = FakeClient::new(tree);
    let (mut p, snapshot) = detected_provider(&client);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(client.focus_set(), vec![composer]);
}

#[test]
fn test_focus_falls_back_to_a_focusable_editor() {
    let client = FakeClient::new(claude_tree());
    let composer = client.ids().secondary_composer;
    client.update(|t| t.node(composer).cls = "some-textarea".into());
    let (mut p, snapshot) = detected_provider(&client);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(client.focus_set(), vec![composer]);
    // Not focusable either: nothing to focus.
    client.update(|t| t.node(composer).focusable = false);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(false));
    assert_eq!(client.focus_set(), vec![composer]); // unchanged
    assert!(client.all_closed());
}

#[test]
fn test_no_focus_when_another_window_came_to_the_front_during_the_walk() {
    let client = FakeClient::new(claude_tree());
    let (mut p, snapshot) = detected_provider(&client);
    client.clear_calls();
    // In front before the walk, not after it.
    client.set_foreground_sequence(&[Some(HWND), Some(0x5151)]);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(false));
    assert!(client.focus_set().is_empty());
    // Asked before opening the view and again right before SetFocus.
    assert_eq!(
        client.calls(),
        vec!["foreground", "control_view", "foreground"]
    );
    assert!(client.all_closed());
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
}

#[test]
fn test_close_releases_the_automation_client() {
    let client = FakeClient::new(claude_tree());
    let (mut p, snapshot) = detected_provider(&client);
    p.close();
    assert_eq!(client.closed(), 1);
    p.close();
    assert_eq!(client.closed(), 1);
    // The window's profile went with it: no focus without a new detection.
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(false));
}

#[test]
fn test_focus_of_a_session_that_is_gone_fails() {
    let client = FakeClient::new(claude_tree());
    let (mut p, snapshot) = detected_provider(&client);
    let gone = pane_of(PaneId::Runtime(vec![42, 7, 999]), SECONDARY_RECT);
    assert_eq!(p.focus(&window(), &gone), Ok(false));
    let terminal = Pane::new(secondary_id(), SECONDARY_RECT, false, "windows_terminal");
    assert_eq!(p.focus(&window(), &terminal), Ok(false));
    let secondary = client.ids().secondary;
    client.update(|t| t.node(secondary).offscreen = true);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(false));
    assert!(client.focus_set().is_empty());
    // A window never detected is not focused into.
    let mut other = provider(&FakeClient::new(claude_tree()));
    assert_eq!(other.focus(&window(), &snapshot.panes[1]), Ok(false));
}

// ---------------------------------------------------------------- ChatGPT
#[test]
fn test_chatgpt_main_conversation_and_side_chat() {
    let tree = chatgpt_tree_with(ChatgptOptions {
        messages: 100,
        ..ChatgptOptions::default()
    });
    let focused = tree.ids.secondary_composer;
    let client = FakeClient::with_focus(tree, focused);
    let snapshot = detect(&mut provider(&client), &gpt_app()).expect("a snapshot");
    assert_eq!(ids(&snapshot), vec![main_id(), side_id()]); // not the small overlay
    assert_eq!(rects(&snapshot), vec![MAIN_RECT, SIDE_RECT]);
    assert_eq!(focused_id(&snapshot), Some(side_id()));
    assert_eq!(client.views().len(), 1);
    let steps = client.steps(0);
    assert!(steps <= 40, "{steps}");
    let ids = client.ids();
    let arena = client.update(|t| t.arena.clone());
    let touched: HashSet<usize> = client
        .reads(0)
        .iter()
        .filter(|(_, n)| *n != ids.secondary_composer)
        .map(|(_, n)| *n)
        .collect();
    assert!(
        !arena
            .descendants(ids.sidebar)
            .iter()
            .any(|n| touched.contains(n))
    );
    for pane in [ids.primary, ids.secondary] {
        assert!(!arena.descendants(pane).iter().any(|n| touched.contains(n)));
    }
    assert!(client.all_closed());
}

#[test]
fn test_chatgpt_container_of_another_build_is_found() {
    let tree = chatgpt_tree_with(ChatgptOptions {
        build_hash: "x7k2p_9",
        ..ChatgptOptions::default()
    });
    let snapshot = detect(&mut provider(&FakeClient::new(tree)), &gpt_app()).expect("a snapshot");
    assert_eq!(focused_id(&snapshot), Some(main_id()));
}

#[test]
fn test_chatgpt_without_a_side_chat_is_not_a_split() {
    // The main conversation and a small overlay: one pane.
    let tree = chatgpt_tree_with(ChatgptOptions {
        side: false,
        ..ChatgptOptions::default()
    });
    assert_eq!(
        detect(&mut provider(&FakeClient::new(tree)), &gpt_app()),
        None
    );
}

#[test]
fn test_chatgpt_container_without_prefix_token_is_not_used() {
    let mut tree = chatgpt_tree();
    let container = tree.arena.nodes[tree.ids.primary]
        .parent
        .expect("a container");
    tree.node(container).cls = "outline-none MainContentSurface".into();
    assert_eq!(
        detect(&mut provider(&FakeClient::new(tree)), &gpt_app()),
        None
    );
}

#[test]
fn test_chatgpt_focus_goes_to_the_message_box_of_each_side() {
    let tree = chatgpt_tree_with(ChatgptOptions {
        messages: 500,
        ..ChatgptOptions::default()
    });
    let ids = tree.ids;
    let client = FakeClient::with_focus(tree, ids.secondary_composer);
    let mut p = provider(&client);
    let snapshot = detect(&mut p, &gpt_app()).expect("a snapshot");
    assert_eq!(p.focus(&window(), &snapshot.panes[0]), Ok(true));
    assert_eq!(client.focus_set(), vec![ids.primary_composer]);
    assert_eq!(p.focus(&window(), &snapshot.panes[1]), Ok(true));
    assert_eq!(
        client.focus_set(),
        vec![ids.primary_composer, ids.secondary_composer]
    );
    assert!(client.all_closed());
}

#[test]
fn test_the_tree_not_built_yet_is_no_panes_for_chatgpt_too() {
    let mut tree = chatgpt_tree();
    let document = tree.ids.document;
    tree.node(document).children.clear();
    assert_eq!(
        detect(&mut provider(&FakeClient::new(tree)), &gpt_app()),
        None
    );
}

// --------------------------------------------------------------- deny-list
fn denied_always() -> Vec<AppIdentity> {
    vec![
        identity("code", "Chrome_WidgetWin_1"),
        identity("code - insiders", "Chrome_WidgetWin_1"),
        identity("cursor", "Chrome_WidgetWin_1"),
        identity("antigravity", "Chrome_WidgetWin_1"),
        identity("windsurf", "Chrome_WidgetWin_1"),
        // The ChatGPT process, but not its main window class.
        identity("chatgpt", "Chrome_WidgetWin_0"),
        identity("chrome", "Chrome_WidgetWin_1"),
        identity("msedge", "Chrome_WidgetWin_1"),
        identity("someapp", "Chrome_WidgetWin_1"),
        // The Claude process, but not its main window class.
        identity("claude", "Chrome_WidgetWin_0"),
        identity("claude", ""),
        // Another process with the Claude window class is not Claude.
        identity("claude-helper", "Chrome_WidgetWin_1"),
    ]
}

fn registry(client: &FakeClient, other: &RecordingProvider, desktop_apps: bool) -> PaneRegistry {
    PaneRegistry::new(vec![Box::new(other.clone()), Box::new(provider(client))])
        .with_desktop_apps(desktop_apps)
}

fn tmux() -> RecordingProvider {
    RecordingProvider::new("tmux").snapshot(|window| Some(two_panes(window.handle.clone())))
}

fn provider_names(providers: Vec<&mut dyn PaneProvider>) -> Vec<String> {
    providers.iter().map(|p| p.name().to_string()).collect()
}

fn synchronous_worker(registry: PaneRegistry) -> PaneWorker {
    PaneWorker::new(
        registry,
        PaneWorkerOptions {
            synchronous: true,
            ..PaneWorkerOptions::default()
        },
    )
}

type TreeFn = fn() -> ClaudeTree;

fn profiled() -> [(AppIdentity, TreeFn); 2] {
    [
        (app(), claude_tree as TreeFn),
        (gpt_app(), chatgpt_tree as TreeFn),
    ]
}

#[test]
fn test_profiled_apps_are_never_asked_while_the_setting_is_off() {
    for (app, tree) in profiled() {
        let client = FakeClient::new(tree());
        let other = tmux();
        let mut registry = registry(&client, &other, false);
        assert!(is_denied(&app, false));
        assert!(registry.providers_for(Some(&app)).is_empty());
        let worker = synchronous_worker(registry);
        worker.request_detect(&window(), &app);
        assert_eq!(client.calls(), Vec::<String>::new());
        assert_eq!(other.calls(), vec![]);
        assert_eq!(detected(&worker), vec![None]);
    }
}

#[test]
fn test_with_the_setting_on_profiled_apps_go_to_this_provider_alone() {
    for (app, tree) in profiled() {
        let client = FakeClient::new(tree());
        let other = tmux();
        let mut registry = registry(&client, &other, true);
        assert!(!is_denied(&app, true));
        assert_eq!(
            provider_names(registry.providers_for(Some(&app))),
            vec!["desktop_apps"]
        );
        // Terminals still go to their providers, never to this one.
        let terminal = identity("windowsterminal", "CASCADIA_HOSTING_WINDOW_CLASS");
        assert_eq!(
            provider_names(registry.providers_for(Some(&terminal))),
            vec!["tmux"]
        );
        // The calls the terminal lookup made on `other` are not part of the detection below.
        let before = other.calls().len();
        let worker = synchronous_worker(registry);
        worker.request_detect(&window(), &app);
        assert_eq!(client.calls(), vec!["foreground", "control_view"]);
        assert_eq!(other.calls().len(), before); // tmux is not asked about the app's window
        assert!(!other.calls().iter().any(|c| matches!(c, Call::Detect(_))));
        let results = detected(&worker);
        let [Some(snapshot)] = &results[..] else {
            panic!("expected one snapshot, got {results:?}");
        };
        assert_eq!(snapshot.provider(), Some("desktop_apps"));
    }
}

#[test]
fn test_other_chromium_apps_stay_denied() {
    for desktop_apps in [false, true] {
        for app in denied_always() {
            let client = FakeClient::new(claude_tree());
            let other = tmux();
            let mut registry = registry(&client, &other, desktop_apps);
            let label = format!("{}|{} desktop_apps={desktop_apps}", app.process, app.app_id);
            assert!(is_denied(&app, desktop_apps), "{label}");
            assert!(registry.providers_for(Some(&app)).is_empty(), "{label}");
            let worker = synchronous_worker(registry);
            worker.request_detect(&window(), &app);
            assert_eq!(client.calls(), Vec::<String>::new(), "{label}");
            assert_eq!(other.calls(), vec![], "{label}");
            assert_eq!(detected(&worker), vec![None], "{label}");
        }
    }
}

#[test]
fn test_default_providers_offer_desktop_apps_only_when_on_and_on_windows() {
    let mut settings = PaneSettings::default();
    assert!(!settings.desktop_apps);

    let names = |settings: &PaneSettings, system: &str| -> Vec<String> {
        default_providers(settings, Arc::new(|_| None), None, system)
            .iter()
            .map(|p| p.name().to_string())
            .collect()
    };

    assert!(!names(&settings, "win32").contains(&"desktop_apps".to_string()));
    settings.desktop_apps = true;
    if cfg!(windows) {
        assert_eq!(
            names(&settings, "win32"),
            vec!["wezterm", "windows_terminal", "desktop_apps", "tmux"]
        );
    } else {
        // The real factory has no UI Automation here.
        assert_eq!(names(&settings, "win32"), vec!["wezterm", "tmux"]);
    }
    assert!(!names(&settings, "linux").contains(&"desktop_apps".to_string()));
    assert!(!names(&settings, "darwin").contains(&"desktop_apps".to_string()));
}

#[test]
fn test_the_setting_is_off_by_default_and_says_what_it_costs() {
    assert!(!Settings::default().panes.desktop_apps);
    let row = describe_settings()
        .into_iter()
        .find(|row| row.key == "panes.desktop_apps")
        .expect("the setting is described");
    for needle in ["Claude", "ChatGPT", "Codex", "accessibility tree", "CPU"] {
        assert!(row.doc.contains(needle), "{needle}");
    }
}
