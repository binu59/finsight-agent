import os
import tempfile
import datetime
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src.agent import build_agent_executor
from src.tools.document_search_tool import build_index_from_file

# ── Page config ──────────────
st.set_page_config(
    page_title="FinSight Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed", # We don't use the native sidebar anymore
)

# ── CSS ──────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hide native sidebar completely if possible */
[data-testid="stSidebar"] { display: none !important; }

/* ── Custom Columns ────────────────────────────────────────────────── */
div[data-testid="column"]:has(.sidebar-marker) {
    position: fixed !important;
    top: 0;
    left: 0;
    height: 100vh;
    width: 260px !important;
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
    padding: 0 !important;
    overflow-y: auto;
    z-index: 999;
    display: flex;
    flex-direction: column;
    /* Slide animation */
    transform: translateX(-100%);
    transition: transform 0.28s cubic-bezier(0.4, 0, 0.2, 1),
                box-shadow 0.28s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: none;
}

/* Sidebar open state — applied by JS */
div[data-testid="column"]:has(.sidebar-marker).sidebar-open {
    transform: translateX(0) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.12) !important;
}

/* ── Sidebar user profile pinned to bottom ────────────────────────── */
.sidebar-profile {
    margin-top: auto;
    position: sticky;
    bottom: 0;
    background: #FFFFFF;
    border-top: 1px solid #E2E8F0;
    z-index: 10;
}

/* Main column — full width by default (sidebar closed) */
div[data-testid="column"]:nth-of-type(2) {
    margin-left: 0 !important;
    width: 100% !important;
    background: #F5F5F5 !important;
    min-height: 100vh;
    transition: margin-left 0.28s cubic-bezier(0.4, 0, 0.2, 1),
                width 0.28s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Main column shifts when sidebar is open */
div[data-testid="column"]:nth-of-type(2).main-shifted {
    margin-left: 260px !important;
    width: calc(100% - 260px) !important;
}

/* ── Hamburger toggle button ──────────────────────────────────────── */
#fs-sidebar-toggle {
    position: fixed;
    top: 12px;
    left: 14px;
    z-index: 1100;
    width: 38px;
    height: 38px;
    border: none;
    border-radius: 9px;
    background: #FFFFFF;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 5px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.12);
    transition: background 0.15s, box-shadow 0.15s;
    padding: 0;
}
#fs-sidebar-toggle:hover {
    background: #F3E8FF;
    box-shadow: 0 2px 8px rgba(124,58,237,0.18);
}
#fs-sidebar-toggle span {
    display: block;
    width: 18px;
    height: 2px;
    background: #374151;
    border-radius: 2px;
    transition: transform 0.25s, opacity 0.2s;
    transform-origin: center;
}
/* Animate to X when open */
#fs-sidebar-toggle.open span:nth-child(1) {
    transform: translateY(7px) rotate(45deg);
}
#fs-sidebar-toggle.open span:nth-child(2) {
    opacity: 0;
    transform: scaleX(0);
}
#fs-sidebar-toggle.open span:nth-child(3) {
    transform: translateY(-7px) rotate(-45deg);
}

/* ── Backdrop overlay ─────────────────────────────────────────────── */
#fs-sidebar-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.35);
    z-index: 998;
    backdrop-filter: blur(1px);
    animation: fsBackdropIn 0.2s ease;
}
#fs-sidebar-backdrop.visible { display: block; }

@keyframes fsBackdropIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}

/* ── Top bar ──────────────────────────────────────────────────────── */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px 14px 64px; /* leave room for hamburger */
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    position: sticky;
    top: 0;
    z-index: 100;
}
.top-bar-title { font-size: 15px; font-weight: 600; color: #111111; }
.agent-pill {
    display: inline-flex; align-items: center; gap: 5px;
    background: #F3E8FF; color: #7C3AED;
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 999px; margin-left: 10px; letter-spacing: 0.3px;
}
.agent-pill-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #7C3AED; display: inline-block;
}

/* ── Sidebar inner elements ───────────────────────────────────────── */
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 18px 16px 12px; border-bottom: 1px solid #F1F5F9;
}
.logo-icon {
    width: 34px; height: 34px;
    background: #7C3AED;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px; font-weight: 700; flex-shrink: 0;
}
.logo-text {
    font-size: 15px; font-weight: 700; color: #111111;
    letter-spacing: -0.3px;
}

.new-chat-btn {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    background: #1A1A1A; color: white;
    border: none; border-radius: 8px; padding: 9px 16px;
    font-size: 13px; font-weight: 600; cursor: pointer;
    width: calc(100% - 32px); margin: 12px 16px 8px;
    transition: background 0.15s; letter-spacing: 0.1px;
}
.new-chat-btn:hover { background: #2D2D2D; }

.sidebar-section-label {
    font-size: 10px; font-weight: 700; color: #6B7280;
    letter-spacing: 0.8px; text-transform: uppercase;
    padding: 10px 16px 4px;
}
.conv-item {
    padding: 8px 16px; cursor: pointer; border-radius: 6px;
    margin: 1px 8px; transition: background 0.1s;
}
.conv-item:hover { background: #F1F5F9; }
.conv-item.active { background: #F3E8FF; }
.conv-title {
    font-size: 13px; font-weight: 500; color: #111111;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Document status chip ─────────────────────────────────────────── */
.doc-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #F8FAFC; border: 1px solid #E2E8F0;
    color: #6B7280; font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 6px; margin: 6px 16px;
    max-width: calc(100% - 32px); overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
}
.doc-chip.active { background: #F3E8FF; border: 1px solid #D8B4FE; color: #7C3AED; }

/* ── Chat messages ────────────────────────────────────────────────── */
.msg-row {
    display: flex; align-items: flex-end; gap: 10px;
    padding: 6px 28px; width: 100%;
}
.msg-row.user { justify-content: flex-end; }

.msg-avatar {
    width: 30px; height: 30px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 700;
}
.msg-avatar.user { background: #7C3AED; color: white; }
.msg-avatar.assistant { background: #F3E8FF; color: #7C3AED; border: 1px solid #E9D5FF; }

.msg-bubble {
    padding: 12px 16px; font-size: 14px; line-height: 1.6;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06); margin-bottom: 6px;
}
/* User bubble natively in div */
.msg-bubble.user {
    background: #1A1A1A; color: #FFFFFF;
    border-radius: 14px; border-bottom-right-radius: 4px;
}

/* Agent bubble via col targeting */
div[data-testid="column"]:has(.agent-bubble-marker) {
    background: #FFFFFF !important;
    border: 1px solid #E5E5E5 !important;
    border-radius: 14px !important;
    border-bottom-left-radius: 4px !important;
    padding: 12px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06) !important;
}

.msg-time { font-size: 10px; color: #6B7280; margin-top: 4px; text-align: right; }
.msg-time.assistant { text-align: left; margin-left: 28px;}

/* ── File attachment chip ─────────────────────────────────────────── */
.bottom-file-chip {
    position: fixed; bottom: 85px; left: 50%;
    transform: translateX(-50%); background: #E5E7EB; color: #374151;
    padding: 6px 14px; border-radius: 16px; z-index: 1000; font-size: 12px;
    font-weight: 500; display: inline-flex; align-items: center; gap: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* ── Agent status card ────────────────────────────────────────────── */
.agent-status-card {
    background: #F3E8FF; border: 1px solid #E9D5FF;
    border-radius: 12px; padding: 12px 18px;
    display: flex; align-items: center; gap: 12px;
    max-width: 520px; margin: 24px auto 8px;
}
.agent-status-icon {
    width: 36px; height: 36px; background: #7C3AED; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; flex-shrink: 0; color: white;
}
.agent-status-title { font-size: 13px; font-weight: 600; color: #111111; }
.agent-status-sub { font-size: 12px; color: #6B7280; margin-top: 2px; }

/* ── Investment Brief card rendering ─────────────────────────────── */
.brief-card { margin-top: 4px; }
.brief-title {
    font-size: 15px; font-weight: 700; color: #111111;
    margin-bottom: 14px; padding-bottom: 10px;
    border-bottom: 1px solid #E5E5E5; letter-spacing: -0.2px;
}
.brief-section { margin-bottom: 12px; }
.brief-section-header {
    font-size: 12px; font-weight: 700; color: #7C3AED;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 5px;
}
.brief-section-body {
    font-size: 13px; color: #374151; line-height: 1.65;
}

/* ── Chat Input — full width by default, adjusts when sidebar open ── */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07) !important;
    left: 20px !important;
    right: 20px !important;
    width: calc(100% - 40px) !important;
    bottom: 16px !important;
    transition: left 0.28s cubic-bezier(0.4, 0, 0.2, 1),
                width 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
body.sidebar-open-body [data-testid="stChatInput"] {
    left: calc(260px + 20px) !important;
    right: 20px !important;
    width: calc(100% - 260px - 40px) !important;
}
[data-testid="stChatInputSubmitButton"] {
    color: #FFFFFF !important;
    background: #7C3AED !important;
    border-radius: 8px !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
    background: #6D28D9 !important;
}
/* Attachment icon inside the chat input - native Streamlit paperclip */
[data-testid="stChatInputFileButton"] svg { color: #7C3AED !important; }
</style>
""",unsafe_allow_html=True)

# ── Inject hamburger button + JS sidebar toggle ───────────────────────
st.markdown("""
<!-- Backdrop overlay — closes sidebar when clicking outside -->
<div id="fs-sidebar-backdrop"></div>

<!-- Persistent hamburger button -->
<button id="fs-sidebar-toggle" aria-label="Toggle sidebar" title="Toggle sidebar">
    <span></span>
    <span></span>
    <span></span>
</button>

<script>
(function() {
    var STORAGE_KEY = 'finsight_sidebar_open';

    function getSidebarCol() {
        return document.querySelector('div[data-testid="column"]:has(.sidebar-marker)');
    }
    function getMainCol() {
        var cols = document.querySelectorAll('div[data-testid="column"]');
        for (var i = 0; i < cols.length; i++) {
            if (!cols[i].querySelector('.sidebar-marker')) return cols[i];
        }
        return null;
    }

    function applyState(open) {
        var sidebar = getSidebarCol();
        var main    = getMainCol();
        var btn     = document.getElementById('fs-sidebar-toggle');
        var bd      = document.getElementById('fs-sidebar-backdrop');
        var profile = document.getElementById('fs-profile-strip');

        if (open) {
            if (sidebar) sidebar.classList.add('sidebar-open');
            if (main)    main.classList.add('main-shifted');
            if (btn)     btn.classList.add('open');
            if (bd)      bd.classList.add('visible');
            if (profile) profile.style.transform = 'translateX(0)';
            document.body.classList.add('sidebar-open-body');
        } else {
            if (sidebar) sidebar.classList.remove('sidebar-open');
            if (main)    main.classList.remove('main-shifted');
            if (btn)     btn.classList.remove('open');
            if (bd)      bd.classList.remove('visible');
            if (profile) profile.style.transform = 'translateX(-100%)';
            document.body.classList.remove('sidebar-open-body');
        }
    }

    function toggle() {
        var isOpen = sessionStorage.getItem(STORAGE_KEY) === 'true';
        var next   = !isOpen;
        sessionStorage.setItem(STORAGE_KEY, String(next));
        applyState(next);
    }

    function init() {
        var open = sessionStorage.getItem(STORAGE_KEY) === 'true';
        applyState(open);

        var btn = document.getElementById('fs-sidebar-toggle');
        if (btn && !btn._fsBound) {
            btn._fsBound = true;
            btn.addEventListener('click', toggle);
        }

        var bd = document.getElementById('fs-sidebar-backdrop');
        if (bd && !bd._fsBound) {
            bd._fsBound = true;
            bd.addEventListener('click', function() {
                sessionStorage.setItem(STORAGE_KEY, 'false');
                applyState(false);
            });
        }
    }

    // Run immediately then re-apply after every Streamlit DOM update
    init();
    var _observer = new MutationObserver(init);
    _observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────

def escape_dollars(text: str) -> str:
    # Fix Bug 1: escape $ as \$ before passing to st.markdown
    return text.replace("$", "\\$")

def format_time(dt: datetime.datetime) -> str:
    now = datetime.datetime.now()
    diff = now - dt
    if diff.seconds < 120: return "just now"
    if diff.seconds < 3600: return f"{diff.seconds // 60}m ago"
    if diff.days == 0: return dt.strftime("%I:%M %p").lstrip("0")
    if diff.days == 1: return "Yesterday"
    return dt.strftime("%b %d")

def group_conversations(convs):
    now = datetime.datetime.now()
    groups = {"TODAY": [], "YESTERDAY": [], "THIS WEEK": [], "OLDER": []}
    for c in convs:
        diff = (now - c["created_at"]).days
        if diff == 0: groups["TODAY"].append(c)
        elif diff == 1: groups["YESTERDAY"].append(c)
        elif diff <= 7: groups["THIS WEEK"].append(c)
        else: groups["OLDER"].append(c)
    return groups

def render_brief_as_card(text: str):
    if "INVESTMENT BRIEF" not in text:
        st.markdown(escape_dollars(text))
        return

    lines = text.strip().split("\n")
    title_line = next((l for l in lines if "INVESTMENT BRIEF" in l), "INVESTMENT BRIEF")

    section_icons = {
        "📊": ("Key Metrics", "#7C3AED"),
        "📰": ("Recent News", "#7C3AED"),
        "📋": ("From Annual Report", "#7C3AED"),
        "⚠️": ("Risk Flags", "#7C3AED"),
        "🔮": ("Outlook", "#7C3AED"),
    }

    sections = []
    current_icon = None
    current_lines = []
    for line in lines:
        if line.startswith("INVESTMENT BRIEF"):
            continue
        matched = False
        for icon in section_icons:
            if line.startswith(icon):
                if current_icon:
                    sections.append((current_icon, current_lines))
                current_icon = icon
                # Fix Bug 2: split by colon instead of stripping off len(icon)
                parts = line.split(":", 1)
                rest = parts[1].strip() if len(parts) > 1 else ""
                current_lines = [rest] if rest else []
                matched = True
                break
        if not matched and current_icon:
            current_lines.append(line)
    if current_icon:
        sections.append((current_icon, current_lines))

    # Render natively to avoid breaking HTML divs with markdown.
    st.markdown(f"**{title_line.strip()}**")
    st.divider()
    
    for icon, lines_arr in sections:
        label, color = section_icons[icon]
        st.markdown(f"<span style='color:{color};font-weight:700;font-size:12px;text-transform:uppercase;'>{icon} {label}</span>", unsafe_allow_html=True)
        # Double newlines ensure markdown handles bullet points properly inside the stream
        safe_content = escape_dollars("\n".join(lines_arr).strip())
        st.markdown(safe_content + "\n\n")

def render_user_message(content: str, timestamp: datetime.datetime = None):
    ts = format_time(timestamp) if timestamp else ""
    st.markdown(
        f'<div class="msg-row user">'
        f'<div style="max-width: 70%; display: flex; flex-direction: column; align-items: flex-end;">'
        f'<div class="msg-bubble user">{content}</div>'
        f'<div class="msg-time">{ts}</div>'
        f'</div>'
        f'<div class="msg-avatar user">U</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Session state initialisation ─────────────────────────────────────
def _init_state():
    defaults = {
        "conversations": [],
        "active_conv_id": None,
        "uploaded_index": None,
        "last_uploaded_filename": None,
        "executor": None,
        "active_doc_name": "Default (Apple 10-K)",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if not st.session_state.conversations:
        _new_conversation()

def _new_conversation():
    conv_id = str(len(st.session_state.conversations) + 1)
    conv = {"id": conv_id, "title": "New conversation", "created_at": datetime.datetime.now(), "messages": []}
    st.session_state.conversations.insert(0, conv)
    st.session_state.active_conv_id = conv_id
    st.session_state.executor = build_agent_executor(verbose=False, document_index=st.session_state.uploaded_index)

def _active_conv():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_conv_id:
             return c
    return None

_init_state()

# ── Layout: Two Columns ───────────────────────────────────────────────

col_sidebar, col_main = st.columns([260, 1000])

# --- LEFT SIDEBAR ---
with col_sidebar:
    st.markdown('<div class="sidebar-marker"></div>', unsafe_allow_html=True)
    
    st.markdown(
        '<div class="sidebar-logo"><div class="logo-icon">📊</div>'
        '<span class="logo-text">FinSight Agent</span></div>',
        unsafe_allow_html=True,
    )

    if st.button("＋ New chat", key="new_chat_btn", use_container_width=True):
        _new_conversation()
        st.rerun()

    # Conversation history search (placeholder)

    groups = group_conversations(st.session_state.conversations)
    for group_label, convs in groups.items():
        if not convs: continue
        st.markdown(f'<p class="sidebar-section-label">{group_label}</p>', unsafe_allow_html=True)
        for conv in convs:
            is_active = conv["id"] == st.session_state.active_conv_id
            active_class = " active" if is_active else ""
            st.markdown(f'<div class="conv-item{active_class}"><div class="conv-title">{conv["title"]}</div></div>', unsafe_allow_html=True)

    # Profile — slides with sidebar (JS controls visibility via id)
    st.markdown(
        '<div id="fs-profile-strip" style="position:fixed;bottom:0;left:0;width:260px;'
        'background:#FFFFFF;border-top:1px solid #E2E8F0;z-index:1000;'
        'display:flex;align-items:center;gap:10px;padding:14px 16px;'
        'transform:translateX(-100%);transition:transform 0.28s cubic-bezier(0.4,0,0.2,1);">'
        '<div style="width:32px;height:32px;border-radius:50%;background:#7C3AED;flex-shrink:0;'
        'display:flex;align-items:center;justify-content:center;color:white;'
        'font-size:13px;font-weight:700;">U</div>'
        '<div><div style="font-size:13px;font-weight:600;color:#111111">FinSight User</div>'
        '<div style="font-size:11px;color:#7C3AED;font-weight:500;">Free plan</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

# --- MAIN CHAT AREA ---
with col_main:
    conv = _active_conv()
    
    st.markdown(
        f'<div class="top-bar" style="justify-content: center; padding: 20px;">'
        f'<span style="font-size: 28px; font-weight: 800; background: linear-gradient(90deg, #7C3AED, #C084FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">Welcome to FinSight</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


        
    st.markdown('<div style="padding: 20px;">', unsafe_allow_html=True)

    if conv:
        for msg in conv["messages"]:
            if msg["role"] == "user":
                 render_user_message(msg["content"], msg.get("timestamp"))
            else:
                 col_avatar, col_bubble = st.columns([1, 11])
                 with col_avatar:
                     st.markdown('<div class="msg-avatar assistant">📊</div>', unsafe_allow_html=True)
                 with col_bubble:
                     st.markdown('<div class="agent-bubble-marker"></div>', unsafe_allow_html=True)
                     render_brief_as_card(msg["content"])
                     ts = format_time(msg["timestamp"]) if msg.get("timestamp") else ""
                     st.markdown(f'<div class="msg-time assistant">{ts}</div>', unsafe_allow_html=True)

    # Show active document chip above the input bar
    if st.session_state.active_doc_name != "Default (Apple 10-K)":
         st.markdown(
             f'<div class="bottom-file-chip">📄 {st.session_state.active_doc_name} &times;</div>',
             unsafe_allow_html=True,
         )

    # ── Native file-upload embedded in the chat input bar (Streamlit ≥ 1.38) ──
    chat_msg = st.chat_input(
        "Ask about a company or attach an annual report…",
        accept_file="multiple",
        file_type=["pdf"],
    )

    # Unpack: chat_msg is None, or a ChatInputValue with .text and .files
    user_input = None
    if chat_msg is not None:
        user_input = chat_msg.text
        uploaded_files = chat_msg.files          # list of UploadedFile objects

        # Process any attached PDF files
        for uploaded_file in uploaded_files:
            if uploaded_file.name != st.session_state.last_uploaded_filename:
                with st.spinner(f"Indexing {uploaded_file.name}…"):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name
                        new_index = build_index_from_file(tmp_path)
                        st.session_state.uploaded_index = new_index
                        st.session_state.active_doc_name = uploaded_file.name
                        st.session_state.last_uploaded_filename = uploaded_file.name
                        st.session_state.executor = build_agent_executor(
                            verbose=False, document_index=new_index
                        )
                        st.toast(f"✅ {uploaded_file.name} indexed!", icon="📄")
                    except Exception as e:
                        st.error(f"Error indexing file: {e}")

    if user_input and conv is not None:
         ts = datetime.datetime.now()
         if not conv["messages"]:
             conv["title"] = user_input[:42] + ("…" if len(user_input) > 42 else "")
             
         attachment = st.session_state.active_doc_name if st.session_state.active_doc_name != "Default (Apple 10-K)" else None
         conv["messages"].append({"role": "user", "content": user_input, "timestamp": ts, "attachment": attachment})
         
         # The immediate re-render hack so user message shows before assistant streams
         render_user_message(user_input, ts)
         
         col_avatar, col_bubble = st.columns([1, 11])
         with col_avatar:
             st.markdown('<div class="msg-avatar assistant">📊</div>', unsafe_allow_html=True)
         with col_bubble:
             st.markdown('<div class="agent-bubble-marker"></div>', unsafe_allow_html=True)
             with st.expander("🔍 Agent reasoning (Thought / Action / Observation)", expanded=False):
                 st_callback = StreamlitCallbackHandler(st.container())
                 result = st.session_state.executor.invoke({"input": user_input}, {"callbacks": [st_callback]})
                 
             output = result["output"]
             agent_ts = datetime.datetime.now()
             render_brief_as_card(output)
             conv["messages"].append({"role": "assistant", "content": output, "timestamp": agent_ts})
         st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)