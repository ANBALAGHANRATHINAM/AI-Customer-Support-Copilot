"""Premium Streamlit chat presentation for the ZENDS Support Copilot."""

from __future__ import annotations

import html
import json
import csv
import sys
import os
from pathlib import Path
from typing import Any

import streamlit as st


# The support stack uses the saved PyTorch models; resolve that backend before
# importing the response-engine modules that eventually lazy-load Transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "4_AI_Response_Engine" / "code", ROOT / "5_Streamlit_Integration" / "code"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm import StaticAcknowledgementLLM
from response_engine import ZendsResponseEngine
from ui_helpers import OUT_OF_SCOPE_RESPONSE, apply_scope_guard, format_confidence, is_abstention, priority_style


st.set_page_config(
    page_title="ZENDS Communications | AI Customer Support Copilot", 
    page_icon="Z", 
    layout="wide", 
    initial_sidebar_state="locked"
)


@st.cache_resource(show_spinner="Loading secure support resources...")
def load_response_engine() -> ZendsResponseEngine:
    """Reuse the existing, fully local Segment 4 response-engine integration."""
    return ZendsResponseEngine.from_project_assets(llm=StaticAcknowledgementLLM())


def _escape(value: Any) -> str:
    return html.escape(str(value))


def inject_styles() -> None:
    """Create a desktop-first enterprise AI workspace without changing behavior."""
    st.markdown(
        """
        <style>
        :root {
          --ink:#172233; --muted:#64748b; --line:#e2e8f0; --canvas:#f7f8fa; --surface:#ffffff;
          --blue:#3569d4; --blue-dark:#2a54ab; --blue-soft:#eaf0fd; --shadow:0 1px 2px rgba(23,34,51,.05); --shadow-md:0 4px 14px rgba(23,34,51,.07);
        }
        .stApp { background-color:var(--canvas); color:var(--ink); font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif; }
        .stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp textarea,.stApp button { font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif; }
        [data-testid="stHeader"] { background:rgba(247,248,250,.9); } [data-testid="stToolbar"] { display:none; }
        .block-container { max-width:1120px; min-height:calc(100vh - 3rem); padding:2rem 2.5rem 2.5rem; }

        [data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); min-width:232px; max-width:232px; opacity:1 !important; visibility:visible !important; }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.1rem; }
        .sidebar-brand { align-items:flex-start; display:flex; gap:.65rem; margin:.2rem .1rem 1.9rem; }
        .brand-mark { align-items:center; background:var(--blue); border-radius:9px; color:#fff; display:flex; font-size:1rem; font-weight:750; height:34px; justify-content:center; width:34px; }
        .brand-name { color:var(--ink); font-size:.96rem; font-weight:720; letter-spacing:-.01em; line-height:1.15; }
        .brand-caption { color:var(--muted); font-size:.71rem; line-height:1.3; margin-top:.25rem; }
        .nav-label { color:#94a3b8; font-size:.64rem; font-weight:730; letter-spacing:.1em; margin:0 .6rem .45rem; text-transform:uppercase; }
        .nav-active { background:var(--blue-soft); border:1px solid #d7e3fa; border-radius:9px; color:var(--blue-dark); font-size:.86rem; font-weight:650; margin:.08rem 0 .3rem; padding:.58rem .72rem; }
        .sidebar-footer { border-top:1px solid var(--line); color:#94a3b8; font-size:.7rem; line-height:1.5; margin:2.6rem .1rem .3rem; padding:.9rem .55rem 0; }
        [data-testid="stSidebar"] [data-testid="stButton"] button { background:transparent; border:0; border-radius:9px; box-shadow:none; color:#475569; font-size:.86rem; font-weight:560; justify-content:flex-start; min-height:37px; padding:.5rem .72rem; text-align:left; transition:background .14s ease,color .14s ease; }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover { background:#f1f5f9; color:var(--ink); }

        .topbar { align-items:center; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; margin:0 0 1.4rem; padding:0 0 1.1rem; }
        .eyebrow { color:var(--blue); font-size:.65rem; font-weight:730; letter-spacing:.1em; margin-bottom:.35rem; text-transform:uppercase; }
        .page-title { color:var(--ink); font-size:1.7rem; font-weight:730; letter-spacing:-.03em; line-height:1.1; margin:0; }
        .page-subtitle { color:var(--muted); font-size:.9rem; margin:.4rem 0 0; }
        .new-chat-note { color:#94a3b8; font-size:.71rem; margin-top:.3rem; text-align:right; }
        .stButton > button { background:var(--surface); border:1px solid var(--line); border-radius:9px; box-shadow:none; color:#334155; font-size:.84rem; font-weight:610; transition:border-color .14s ease,background .14s ease; }
        .stButton > button:hover { background:#f8fafc; border-color:#c7d2e0; }

        .empty-state { background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow); margin:1.6rem auto .9rem; max-width:640px; padding:2.1rem 1.8rem 1.9rem; text-align:center; }
        .ai-orb { align-items:center; background:var(--blue-soft); border-radius:12px; color:var(--blue-dark); display:inline-flex; font-size:1.05rem; font-weight:730; height:42px; justify-content:center; margin-bottom:.95rem; width:42px; }
        .empty-title { color:var(--ink); font-size:1.28rem; font-weight:700; letter-spacing:-.02em; }
        .empty-copy { color:var(--muted); font-size:.92rem; line-height:1.55; margin:.5rem auto 0; max-width:480px; }
        .suggestion-caption { color:#94a3b8; font-size:.71rem; margin:1.1rem 0 .5rem; }

        .chat-area { margin:0 auto; max-width:940px; }
        .message-row { display:flex; margin:1rem 0; width:100%; }
        .message-row.customer { justify-content:flex-end; }
        .message-stack { display:flex; gap:.6rem; max-width:75%; }
        .message-row.customer .message-stack { flex-direction:row-reverse; }
        .message-meta { color:#94a3b8; font-size:.66rem; font-weight:700; letter-spacing:.06em; margin:0 0 .28rem; text-transform:uppercase; }
        .message-row.customer .message-meta { text-align:right; }
        .message-avatar { align-items:center; background:var(--blue-soft); border-radius:10px; color:var(--blue-dark); display:flex; flex:0 0 32px; font-size:.78rem; font-weight:730; height:32px; justify-content:center; margin-top:1.3rem; width:32px; }
        .message-row.customer .message-avatar { background:#eef2f6; color:#475569; }
        .message-bubble { background:var(--surface); border:1px solid var(--line); border-radius:14px 14px 14px 4px; box-shadow:var(--shadow); color:var(--ink); font-size:.95rem; line-height:1.62; padding:.85rem 1rem; white-space:pre-wrap; }
        .message-row.customer .message-bubble { background:var(--blue-soft); border-color:#d7e3fa; border-radius:14px 14px 4px 14px; color:#1c3766; }

        .insight-panel { background:var(--surface); border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow); margin:.9rem 0 1.05rem 2.6rem; max-width:700px; padding:.9rem 1rem; }
        .insight-heading { color:#94a3b8; font-size:.66rem; font-weight:730; letter-spacing:.1em; margin:0 0 .65rem; text-transform:uppercase; }
        .insight-grid { display:grid; gap:.6rem; grid-template-columns:repeat(3,1fr); }
        .insight-card { background:#fafbfc; border:1px solid var(--line); border-left:3px solid #cbd5e1; border-radius:8px; min-width:0; padding:.62rem .72rem; }
        .insight-card.intent { border-left-color:var(--blue); }
        .insight-card.sentiment { border-left-color:#8b5cf6; }
        .insight-card.priority { border-left-color:#94a3b8; }
        .insight-name { color:#94a3b8; font-size:.64rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; }
        .insight-value { color:var(--ink); font-size:.94rem; font-weight:680; margin-top:.22rem; overflow-wrap:anywhere; }
        .insight-detail { color:#94a3b8; font-size:.74rem; margin-top:.1rem; }
        .priority-high { color:#b3413a; } .priority-medium { color:#a06a12; } .priority-low { color:#2f7a4f; }
        .grounding-note { color:#a06a12; font-size:.74rem; margin-top:.6rem; }
        .error-bubble { background:#fff7f6; border:1px solid #f3d3d0; border-radius:14px; color:#8a2f28; font-size:.9rem; line-height:1.6; margin:0 0 1.4rem 2.6rem; max-width:700px; padding:.85rem 1rem; }
        .error-title { font-weight:700; margin-bottom:.2rem; }

        [data-testid="stExpander"] { background:var(--surface); border:1px solid var(--line); border-radius:10px; box-shadow:none; margin:0 0 1.6rem 2.6rem; max-width:700px; }
        [data-testid="stExpander"] summary { color:#475569; font-size:.83rem; font-weight:620; }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] { background:#fafbfc; border-radius:0 0 10px 10px; }

        /* Composer: a normal, in-flow st.form — deliberately NOT stChatInput, whose native
           stBottom wrapper is position:sticky/bottom:0 (viewport-anchored). This block-level
           form scrolls with the page like any other element and can never cover prior messages. */
        [data-testid="stForm"] { background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow-md); margin:1.5rem auto 0; max-width:940px; padding:.5rem .55rem; transition:border-color .14s ease,box-shadow .14s ease; }
        [data-testid="stForm"]:focus-within { border-color:var(--blue); box-shadow:0 0 0 3px rgba(53,105,212,.12),var(--shadow-md); }
        [data-testid="stForm"] [data-testid="stTextInput"] { margin:0; }
        [data-testid="stForm"] [data-testid="stTextInput"] > div, [data-testid="stForm"] [data-baseweb="input"] { background:transparent; border:0; box-shadow:none; }
        [data-testid="stForm"] input { background:transparent; border:0; box-shadow:none !important; color:var(--ink); font-size:.94rem; padding:.6rem .5rem; }
        [data-testid="stForm"] input:focus { outline:none; }
        [data-testid="stForm"] [data-testid="stFormSubmitButton"] button { background:var(--blue); border:0; border-radius:11px; color:#fff; font-weight:650; min-height:2.3rem; transition:background .14s ease; }
        [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover { background:var(--blue-dark); }
        .composer-hint { color:#94a3b8; font-size:.72rem; margin:.5rem auto 0; max-width:940px; text-align:center; }

        .workflow-row { align-items:center; display:flex; flex-wrap:wrap; gap:.4rem; margin:.3rem 0 .2rem; }
        .workflow-step { background:var(--surface); border:1px solid var(--line); border-radius:8px; color:#334155; font-size:.8rem; font-weight:600; padding:.4rem .68rem; white-space:nowrap; }
        .workflow-arrow { color:#94a3b8; font-size:.8rem; }

        /* ========================================================= PREMIUM ANALYTICS + ABOUT PAGES ========================================================= */

        .analytics-hero {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 28px;
            margin: 10px 0 24px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: var(--shadow-md);
        }

        .analytics-hero-icon {
            width: 52px;
            height: 52px;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #edf4ff;
            color: #2563eb;
            font-size: 24px;
            font-weight: 800;
        }

        .analytics-eyebrow,
        .section-label {
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .12em;
            color: #64748b;
            text-transform: uppercase;
        }

        .analytics-hero h2 {
            margin: 3px 0 5px;
            color: #111827;
            font-size: 26px;
            font-weight: 800;
        }

        .analytics-hero p {
            margin: 0;
            color: #64748b;
            font-size: 14px;
            line-height: 1.55;
        }

        .analytics-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 20px;
            min-height: 115px;
            box-shadow: var(--shadow-md);
        }

        .card-label {
            font-size: 10px;
            font-weight: 800;
            letter-spacing: .1em;
            color: #64748b;
            text-transform: uppercase;
        }

        .card-value {
            margin-top: 10px;
            font-size: 29px;
            line-height: 1;
            font-weight: 800;
            color: #111827;
        }

        .card-caption {
            margin-top: 9px;
            color: #94a3b8;
            font-size: 12px;
        }

        .model-pipeline {
            padding: 22px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: var(--shadow-md);
        }

        .pipeline-title {
            font-size: 10px;
            font-weight: 800;
            letter-spacing: .1em;
            color: #64748b;
            margin-bottom: 18px;
            text-transform: uppercase;
        }

        .pipeline-items {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
            flex-wrap: wrap;
        }

        .pipeline-node {
            flex: 1;
            min-width: 100px;
            padding: 14px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: #f8fafc;
            text-align: center;
        }

        .pipeline-node span {
            display: block;
            font-size: 10px;
            font-weight: 800;
            color: #2563eb;
            margin-bottom: 5px;
            text-transform: uppercase;
        }

        .pipeline-node strong {
            color: #1e293b;
            font-size: 13px;
            display: block;
        }

        .pipeline-arrow {
            color: #94a3b8;
            font-size: 18px;
            flex: 0 0 auto;
        }

        .panel-heading {
            margin-bottom: 12px;
        }

        .panel-heading h3,
        .info-panel h3 {
            margin: 5px 0 0;
            color: #111827;
            font-weight: 800;
        }

        .info-panel,
        .project-card {
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 22px;
            box-shadow: var(--shadow-md);
        }

        .config-row {
            display: flex;
            justify-content: space-between;
            gap: 15px;
            padding: 13px 0;
            border-bottom: 1px solid #eef2f7;
            font-size: 13px;
        }

        .config-row span {
            color: #64748b;
        }

        .config-row strong {
            color: #1e293b;
            font-weight: 700;
        }

        .info-panel-footer {
            margin-top: 16px;
            padding-top: 14px;
            color: #94a3b8;
            font-size: 11px;
            line-height: 1.5;
        }

        .analytics-note {
            display: flex;
            gap: 16px;
            align-items: center;
            padding: 18px 20px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            color: #64748b;
            font-size: 13px;
        }

        .analytics-note strong {
            color: #1e293b;
            white-space: nowrap;
            font-weight: 700;
        }

        /* ABOUT PAGE */

        .about-hero {
            min-height: 330px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 35px;
            padding: 42px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow: var(--shadow-md);
            overflow: hidden;
        }

        .about-hero-copy {
            max-width: 650px;
        }

        .about-hero h1 {
            margin: 8px 0 15px;
            font-size: 44px;
            line-height: 1.08;
            letter-spacing: -.035em;
            color: #111827;
            font-weight: 800;
        }

        .about-hero p {
            max-width: 650px;
            color: #64748b;
            font-size: 15px;
            line-height: 1.7;
        }

        .about-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 22px;
        }

        .about-tags span {
            padding: 7px 11px;
            background: #f1f5f9;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            color: #475569;
            font-size: 11px;
            font-weight: 700;
        }

        .about-hero-visual {
            position: relative;
            width: 230px;
            height: 230px;
            flex-shrink: 0;
        }

        .visual-core {
            position: absolute;
            inset: 55px;
            border-radius: 50%;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: #2563eb;
        }

        .visual-core div {
            font-size: 32px;
            font-weight: 900;
        }

        .visual-core span {
            font-size: 9px;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .visual-orbit {
            position: absolute;
            inset: 18px;
            border: 1px solid #dbeafe;
            border-radius: 50%;
        }

        .orbit-two {
            inset: 0;
            border-color: #e2e8f0;
        }

        .content-title {
            margin: 5px 0 18px;
            color: #111827;
            font-size: 25px;
            letter-spacing: -.02em;
            font-weight: 800;
        }

        .capability-card {
            min-height: 180px;
            padding: 20px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            box-shadow: var(--shadow-md);
        }

        .capability-number {
            color: #2563eb;
            font-size: 11px;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .capability-card h3 {
            margin: 18px 0 9px;
            color: #111827;
            font-size: 17px;
            font-weight: 800;
        }

        .capability-card p {
            margin: 0;
            color: #64748b;
            font-size: 12px;
            line-height: 1.55;
        }

        .workflow-card {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 16px 20px;
            margin-bottom: 8px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: var(--shadow-md);
        }

        .workflow-number {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: #eff6ff;
            color: #2563eb;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 900;
            flex-shrink: 0;
            text-transform: uppercase;
        }

        .workflow-content {
            display: flex;
            flex-direction: column;
            gap: 3px;
            flex: 1;
        }

        .workflow-content strong {
            color: #1e293b;
            font-size: 14px;
            font-weight: 800;
        }

        .workflow-content span {
            color: #64748b;
            font-size: 12px;
        }

        .workflow-connector {
            color: #94a3b8;
            font-size: 20px;
        }

        .workflow-vertical {
            display: flex;
            flex-direction: column;
            gap: 0;
        }

        .workflow-card-vertical {
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 100%;
        }

        .workflow-arrow-vertical {
            color: #94a3b8;
            font-size: 24px;
            text-align: center;
            padding: 8px 0;
        }

        .tech-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .tech-card {
            padding: 15px;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 13px;
        }

        .tech-card strong {
            display: block;
            color: #1e293b;
            font-size: 13px;
            font-weight: 800;
        }

        .tech-card span {
            display: block;
            margin-top: 5px;
            color: #94a3b8;
            font-size: 11px;
        }

        .project-card h2 {
            margin: 7px 0 12px;
            color: #111827;
            font-weight: 800;
        }

        .project-card p {
            color: #64748b;
            font-size: 13px;
            line-height: 1.7;
        }

        .principle-row {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 13px 0;
            border-top: 1px solid #eef2f7;
            font-size: 12px;
        }

        .principle-row span {
            min-width: 50px;
            color: #2563eb;
            font-size: 10px;
            font-weight: 900;
            text-transform: uppercase;
        }

        .principle-row strong {
            color: #334155;
            font-weight: 700;
        }

        .about-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 20px 22px;
            background: #111827;
            border-radius: 16px;
            color: white;
        }

        .about-footer strong {
            display: block;
            font-size: 13px;
            font-weight: 700;
        }

        .about-footer span {
            display: block;
            margin-top: 4px;
            color: #94a3b8;
            font-size: 11px;
        }

        .footer-badge {
            padding: 7px 10px;
            border: 1px solid #374151;
            border-radius: 8px;
            color: #cbd5e1;
            font-size: 9px;
            font-weight: 800;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        @media (max-width:800px) {
          .block-container { padding:1.25rem .9rem 2rem; }
          .message-stack { max-width:92%; }
          .insight-panel,[data-testid="stExpander"],.error-bubble { margin-left:0; max-width:100%; }
          .insight-grid { grid-template-columns:1fr; }
          .page-title { font-size:1.4rem; }
          .workflow-step { font-size:.74rem; padding:.34rem .55rem; }
          .about-hero { flex-direction: column; gap: 25px; min-height: auto; }
          .tech-grid { grid-template-columns: 1fr; }
          .pipeline-items { flex-wrap: wrap; }
        }
        </style>
""",
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render the three-page navigation."""
    with st.sidebar:
        st.markdown(
'<div class="sidebar-brand"><div class="brand-mark">Z</div><div><div class="brand-name">ZENDS<br>Communications</div><div class="brand-caption">AI Customer Support Copilot</div></div></div>',
            unsafe_allow_html=True,
        )

        st.markdown(
'<div class="nav-label">WORKSPACE</div>',
            unsafe_allow_html=True,
        )

        current_page = str(
            st.session_state.get("active_page", "Copilot")
        )

        pages = [
            ("Copilot", "💬", "Copilot"),
            ("Model Analytics", "📊", "Model Analytics"),
            ("About Project", "ℹ️", "About Project"),
        ]

        for page_key, icon, label in pages:
            if current_page == page_key:
                st.markdown(
f'<div class="nav-active">{icon}&nbsp;&nbsp;{label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"{icon}  {label}",
                    key=f"sidebar_{page_key}",
                    width="stretch",
                ):
                    st.session_state.active_page = page_key
                    st.rerun()

        st.markdown(
'<div class="sidebar-footer"><strong>AI for Better Connections</strong><br>People&nbsp;&nbsp;•&nbsp;&nbsp;Technology&nbsp;&nbsp;•&nbsp;&nbsp;Trust</div>',
            unsafe_allow_html=True,
        )

    return current_page


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="topbar"><div><div class="eyebrow">ZENDS Communications</div><h1 class="page-title">{_escape(title)}</h1><p class="page-subtitle">{_escape(subtitle)}</p></div></div>',
        unsafe_allow_html=True,
    )


def _read_json(path: Path) -> dict[str, Any]:
    """Read an optional existing artifact without making it a runtime dependency."""
    try:
        with path.open(encoding="utf-8") as artifact:
            value = json.load(artifact)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def render_model_analytics():
    render_page_header(
        "Model Analytics",
        "A clear view of model performance, training configuration, and evaluation results."
    )

    metrics_path = ROOT / "2_NLP_Intelligence" / "evaluation" / "intent_test_metrics.json"
    config_path = ROOT / "2_NLP_Intelligence" / "evaluation" / "intent_training_config.json"
    sentiment_path = ROOT / "2_NLP_Intelligence" / "evaluation" / "sentiment_metrics.json"

    metrics = {}
    config = {}
    sentiment = {}

    try:
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}

    try:
        if config_path.exists():
            config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        config = {}

    try:
        if sentiment_path.exists():
            sentiment = json.loads(sentiment_path.read_text(encoding="utf-8"))
    except Exception:
        sentiment = {}

    # Intent accuracy from test metrics
    accuracy = (
        metrics.get("accuracy") or
        metrics.get("test_accuracy") or
        metrics.get("intent_accuracy") or
        0
    )

    # Sentiment accuracy from sentiment_metrics.json
    sentiment_accuracy = (
        sentiment.get("accuracy") or
        sentiment.get("test_accuracy") or
        sentiment.get("sentiment_accuracy") or
        0
    )

    def format_score(value):
        if isinstance(value, (int, float)):
            return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"
        return str(value)

    # Hero section
    st.markdown(
'<div class="analytics-hero"><div class="analytics-hero-icon">◈</div><div><div class="analytics-eyebrow">MODEL INTELLIGENCE</div><h2>Performance at a glance</h2><p>Evaluation results from the trained ZENDS intent classifier and sentiment intelligence layer.</p></div></div>',
        unsafe_allow_html=True,
    )

    # KPI cards
    st.markdown('<div class="section-label">CORE PERFORMANCE</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'<div class="analytics-card"><div class="card-label">INTENT ACCURACY</div><div class="card-value">{format_score(accuracy)}</div><div class="card-caption">Test-set classification</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="analytics-card"><div class="card-label">SENTIMENT ACCURACY</div><div class="card-value">{format_score(sentiment_accuracy)}</div><div class="card-caption">100 manually curated messages</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # AI Pipeline visualization
    st.markdown(
'<div class="model-pipeline"><div class="pipeline-title">AI INTELLIGENCE PIPELINE</div><div class="pipeline-items"><div class="pipeline-node"><span>01</span><strong>Customer Query</strong></div><div class="pipeline-arrow">→</div><div class="pipeline-node"><span>02</span><strong>Intent</strong></div><div class="pipeline-arrow">→</div><div class="pipeline-node"><span>03</span><strong>Sentiment</strong></div><div class="pipeline-arrow">→</div><div class="pipeline-node"><span>04</span><strong>Priority</strong></div><div class="pipeline-arrow">→</div><div class="pipeline-node"><span>05</span><strong>RAG + Response</strong></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    # Evaluation details
    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
'<div class="panel-heading"><div><div class="section-label">EVALUATION</div><h3>Intent model results</h3></div></div>',
            unsafe_allow_html=True,
        )

        # Try to load confusion matrix from CSV first
        confusion_csv_path = ROOT / "2_NLP_Intelligence" / "evaluation" / "intent_test_confusion_matrix.csv"
        confusion = None
        
        if confusion_csv_path.exists():
            try:
                import pandas as pd
                confusion_df = pd.read_csv(confusion_csv_path, index_col=0)
                st.dataframe(
                    confusion_df,
                    width="stretch",
                    height=300,
                )
            except Exception as e:
                st.info("Confusion matrix data is available but could not be rendered.")
        else:
            # Fallback to metrics JSON if CSV doesn't exist
            confusion = metrics.get("confusion_matrix")
            if confusion:
                try:
                    import pandas as pd

                    labels = metrics.get(
                        "labels",
                        [
                            "Billing",
                            "Refund",
                            "Technical",
                            "Complaint",
                            "Product Inquiry",
                        ],
                    )

                    df = pd.DataFrame(
                        confusion,
                        index=labels[:len(confusion)],
                        columns=labels[:len(confusion[0])],
                    )

                    st.dataframe(
                        df,
                        width="stretch",
                        height=300,
                    )
                except Exception:
                    st.info("Confusion matrix data is available but could not be rendered.")
            else:
                st.info("Confusion matrix data is not available.")

    with right:
        st.markdown(
'<div class="info-panel"><div class="section-label">TRAINING CONFIGURATION</div><h3>Intent classifier</h3>',
            unsafe_allow_html=True,
        )

        model_name = config.get(
            "model_name",
            config.get("base_model", "DistilBERT")
        )

        epochs = config.get("epochs", "—")
        batch_size = config.get("batch_size", "—")
        learning_rate = config.get("learning_rate", "—")

        st.markdown(
            f'<div class="config-row"><span>Base model</span><strong>{html.escape(str(model_name))}</strong></div><div class="config-row"><span>Epochs</span><strong>{html.escape(str(epochs))}</strong></div><div class="config-row"><span>Batch size</span><strong>{html.escape(str(batch_size))}</strong></div><div class="config-row"><span>Learning rate</span><strong>{html.escape(str(learning_rate))}</strong></div><div class="info-panel-footer">Evaluation data is read from the project\'s existing evaluation artifacts.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    st.markdown(
'<div class="analytics-note"><strong>What this page shows</strong><span>Model Analytics is intentionally separated from the live Copilot experience so agents can focus on customer conversations while technical evaluation remains available as a dedicated view.</span></div>',
        unsafe_allow_html=True,
    )


def render_about_project():
    render_page_header(
        "About Project",
        "A production-style AI support copilot designed for ZENDS Communications."
    )

    st.markdown(
'<div class="about-hero"><div class="about-hero-copy"><div class="analytics-eyebrow">ZENDS AI SUPPORT PLATFORM</div><h1>Faster answers.<br>Better support.</h1><p>An AI-powered customer support copilot that helps agents understand customer intent, detect sentiment, determine urgency, retrieve relevant ZENDS knowledge, and generate a grounded response.</p><div class="about-tags"><span>Intent Classification</span><span>Sentiment Analysis</span><span>RAG</span><span>AI Response Generation</span></div></div><div class="about-hero-visual"><div class="visual-orbit orbit-one"></div><div class="visual-orbit orbit-two"></div><div class="visual-core"><div>AI</div><span>COPILOT</span></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(
'<div class="section-label">WHAT THE SYSTEM DOES</div><h2 class="content-title">From customer message to recommended response</h2>',
        unsafe_allow_html=True,
    )

    capabilities = [
        ("01", "Understand", "Identifies the customer's support intent from natural language."),
        ("02", "Detect", "Recognizes sentiment and translates it into a simple support signal."),
        ("03", "Prioritize", "Determines urgency using the project's priority logic."),
        ("04", "Retrieve", "Searches the ZENDS knowledge base for relevant policy and product information."),
        ("05", "Respond", "Produces a recommended response grounded in retrieved information."),
    ]

    cols = st.columns(5)

    for col, (number, title, description) in zip(cols, capabilities):
        with col:
            st.markdown(
                f'<div class="capability-card"><div class="capability-number">{number}</div><h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    st.markdown(
'<div class="section-label">SYSTEM WORKFLOW</div><h2 class="content-title">How the copilot works</h2>',
        unsafe_allow_html=True,
    )

    workflow = [
        ("01", "Customer Query", "Customer enters a support message."),
        ("02", "NLP Intelligence", "Intent and sentiment are identified."),
        ("03", "Priority", "The support urgency is determined."),
        ("04", "Knowledge Retrieval", "Relevant ZENDS information is retrieved."),
        ("05", "Response Engine", "A recommended response is generated."),
        ("06", "Agent Review", "The support agent reviews the recommendation."),
    ]

    st.markdown('<div class="workflow-vertical">', unsafe_allow_html=True)
    for index, (number, title, description) in enumerate(workflow):
        st.markdown(
            f'<div class="workflow-card-vertical"><div class="workflow-card"><div class="workflow-number">{number}</div><div class="workflow-content"><strong>{title}</strong><span>{description}</span></div></div></div>',
            unsafe_allow_html=True,
        )
        if index < len(workflow) - 1:
            st.markdown(
                '<div class="workflow-arrow-vertical">↓</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown(
'<div class="section-label">TECHNOLOGY STACK</div><h2 class="content-title">Built as an end-to-end data science project</h2><div class="tech-grid"><div class="tech-card"><strong>Python</strong><span>Core application & ML pipeline</span></div><div class="tech-card"><strong>Hugging Face</strong><span>NLP model ecosystem</span></div><div class="tech-card"><strong>DistilBERT</strong><span>Intent classification</span></div><div class="tech-card"><strong>Sentence Transformers</strong><span>Semantic embeddings</span></div><div class="tech-card"><strong>ChromaDB</strong><span>Vector knowledge retrieval</span></div><div class="tech-card"><strong>Streamlit</strong><span>Interactive support interface</span></div></div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
'<div class="project-card"><div class="section-label">PROJECT PRINCIPLE</div><h2>Human-in-the-loop AI</h2><p>The copilot is designed to assist support agents rather than replace them. AI analysis and recommendations remain visible to the agent, allowing the human to review the suggested response before using it.</p><div class="principle-row"><span>AI</span><strong>Analyze → Retrieve → Recommend</strong></div><div class="principle-row"><span>HUMAN</span><strong>Review → Decide → Respond</strong></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    st.markdown(
'<div class="about-footer"><div><strong>ZENDS Communications — AI Customer Support Copilot</strong><span>Data Science • NLP • RAG • Generative AI • Streamlit</span></div><div class="footer-badge">PROJECT DEMO</div></div>',
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
'<div class="empty-state"><div class="empty-state-icon">✦</div><h2 class="empty-title">How can I help today?</h2><p class="empty-copy">Ask about billing, refunds, technical issues, complaints, or ZENDS products.</p></div>',
        unsafe_allow_html=True,
    )

    suggestions = [
        "What is the ZENDS refund policy?",
        "What discounts does ZENDS offer?",
        "What support tiers does ZENDS offer?"
    ]

    cols = st.columns(3)

    for col, suggestion in zip(cols, suggestions):
        with col:
            if st.button(
                suggestion,
                key=f"suggestion_{suggestion}",
                width="stretch",
            ):
                return suggestion

    return None


def render_message(role: str, content: str) -> None:
    """Render one labelled conversation message, independent of backend logic."""
    escaped = _escape(content)
    if role == "customer":
        markup = f'<div class="message-row customer"><div class="message-stack"><div class="message-avatar">C</div><div><div class="message-meta">Customer</div><div class="message-bubble">{escaped}</div></div></div></div>'
    else:
        markup = f'<div class="message-row assistant"><div class="message-stack"><div class="message-avatar">Z</div><div><div class="message-meta">ZENDS AI</div><div class="message-bubble">{escaped}</div></div></div></div>'
    st.markdown(markup, unsafe_allow_html=True)


def render_analysis(result: dict[str, Any]) -> None:
    """Render the latest response's returned analysis in compact presentation cards."""
    out_of_scope = str(result["recommended_response"]) == OUT_OF_SCOPE_RESPONSE
    if out_of_scope:
        intent, intent_detail = "—", "Not applicable"
        sentiment, sentiment_detail = "—", "Not applicable"
        priority, priority_class = "—", ""
    else:
        intent, intent_detail = str(result["intent"]), format_confidence(float(result["intent_confidence"]))
        sentiment, sentiment_detail = str(result["sentiment"]), format_confidence(float(result["sentiment_confidence"]))
        priority_class, priority = priority_style(str(result["priority"]))

    grounding_note = (
        '<div class="grounding-note">Grounding unavailable for this ZENDS question.</div>'
        if not out_of_scope and is_abstention(str(result["recommended_response"]))
        else ""
    )
    st.markdown(
        f"""<div class="insight-panel"><div class="insight-heading">AI Analysis</div><div class="insight-grid">
        <div class="insight-card intent"><div class="insight-name">Intent</div><div class="insight-value">{_escape(intent)}</div><div class="insight-detail">{_escape(intent_detail)}</div></div>
        <div class="insight-card sentiment"><div class="insight-name">Sentiment</div><div class="insight-value">{_escape(sentiment)}</div><div class="insight-detail">{_escape(sentiment_detail)}</div></div>
        <div class="insight-card priority"><div class="insight-name">Priority</div><div class="insight-value {priority_class}">{_escape(priority)}</div></div>
        </div>{grounding_note}</div>""",
        unsafe_allow_html=True,
    )


def render_details(result: dict[str, Any]) -> None:
    """Keep existing retrieved context available only on deliberate inspection."""
    with st.expander("View Details & Sources", expanded=False):
        chunks = result.get("retrieved_context", [])
        if not chunks:
            st.caption("No ZENDS knowledge chunks were retrieved for this request.")
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "ZENDS Communications")
            page = metadata.get("page", "—")
            chunk_id = metadata.get("chunk_id", "—")
            policy = metadata.get("policy_category")
            st.caption(
                f"{source} · page {page} · {chunk_id}"
                + (f" · Policy: {policy}" if policy else "")
            )
            st.write(chunk.get("text", ""))


def render_error(message: str) -> None:
    """Show a stable, on-brand error state."""
    st.markdown(
        '<div class="error-bubble"><div class="error-title">Something went wrong processing this message.</div><div>Please try again in a moment. If this keeps happening, let the support team know.</div></div>',
        unsafe_allow_html=True,
    )
    with st.expander("Technical details", expanded=False):
        st.caption(_escape(message))


def main() -> None:
    inject_styles()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    current_page = render_sidebar()
    if current_page == "Model Analytics":
        render_model_analytics()
        return
    if current_page == "About Project":
        render_about_project()
        return

    header_left, header_right = st.columns((6, 1.1), vertical_alignment="center")
    with header_left:
        st.markdown('<div class="topbar"><div><div class="eyebrow">ZENDS Communications</div><h1 class="page-title">AI Customer Support Copilot</h1><p class="page-subtitle">Get intelligent support responses for ZENDS customer queries</p></div></div>', unsafe_allow_html=True)
    with header_right:
        new_chat = st.button("＋ New Chat", key="new_chat", width="stretch")
        st.markdown('<div class="new-chat-note">Start a fresh conversation</div>', unsafe_allow_html=True)
    if new_chat:
        st.session_state.chat_history = []
        st.rerun()

    pending_query = render_empty_state() if not st.session_state.chat_history else None
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for turn in st.session_state.chat_history:
        render_message("customer", str(turn["query"]))
        if turn.get("error"):
            render_error(str(turn["error"]))
        else:
            render_message("assistant", str(turn["result"]["recommended_response"]))
            render_analysis(turn["result"])
            render_details(turn["result"])
    st.markdown('</div>', unsafe_allow_html=True)

    with st.form("composer_form", clear_on_submit=True, border=False):
        input_col, button_col = st.columns((11, 1.6), vertical_alignment="bottom")
        with input_col:
            typed_message = st.text_input(
                "Message",
                placeholder="Type your message here...",
                label_visibility="collapsed",
                key="composer_text",
            )
        with button_col:
            sent = st.form_submit_button("Send ↑", width="stretch")
    st.markdown('<div class="composer-hint">Press Enter or Send to submit</div>', unsafe_allow_html=True)

    query = (typed_message.strip() if sent and typed_message.strip() else None) or pending_query
    if query:
        try:
            with st.spinner("Analyzing your query..."):
                result = load_response_engine().respond(query)
                result = apply_scope_guard(query, result)
            st.session_state.chat_history.append({"query": query, "result": result, "error": None})
        except Exception as exc:  # noqa: BLE001 - keep the page stable; never let a backend failure break the UI shell
            st.session_state.chat_history.append({"query": query, "result": None, "error": str(exc)})
        st.rerun()


if __name__ == "__main__":
    main()