import os
import base64
import re
import time
import requests
from pathlib import Path

import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8765")

st.set_page_config(page_title="VideoRAG - Okko", layout="wide", initial_sidebar_state="collapsed")

# Logo
logo_path = os.path.join(os.path.dirname(__file__), "brand", "okko_white.png")
logo_b64 = ""
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

if 'query' not in st.session_state:
    st.session_state.query = ""

st.markdown(f"""
<style>
    :root {{
        --okko-primary: #4A3AFF;
        --okko-deep-base: #0B041C;
        --okko-deep-accent: #1C054D;
        --okko-white: #FFFFFF;
        --text-main: #FFFFFF;
        --text-muted: #A0A5B5;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --timebar-bg: rgba(255,255,255,0.15);
    }}
    .stApp {{
        background: radial-gradient(circle at top center, var(--okko-deep-accent) 0%, var(--okko-deep-base) 100%);
        background-attachment: fixed;
        color: var(--text-main);
    }}
    .main .block-container {{
        padding-top: 0rem; padding-bottom: 3rem; max-width: 1400px;
        background: transparent !important;
    }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
    .hero-container-top {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        margin-top: 6vh; margin-bottom: 2rem; animation: fadeInDown 0.8s ease-out forwards;
    }}
    .hero-container-top .logo-img {{ height: 70px; margin-bottom: 10px; object-fit: contain; }}
    .hero-container-top h1 {{
        font-family: 'Inter', system-ui, sans-serif; font-weight: 800; font-size: 3.5rem !important;
        margin-bottom: 0; padding-bottom: 0; color: var(--okko-white); text-align: center;
    }}
    .hero-subtitle-wrapper {{ width: 100%; display: flex; justify-content: center; margin-top: 1rem; margin-bottom: 3rem; }}
    .hero-subtitle {{ color: var(--text-muted); font-size: 1.15rem; max-width: 650px; line-height: 1.5; text-align: center; }}
    .compact-header-top {{
        display: flex; align-items: center; justify-content: center; gap: 15px;
        animation: fadeIn 0.4s ease forwards; margin-bottom: -5px; margin-top: -110px;
    }}
    .compact-header-top .logo-img {{ height: 35px; object-fit: contain; }}
    .compact-header-top .divider {{ color: rgba(255,255,255,0.3); font-size: 1.8rem; font-weight: 300; }}
    .compact-header-top h1 {{
        font-family: 'Inter', system-ui, sans-serif; font-weight: 800; font-size: 1.8rem !important;
        margin: 0; padding: 0; color: var(--okko-white);
    }}
    .glass-card {{
        background: var(--glass-bg); border: 1px solid var(--glass-border); border-radius: 16px;
        padding: 16px; backdrop-filter: blur(16px); box-shadow: 0 8px 32px 0 rgba(0,0,0,0.2);
        height: 100%; display: flex; flex-direction: column; animation: fadeInUp 0.5s ease forwards;
        position: relative;
    }}
    [data-testid="column"] > div {{ gap: 0 !important; }}
    .card-rank {{
        font-size: 3rem; font-weight: 900; color: rgba(255,255,255,0.05);
        position: absolute; top: 5px; right: 15px; pointer-events: none;
    }}
    .card-title {{ font-size: 1.05rem; font-weight: 700; color: var(--okko-white); margin-bottom: 8px; z-index: 10; position: relative; }}
    .timecode-area {{
        display: flex; align-items: center; gap: 10px;
        font-family: 'Courier New', monospace; font-size: 0.75rem; color: var(--text-muted); margin-bottom: 12px;
    }}
    .timecode-bar-wrapper {{ flex-grow: 1; height: 5px; background: var(--timebar-bg); border-radius: 3px; position: relative; overflow: hidden; }}
    .timecode-fill {{ background: rgba(255,255,255,0.4); position: absolute; height: 100%; width: 100%; border-radius: 3px; }}
    .card-desc {{ font-size: 0.9rem; color: rgba(255,255,255,0.7); line-height: 1.5; flex-grow: 1; }}
    .stats-container-wrapper {{ display: flex; justify-content: center; margin-top: -20px; margin-bottom: 0px; }}
    .search-stats-glass {{
        background: rgba(0,0,0,0.3); border: 1px solid rgba(74,58,255,0.2); border-radius: 50px;
        padding: 10px 24px; display: flex; align-items: center; justify-content: center; gap: 15px;
        backdrop-filter: blur(8px); color: var(--okko-white); font-size: 0.95rem;
    }}
    .stats-highlight {{ color: #BDB2FF; font-weight: 700; }}
    @keyframes fadeInDown {{ from {{ opacity: 0; transform: translateY(-30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes fadeInUp {{ from {{ opacity: 0; transform: translateY(30px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
    div[data-testid="stTextInput"] {{ max-width: 650px; margin: 0 auto; }}
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {{
        border-color: #4A3AFF !important; box-shadow: 0 0 0 1px #4A3AFF !important;
    }}
    div[data-testid="stVideo"] {{ border-radius: 10px; overflow: hidden; margin-bottom: 12px; border: 1px solid rgba(0,0,0,0.3); box-shadow: 0 4px 15px rgba(0,0,0,0.5); }}
</style>
""", unsafe_allow_html=True)

is_searched = bool(st.session_state.get('last_query', ''))
logo_img_tag = f"<img src='data:image/png;base64,{logo_b64}' class='logo-img'>" if logo_b64 else "<h2>Okko</h2>"

if not is_searched:
    st.markdown(f'<div class="hero-container-top">{logo_img_tag}<h1>VideoRAG</h1></div>', unsafe_allow_html=True)
    user_input = st.text_input("q", placeholder="Опишите сцену или фразу...", label_visibility="collapsed", key="hero_input")
    st.markdown('<div class="hero-subtitle-wrapper"><div class="hero-subtitle">Интеллектуальный поиск фрагментов видео по текстовому запросу</div></div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="compact-header-top">{logo_img_tag}<span class="divider">|</span><h1>VideoRAG</h1></div>', unsafe_allow_html=True)
    user_input = st.text_input("q", value=st.session_state.last_query, placeholder="Опишите сцену...", label_visibility="collapsed", key="compact_input")

active_query = user_input.strip()

if active_query and active_query != st.session_state.get("last_query", ""):
    st.session_state.last_query = active_query
    status_placeholder = st.empty()
    status_placeholder.markdown('<div style="text-align:center;color:#BDB2FF;font-weight:600;font-size:1.1rem;margin-top:50px;animation:pulse 1s infinite;">Поиск...</div><style>@keyframes pulse{0%{opacity:0.3}50%{opacity:1}100%{opacity:0.3}}</style>', unsafe_allow_html=True)

    try:
        resp = requests.get(f"{BACKEND_URL}/search", params={"q": active_query, "top_k": 5}, timeout=15)
        data = resp.json()
    except Exception as e:
        status_placeholder.empty()
        st.error(f"Backend error: {e}")
        st.stop()

    results = []
    for r in data.get("results", []):
        results.append({
            'video_id': r['video_id'],
            'chunk_type': r.get('chunk_type', 'window'),
            'start_time': r['start_time'],
            'end_time': r['end_time'],
            'text': r.get('transcript', ''),
            'score': r.get('score', 0),
        })

    st.session_state.results = results
    st.session_state.search_meta = {'latency_ms': data.get('latency_ms', 0)}
    status_placeholder.empty()
    st.rerun()

if is_searched:
    results = st.session_state.get("results", [])
    meta = st.session_state.get("search_meta", {})

    if not results:
        st.error("Ничего не найдено.")
    else:
        latency = meta.get('latency_ms', 0)
        st.markdown(f'<div class="stats-container-wrapper"><div class="search-stats-glass"><div style="color:#D3D3E0;">Latency: <span class="stats-highlight">{latency:.0f}ms</span> | Результатов: <span class="stats-highlight">{len(results)}</span></div></div></div>', unsafe_allow_html=True)

        def render_card(scene, rank):
            start_time = scene['start_time']
            end_time = scene['end_time']
            vname = scene['video_id']

            st.markdown(f"""
            <div class='glass-card'>
                <div class='card-rank'>0{rank}</div>
                <div class='card-title'>{vname}</div>
            """, unsafe_allow_html=True)

            video_url = f"{BACKEND_URL}/video/{vname}"
            st.video(video_url, start_time=int(start_time))

            st.markdown(f"""
                <div class='timecode-area'>
                    <span>{start_time:04.1f}s</span>
                    <div class='timecode-bar-wrapper'><div class='timecode-fill'></div></div>
                    <span>{end_time:04.1f}s</span>
                </div>
                <div class='card-desc'>{scene.get('text', '')[:300]}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(results) >= 2:
            st.markdown("<div style='margin-bottom:20px;'>", unsafe_allow_html=True)
            top_cols = st.columns(2)
            with top_cols[0]: render_card(results[0], 1)
            with top_cols[1]: render_card(results[1], 2)
            st.markdown("</div>", unsafe_allow_html=True)

        if len(results) >= 3:
            remaining = results[2:5]
            bottom_cols = st.columns(len(remaining))
            for i, col in enumerate(bottom_cols):
                with col:
                    render_card(remaining[i], i + 3)
