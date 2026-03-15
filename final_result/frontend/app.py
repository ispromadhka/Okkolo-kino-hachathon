"""
Streamlit frontend for Video RAG Search.
Connects to FastAPI backend for search and video streaming.
"""
import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8765")

st.set_page_config(page_title="Video RAG Search — Okko", page_icon="🎬", layout="wide")

st.title("🎬 Video RAG Search")
st.caption("Поиск фрагментов видео по текстовому запросу")

# Sidebar
with st.sidebar:
    st.header("Настройки")
    top_k = st.slider("Количество результатов", 1, 10, 5)
    st.divider()
    st.markdown("**Архитектура:**")
    st.markdown("- Fine-tuned BGE-M3 (1024d)")
    st.markdown("- Answer augmentation")
    st.markdown("- Dynamic HyDE")
    st.markdown("- Score: **0.577** (1st place)")
    st.divider()
    # Health check
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.success(f"Backend: {health['status']} ({health['chunks']} chunks)")
    except Exception:
        st.error("Backend недоступен")

# Search
query = st.text_input(
    "Что вы ищете?",
    placeholder='например, "How to build a table step by step" или "Как произносится слово на урду"',
)

if query:
    with st.spinner("Ищем..."):
        try:
            resp = requests.get(
                f"{BACKEND_URL}/search",
                params={"q": query, "top_k": top_k},
                timeout=10,
            )
            data = resp.json()
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Latency", f"{data['latency_ms']}ms")
    col2.metric("HyDE", "Yes" if data['hyde_used'] else "No")
    col3.metric("HyDE similarity", f"{data['hyde_similarity']:.3f}")

    st.divider()

    # Results
    results = data.get("results", [])
    if not results:
        st.warning("Ничего не найдено")
        st.stop()

    # Scene selector
    scene_labels = [
        f"#{r['rank']} — {r['video_id']} [{r['start_time']:.0f}s - {r['end_time']:.0f}s] "
        f"(score: {r['score']:.3f}, {r['chunk_type']})"
        for r in results
    ]
    selected = st.radio("Найденные фрагменты:", range(len(results)),
                        format_func=lambda i: scene_labels[i])

    r = results[selected]
    st.divider()

    col_player, col_info = st.columns([3, 2])

    with col_player:
        st.subheader(f"🎬 {r['video_id']}")

        video_url = f"{BACKEND_URL}/video/{r['video_id']}"
        start_sec = r["start_time"]
        end_sec = r["end_time"]

        # Custom HTML5 player with start/end time
        st.components.v1.html(f"""
            <video id="player" width="100%" controls
                   style="border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                <source src="{video_url}" type="video/mp4">
            </video>
            <script>
                const video = document.getElementById('player');
                video.currentTime = {start_sec};
                video.addEventListener('loadedmetadata', function() {{
                    video.currentTime = {start_sec};
                }});
                video.addEventListener('timeupdate', function() {{
                    if (video.currentTime >= {end_sec}) {{
                        video.pause();
                    }}
                }});
                video.play().catch(e => {{}});
            </script>
            <div style="margin-top: 8px; font-size: 14px; color: #888;">
                Фрагмент: {start_sec:.1f}s — {end_sec:.1f}s
                (длительность: {end_sec - start_sec:.1f}s)
            </div>
        """, height=420)

    with col_info:
        st.subheader("Детали")
        st.markdown(f"**Video:** `{r['video_id']}`")
        st.markdown(f"**Время:** {r['start_time']:.1f}s — {r['end_time']:.1f}s")
        st.markdown(f"**Score:** {r['score']:.4f}")
        st.markdown(f"**Тип:** {r['chunk_type']}")

        st.divider()
        st.subheader("Транскрипт")
        if r.get("transcript"):
            st.text_area("", r["transcript"], height=200, disabled=True)
        else:
            st.info("Транскрипт недоступен")
