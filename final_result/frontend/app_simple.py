"""
Streamlit frontend for Video RAG Search — Okko Hackathon.
"""
import streamlit as st
import requests
import os

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8765")

st.set_page_config(page_title="Video RAG Search — Okko", page_icon="🎬", layout="wide")

# Sidebar
with st.sidebar:
    st.header("Настройки")
    top_k = st.slider("Количество результатов", 1, 10, 5)
    st.divider()
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=2).json()
        st.success(f"Backend: {health['status']} ({health['chunks']} chunks)")
    except Exception:
        st.error("Backend недоступен")

# Header
st.title("🎬 Video RAG Search")
st.caption("Поиск фрагментов видео по текстовому запросу")

# Search
query = st.text_input(
    "Что вы ищете?",
    placeholder='например, "How to build a table step by step"',
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

    # Latency only
    st.metric("Latency", f"{data['latency_ms']}ms")
    st.divider()

    results = data.get("results", [])
    if not results:
        st.warning("Ничего не найдено")
        st.stop()

    # Results list
    for i, r in enumerate(results):
        duration = r['end_time'] - r['start_time']
        with st.expander(
            f"#{r['rank']}  {r['video_id']}  [{r['start_time']:.0f}s — {r['end_time']:.0f}s]  "
            f"({duration:.0f}s)  score: {r['score']:.3f}",
            expanded=(i == 0),
        ):
            col_player, col_info = st.columns([3, 2])

            with col_player:
                video_url = f"{BACKEND_URL}/video/{r['video_id']}"
                start_sec = r["start_time"]
                end_sec = r["end_time"]

                st.components.v1.html(f"""
                    <video id="player_{i}" width="100%" controls
                           style="border-radius: 8px;">
                        <source src="{video_url}" type="video/mp4">
                    </video>
                    <script>
                        const v{i} = document.getElementById('player_{i}');
                        v{i}.currentTime = {start_sec};
                        v{i}.addEventListener('loadedmetadata', function() {{
                            v{i}.currentTime = {start_sec};
                        }});
                        v{i}.addEventListener('timeupdate', function() {{
                            if (v{i}.currentTime >= {end_sec}) {{
                                v{i}.pause();
                            }}
                        }});
                    </script>
                """, height=360)

            with col_info:
                st.markdown(f"**Video:** `{r['video_id']}`")
                st.markdown(f"**Время:** {r['start_time']:.1f}s — {r['end_time']:.1f}s ({duration:.0f}s)")
                st.markdown(f"**Score:** {r['score']:.4f}")
                if r.get("transcript"):
                    st.markdown("**Транскрипт:**")
                    st.text_area("", r["transcript"], height=180, disabled=True, key=f"tr_{i}")
