import os
import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Video RAG MVP", layout="wide")

@st.cache_resource
def load_vectorstore():
    """Load ChromaDB with HuggingFace embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    return Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embeddings, 
        collection_name="video_scenes"
    )

st.title("🎬 Video RAG Search MVP (Локальный Поиск)")
st.markdown("Мгновенный поиск сцен по смыслу и действиям без внешних API!")

# Sidebar for config or info
with st.sidebar:
    st.header("Настройки")
    st.info("Эта версия использует исключительно векторный поиск (ChromaDB), что гарантирует максимальную производительность без задержек на генерацию LLM.")
    video_file_path = st.text_input("Путь к видеофайлу (для плеера)", value="movie.mp4")

vectorstore = load_vectorstore()

query = st.text_input("Что вы ищете? (например, 'сцена под дождем, где обсуждают план')")

if query:
    if "last_query" not in st.session_state or st.session_state.last_query != query:
        with st.spinner("Быстрый поиск сцен..."):
            # Retrieve top 3 relevant documents instantly
            results = vectorstore.similarity_search(query, k=3)
            st.session_state.results = results
            st.session_state.last_query = query

    results = st.session_state.get("results", [])
    
    if not results:
        st.warning("Ничего не найдено. Возможно, база данных пуста.")
    else:
        st.subheader("Найденные сцены")
        
        # Let the user select which scene to play
        scene_options = [f"Сцена {i+1} ({doc.metadata.get('start_time'):.1f}s - {doc.metadata.get('end_time'):.1f}s)" for i, doc in enumerate(results)]
        
        selected_scene_idx = st.radio("Выберите сцену для просмотра в плеере:", options=range(len(scene_options)), format_func=lambda x: scene_options[x], horizontal=True, index=0)
        
        selected_scene = results[selected_scene_idx]
        start_time = selected_scene.metadata.get("start_time", 0.0)
        end_time = selected_scene.metadata.get("end_time", 0.0)
        
        # Layout: Left column for Video, Right column for Info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Плеер: Сцена {selected_scene_idx+1}")
            if os.path.exists(video_file_path):
                st.video(video_file_path, start_time=int(start_time))
            else:
                st.error(f"Файл видео '{video_file_path}' не найден. Плеер недоступен.")
        
        with col2:
            st.subheader("Извлеченный контекст сцены")
            st.markdown(f"**Предыдущие реплики:**\n_{selected_scene.metadata.get('prev_dialogue', '—')}_")
            st.markdown(f"**Следующие реплики:**\n_{selected_scene.metadata.get('next_dialogue', '—')}_")
            st.markdown("---")
            st.text("Контекст в базе данных:")
            st.text(selected_scene.page_content)
