import streamlit as st
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import striprtf
import re

# Настройка страницы
st.set_page_config(page_title="Смысловой поиск по нормативам", layout="wide")
st.title("🧠 Смысловой поиск по нормативной документации")
st.markdown("---")

# Загружаем модель
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Функция для извлечения текста из RTF
def extract_text_from_rtf(rtf_path):
    try:
        with open(rtf_path, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        text = striprtf.rtf_to_text(rtf_content)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        st.error(f"Ошибка при чтении {rtf_path}: {e}")
        return ""

# Функция для разбивки текста на чанки
def split_into_chunks(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Функция для поиска
def search_semantic(query, chunks, chunk_metadata, model, top_k=10):
    query_embedding = model.encode([query])
    chunk_embeddings = model.encode(chunks)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(chunks[i], chunk_metadata[i], similarities[i]) for i in top_indices if similarities[i] > 0.3]
    return results

# Показываем список файлов в папке docs
st.sidebar.header("📚 Библиотека нормативов")

# Путь к папке с документами (в репозитории)
docs_folder = Path("./docs")

if not docs_folder.exists():
    st.error("❌ Папка 'docs' не найдена! Создайте её в репозитории и добавьте RTF-файлы.")
    st.info("📖 Инструкция: на GitHub создайте папку 'docs' и загрузите туда ваши RTF-файлы")
    st.stop()

# Получаем список RTF-файлов
rtf_files = list(docs_folder.glob("*.rtf"))

if not rtf_files:
    st.warning("📁 В папке 'docs' нет RTF-файлов. Добавьте их через GitHub.")
    st.info("1. Зайдите в ваш репозиторий на GitHub")
    st.info("2. Нажмите 'Add file' → 'Upload files'")
    st.info("3. Выберите папку 'docs' и загрузите RTF-файлы")
    st.info("4. После загрузки обновите эту страницу")
    st.stop()

# Отображаем список файлов
with st.sidebar:
    st.write(f"📄 Всего нормативов: {len(rtf_files)}")
    for f in rtf_files:
        st.write(f"   - {f.name}")
    
    st.markdown("---")
    st.caption("💡 Чтобы добавить новый норматив, загрузите RTF-файл в папку 'docs' на GitHub")

# Поисковый запрос
query = st.text_input("✏️ Введите запрос:", placeholder="например: сечение проводников заземления")

# Кнопка поиска
if st.button("🔎 Искать по смыслу", type="primary") and query:
    if len(query) < 3:
        st.warning("Введите минимум 3 символа")
        st.stop()
    
    with st.spinner(f"🧠 Обрабатываю {len(rtf_files)} нормативов..."):
        all_chunks = []
        all_metadata = []
        
        progress_bar = st.progress(0)
        for idx, rtf_file in enumerate(rtf_files):
            progress_bar.progress((idx + 1) / len(rtf_files))
            
            text = extract_text_from_rtf(rtf_file)
            if not text:
                continue
            
            chunks = split_into_chunks(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'норматив': rtf_file.stem,
                    'файл': rtf_file.name
                })
        
        progress_bar.empty()
        
        if not all_chunks:
            st.warning("Не удалось извлечь текст из файлов. Проверьте формат RTF.")
            st.stop()
        
        results = search_semantic(query, all_chunks, all_metadata, model)
        
        if results:
            st.success(f"✅ Найдено {len(results)} релевантных фрагментов")
            st.markdown("---")
            
            for i, (chunk, meta, score) in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Результат {i}**")
                        st.write(chunk[:400] + "..." if len(chunk) > 400 else chunk)
                    with col2:
                        st.metric("Сходство", f"{score:.2%}")
                    
                    st.caption(f"📌 Источник: **{meta['норматив']}** (файл: {meta['файл']})")
                    st.divider()
        else:
            st.warning(f"😕 Ничего не найдено по запросу '{query}'")

# Инструкция
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Как пополнить библиотеку")
    st.markdown("""
    1. Зайдите на **GitHub** в ваш репозиторий
    2. Нажмите **'Add file' → 'Upload files'**
    3. Выберите папку **'docs'** на своем компьютере
    4. Перетащите новые RTF-файлы
    5. Нажмите **'Commit changes'**
    6. Обновите эту страницу — новый норматив появится в поиске!
    """)