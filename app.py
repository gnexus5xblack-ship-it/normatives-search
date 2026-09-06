import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Настройка страницы ---
st.set_page_config(page_title="Смысловой поиск по нормативам", layout="wide")
st.title("🧠 Точный поиск по нормативной документации")
st.markdown("**Ищет по смыслу + ключевым словам (гибридный поиск)**")
st.markdown("---")

# --- Загрузка модели (улучшенная) ---
@st.cache_resource
def load_model():
    # Модель, специально обученная на технических/научных текстах
    return SentenceTransformer('sentence-transformers/allenai-specter')

model = load_model()

# --- Очистка текста от мусора ---
def clean_text(text):
    # Удаляем номера страниц, оглавления, повторы
    text = re.sub(r'Стр\.\s*\d+', '', text)
    text = re.sub(r'Страница\s*\d+', '', text)
    text = re.sub(r'\d+\s*из\s*\d+', '', text)
    text = re.sub(r'[\\/*?:"<>|]', ' ', text)
    # Удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Извлечение текста из TXT (с очисткой) ---
def extract_text_from_txt(txt_path):
    try:
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'koi8-r']:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text = f.read()
                text = clean_text(text)
                return text
            except UnicodeDecodeError:
                continue
        return ""
    except Exception:
        return ""

# --- Умная разбивка на фрагменты по предложениям ---
def split_by_sentences(text, max_len=350, overlap=50):
    # Разбиваем по точкам, вопросам, восклицаниям
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_len and current_chunk:
            chunks.append(' '.join(current_chunk))
            # Перекрытие: оставляем последние overlap слов
            overlap_words = ' '.join(current_chunk).split()[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + [sent]
            current_len = len(overlap_words) + sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# --- Гибридный поиск (смысл + ключевые слова) ---
def hybrid_search(query, chunks, chunk_metadata, model, top_k=20, min_similarity=0.35, keyword_weight=0.25):
    if not chunks:
        return []
    
    # 1. Смысловой поиск
    query_emb = model.encode([query])
    chunk_embs = model.encode(chunks)
    semantic_scores = cosine_similarity(query_emb, chunk_embs)[0]
    
    # 2. Ключевые слова (TF-IDF)
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
        tfidf_matrix = vectorizer.fit_transform(chunks + [query])
        # Сходство по ключевым словам
        keyword_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    except:
        keyword_scores = np.zeros(len(chunks))
    
    # 3. Комбинированный рейтинг
    combined_scores = (1 - keyword_weight) * semantic_scores + keyword_weight * keyword_scores
    
    # 4. Сортировка
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        if combined_scores[i] >= min_similarity:
            results.append((chunks[i], chunk_metadata[i], combined_scores[i], semantic_scores[i], keyword_scores[i]))
    
    return results

# --- Интерфейс ---

st.sidebar.header("📚 Библиотека нормативов")

docs_folder = Path("./docs")

if not docs_folder.exists():
    st.error("❌ Папка 'docs' не найдена!")
    st.stop()

# Исключаем служебные файлы
txt_files = [f for f in docs_folder.glob("*.txt") if "placeholder" not in f.name.lower()]

if not txt_files:
    st.warning("📁 В папке 'docs' нет TXT-файлов")
    st.stop()

with st.sidebar:
    st.write(f"📄 Всего нормативов: **{len(txt_files)}**")
    for f in txt_files:
        st.write(f"   - {f.name}")

    st.markdown("---")
    st.caption("🧠 Модель: allenai-specter (технические тексты)")
    st.caption("⚙️ Гибридный поиск: смысл + ключевые слова")
    st.caption("🎯 Порог сходства: 35%")

# --- Ввод запроса ---
query = st.text_input(
    "✏️ Введите запрос:",
    placeholder="например: время срабатывания автоматических выключателей",
    label_visibility="collapsed"
)

# --- Кнопка поиска ---
if st.button("🔎 Искать", type="primary") and query:
    if len(query) < 3:
        st.warning("⚠️ Минимум 3 символа")
        st.stop()

    with st.spinner(f"Обрабатываю {len(txt_files)} файлов..."):
        all_chunks = []
        all_metadata = []

        progress = st.progress(0)
        for idx, txt_file in enumerate(txt_files):
            progress.progress((idx + 1) / len(txt_files))

            text = extract_text_from_txt(txt_file)
            if not text:
                continue

            chunks = split_by_sentences(text, max_len=350, overlap=50)

            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'норматив': txt_file.stem.replace('.txt', '').replace('_', ' '),
                    'файл': txt_file.name
                })

        progress.empty()

        if not all_chunks:
            st.error("❌ Не удалось извлечь текст")
            st.stop()

        results = hybrid_search(query, all_chunks, all_metadata, model, top_k=20, min_similarity=0.35)

        if results:
            st.success(f"✅ Найдено **{len(results)}** релевантных фрагментов")
            st.markdown("---")

            for i, (chunk, meta, combined, semantic, keyword) in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1.2])

                    with col1:
                        st.markdown(f"**Результат {i}**")
                        display = chunk[:700] + "..." if len(chunk) > 700 else chunk
                        st.write(display)

                    with col2:
                        st.metric("Сходство", f"{combined*100:.1f}%")
                        st.caption(f"Смысл: {semantic*100:.0f}%")
                        st.caption(f"Слова: {keyword*100:.0f}%")

                    st.caption(f"📌 **Источник:** {meta['норматив']}")
                    st.divider()

            # Экспорт
            results_df = pd.DataFrame([
                {
                    'текст': r[0][:300] + '...' if len(r[0]) > 300 else r[0],
                    'норматив': r[1]['норматив'],
                    'сходство': f"{r[2]*100:.1f}%"
                }
                for r in results
            ])

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"поиск_{query[:20].replace(' ', '_')}.csv",
                mime="text/csv"
            )

        else:
            st.warning(f"😕 Ничего не найдено по запросу: **{query}**")

# --- Инструкция ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 Как повысить точность")
    st.markdown("""
    - **Формулируйте запрос конкретно**  
      ✅ «время срабатывания автоматического выключателя при перегрузке»  
      ❌ «автомат»
    - **Используйте термины из нормативов**
    - **Проверьте качество TXT** (не должно быть мусора)
    - **Добавьте больше файлов** — модель лучше ищет по большему объёму
    """)