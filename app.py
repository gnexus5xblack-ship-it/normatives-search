import streamlit as st
import os
from pathlib import Path
import pandas as pd
import numpy as np
import re
import pickle
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Настройка ---
st.set_page_config(page_title="Точный поиск по нормативам", layout="wide")
st.title("🎯 Точный поиск по нормативам")
st.markdown("**Гибридный поиск: смысл + ключевые слова + фильтрация шума**")
st.markdown("---")

# --- Словарь синонимов ---
SYNONYMS = {
    'автомат': ['автоматический выключатель', 'АВ', 'защитный аппарат'],
    'заземление': ['заземляющее устройство', 'заземлитель', 'контур заземления'],
    'сечение': ['площадь поперечного сечения', 'жила', 'проводник'],
    'срабатывание': ['отключение', 'выключение', 'защита'],
    'сопротивление': ['омическое сопротивление', 'импеданс'],
    'ток': ['сила тока', 'амперы', 'токовая нагрузка'],
    'зануление': ['нулевой провод', 'PEN'],
    'УЗО': ['устройство защитного отключения'],
}

# --- Стоп-слова (полный список) ---
STOP_WORDS = set([
    'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'от', 'до', 'из', 'за', 'через',
    'при', 'для', 'без', 'под', 'над', 'об', 'про', 'же', 'бы', 'да', 'нет',
    'так', 'как', 'что', 'это', 'все', 'всё', 'или', 'если', 'то', 'но', 'а',
    'его', 'её', 'ее', 'их', 'еще', 'уже', 'ведь', 'вот', 'лишь', 'очень',
    'можно', 'нужно', 'должен', 'должна', 'должны', 'быть',
    'более', 'менее', 'также', 'кроме', 'который', 'которая', 'которое',
    'этом', 'этом', 'этим', 'этой', 'этого', 'этих', 'этим',
])

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# --- Кэш эмбеддингов ---
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def get_embedding(text):
    cache_file = CACHE_DIR / f"{get_cache_key(text)}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    emb = model.encode([text])
    with open(cache_file, 'wb') as f:
        pickle.dump(emb, f)
    return emb

# --- Расширение запроса ---
def expand_query(query):
    words = re.findall(r'[а-яa-z0-9]{3,}', query.lower())
    expanded = [query]
    
    # Синонимы
    for word in words:
        if word in SYNONYMS:
            for syn in SYNONYMS[word]:
                if syn not in query.lower():
                    expanded.append(syn)
    
    return ' '.join(expanded)

# --- Очистка текста ---
def clean_text(text):
    text = re.sub(r'Стр\.\s*\d+', '', text)
    text = re.sub(r'Страница\s*\d+', '', text)
    text = re.sub(r'\d+\s*из\s*\d+', '', text)
    text = re.sub(r'[\\/*?:"<>|]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Извлечение текста ---
def extract_text_from_txt(txt_path):
    try:
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'koi8-r']:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text = f.read()
                return clean_text(text)
            except UnicodeDecodeError:
                continue
        return ""
    except Exception:
        return ""

# --- Разбивка на короткие точные фрагменты ---
def split_by_sentences(text, max_len=200, overlap=30):
    """Разбивает на маленькие фрагменты для точности"""
    # Разбиваем по предложениям
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_len and current:
            chunks.append(' '.join(current))
            # Перекрытие только 30 слов
            overlap_words = ' '.join(current).split()[-overlap:] if overlap > 0 else []
            current = overlap_words + [sent]
            current_len = len(overlap_words) + sent_len
        else:
            current.append(sent)
            current_len += sent_len
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks

# --- Фильтрация фрагментов (отсекаем мусор) ---
def filter_chunks(chunks, metadata):
    """Удаляет слишком короткие, слишком длинные, служебные фрагменты"""
    filtered = []
    for chunk, meta in zip(chunks, metadata):
        words = chunk.split()
        word_count = len(words)
        
        # Минимум 10 слов, максимум 250
        if word_count < 10 or word_count > 250:
            continue
        
        # Удаляем фрагменты с подозрительными маркерами
        if 'таблица' in chunk.lower() and len(words) < 15:
            continue
        
        # Удаляем фрагменты с большим количеством цифр (оглавления)
        digit_ratio = len(re.findall(r'\d', chunk)) / len(chunk)
        if digit_ratio > 0.2:  # больше 20% цифр
            continue
        
        filtered.append((chunk, meta))
    
    return filtered

# --- Основной поиск (с жёсткой фильтрацией) ---
def advanced_search(query, chunks, chunk_metadata, model, top_k=15, min_similarity=0.5):
    if not chunks:
        return []
    
    # 1. Расширяем запрос
    expanded_query = expand_query(query)
    
    # 2. Смысловой поиск
    query_emb = get_embedding(expanded_query)
    chunk_embs = np.array([get_embedding(c)[0] for c in chunks])
    semantic_scores = cosine_similarity(query_emb, chunk_embs)[0]
    
    # 3. Ключевые слова (строгий поиск терминов)
    # Извлекаем ключевые термины из запроса
    query_terms = set(re.findall(r'[а-яa-z]{4,}', query.lower()))
    query_terms = query_terms - STOP_WORDS
    
    keyword_scores = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'[а-яa-z]{4,}', chunk.lower()))
        # Считаем пересечение ключевых терминов
        overlap = len(query_terms & chunk_words)
        if query_terms:
            keyword_scores.append(overlap / len(query_terms))
        else:
            keyword_scores.append(0)
    keyword_scores = np.array(keyword_scores)
    
    # 4. Комбинированный рейтинг (смещение в сторону ключевых слов)
    combined = 0.4 * semantic_scores + 0.6 * keyword_scores
    
    # 5. Фильтр по количеству совпавших терминов
    min_term_matches = max(1, len(query_terms) // 2)
    
    # 6. Сортировка и фильтрация
    top_indices = np.argsort(combined)[::-1]
    
    results = []
    for i in top_indices:
        if len(results) >= top_k:
            break
        
        # Проверяем порог сходства
        if combined[i] < min_similarity:
            continue
        
        # Проверяем, что есть хотя бы один ключевой термин
        chunk_words = set(re.findall(r'[а-яa-z]{4,}', chunks[i].lower()))
        if not (query_terms & chunk_words):
            continue
        
        results.append({
            'text': chunks[i],
            'metadata': chunk_metadata[i],
            'score': combined[i],
            'semantic': semantic_scores[i],
            'keyword': keyword_scores[i],
            'matched_terms': query_terms & chunk_words,
        })
    
    return results

# --- Интерфейс ---

st.sidebar.header("📚 Библиотека нормативов")

docs_folder = Path("./docs")

if not docs_folder.exists():
    st.error("❌ Папка 'docs' не найдена!")
    st.stop()

txt_files = [f for f in docs_folder.glob("*.txt") if "placeholder" not in f.name.lower()]

if not txt_files:
    st.warning("📁 В папке 'docs' нет TXT-файлов")
    st.stop()

with st.sidebar:
    st.write(f"📄 Всего нормативов: **{len(txt_files)}**")
    for f in txt_files[:10]:
        st.write(f"   - {f.name}")
    if len(txt_files) > 10:
        st.write(f"   ... и еще {len(txt_files) - 10}")

    st.markdown("---")
    st.caption("🎯 Порог сходства: **50%**")
    st.caption("📏 Фрагменты: **до 200 слов**")
    st.caption("🔑 Приоритет: **ключевые термины**")
    st.caption("🚫 Фильтрация: удаляет короткие/длинные/цифровые фрагменты")

# --- Ввод запроса ---
query = st.text_input(
    "✏️ Введите запрос:",
    placeholder="например: сечение заземляющего проводника по ПУЭ",
    label_visibility="collapsed"
)

# --- Поиск ---
if st.button("🔎 Искать", type="primary") and query:
    if len(query) < 4:
        st.warning("⚠️ Введите минимум 4 символа")
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

            chunks = split_by_sentences(text, max_len=200, overlap=30)
            
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

        # Фильтруем фрагменты
        filtered = filter_chunks(all_chunks, all_metadata)
        if filtered:
            all_chunks, all_metadata = zip(*filtered)
            all_chunks = list(all_chunks)
            all_metadata = list(all_metadata)
        else:
            st.error("❌ После фильтрации не осталось фрагментов")
            st.stop()

        results = advanced_search(query, all_chunks, all_metadata, model, top_k=12, min_similarity=0.5)

        if results:
            st.success(f"✅ Найдено **{len(results)}** точных фрагментов")

            # Показываем расширенный запрос
            expanded = expand_query(query)
            if expanded != query:
                with st.expander("🔍 Расширенный запрос"):
                    st.write(f"**Было:** {query}")
                    st.write(f"**Стало:** {expanded}")

            st.markdown("---")

            for i, r in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.markdown(f"**Результат {i}**")
                        display = r['text'][:600] + "..." if len(r['text']) > 600 else r['text']
                        st.write(display)
                        
                        if r['matched_terms']:
                            st.caption(f"🔑 Найдены термины: {', '.join(list(r['matched_terms'])[:5])}")

                    with col2:
                        st.metric("Точность", f"{r['score']*100:.1f}%")
                        st.caption(f"Смысл: {r['semantic']*100:.0f}%")
                        st.caption(f"Термины: {r['keyword']*100:.0f}%")

                    st.caption(f"📌 **Источник:** {r['metadata']['норматив']}")
                    st.divider()

        else:
            st.warning(f"😕 Ничего не найдено по запросу: **{query}**")
            st.info("💡 Совет: попробуйте конкретный термин, например 'сечение заземлителя ПУЭ'")