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

# --- Настройка страницы ---
st.set_page_config(page_title="Точный поиск по нормативам", layout="wide")
st.title("🎯 Максимально точный поиск по нормативам")
st.markdown("**Гибридный поиск: смысл + ключевые слова + синонимы + приоритет нормативов**")
st.markdown("---")

# --- Словарь синонимов для расширения запроса ---
SYNONYMS = {
    'автомат': ['автоматический выключатель', 'защитный автомат', 'АВ', 'автоматический аппарат'],
    'заземление': ['защитное заземление', 'заземляющее устройство', 'контур заземления', 'заземлитель'],
    'зануление': ['нулевой провод', 'PEN', 'защитный нулевой проводник'],
    'УЗО': ['устройство защитного отключения', 'дифференциальный автомат', 'дифавтомат'],
    'сечение': ['площадь поперечного сечения', 'жила', 'проводник', 'кабель'],
    'проводник': ['жила', 'кабель', 'провод', 'токоведущая часть'],
    'срабатывание': ['отключение', 'выключение', 'защита', 'реакция'],
    'время': ['длительность', 'период', 'интервал', 'секунды'],
    'сопротивление': ['омическое сопротивление', 'импеданс', 'R'],
    'ток': ['сила тока', 'амперы', 'I', 'токовая нагрузка'],
    'напряжение': ['вольты', 'U', 'потенциал'],
    'молниезащита': ['защита от молнии', 'громоотвод', 'молниеприемник', 'токоотвод'],
    'ПУЭ': ['правила устройства электроустановок', 'пуэ', 'Правила устройства'],
    'ГОСТ': ['государственный стандарт', 'гост'],
    'СО': ['свод правил', 'СП', 'инструкция'],
}

# --- Приоритет нормативов (от更高 к низшему) ---
PRIORITY_ORDER = {
    'ПУЭ': 10,
    'ГОСТ': 9,
    'СО': 8,
    'СП': 8,
    'Инструкция': 7,
    'Правила': 6,
}

# --- Стоп-слова (не влияют на поиск) ---
STOP_WORDS = set([
    'и', 'в', 'на', 'с', 'по', 'к', 'у', 'о', 'от', 'до', 'из', 'за', 'через',
    'при', 'для', 'без', 'под', 'над', 'об', 'про', 'же', 'бы', 'да', 'нет',
    'так', 'как', 'что', 'это', 'все', 'всё', 'или', 'если', 'то', 'но', 'а',
    'его', 'её', 'ее', 'их', 'еще', 'уже', 'еще', 'ведь', 'вот', 'лишь',
])

# --- Загрузка модели (специально для техтекстов) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/allenai-specter')

model = load_model()

# --- Кэширование эмбеддингов ---
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

# --- Расширение запроса синонимами ---
def expand_query(query):
    words = query.lower().split()
    expanded = [query]
    
    # Заменяем слова на синонимы
    for word in words:
        word_clean = re.sub(r'[^а-яa-z]', '', word)
        if word_clean in SYNONYMS:
            for syn in SYNONYMS[word_clean]:
                if syn not in query.lower():
                    expanded.append(syn)
    
    # Если есть номер пункта (п. 1.7.126) — сохраняем как есть
    point_match = re.search(r'[пп]\.\s*[\d\.]+', query)
    if point_match:
        expanded.append(point_match.group())
    
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

# --- Умная разбивка на фрагменты ---
def split_by_sentences(text, max_len=300, overlap=50):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sent in sentences:
        sent_len = len(sent.split())
        if current_len + sent_len > max_len and current_chunk:
            chunks.append(' '.join(current_chunk))
            overlap_words = ' '.join(current_chunk).split()[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + [sent]
            current_len = len(overlap_words) + sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# --- Определение приоритета норматива ---
def get_priority(filename):
    for key, value in PRIORITY_ORDER.items():
        if key in filename:
            return value
    return 5

# --- Гибридный поиск с синонимами и приоритетом ---
def advanced_search(query, chunks, chunk_metadata, model, top_k=25, min_similarity=0.3):
    if not chunks:
        return []
    
    # 1. Расширяем запрос
    expanded_query = expand_query(query)
    
    # 2. Смысловой поиск
    query_emb = get_embedding(expanded_query)
    chunk_embs = np.array([get_embedding(c)[0] for c in chunks])
    semantic_scores = cosine_similarity(query_emb, chunk_embs)[0]
    
    # 3. Ключевые слова (TF-IDF)
    try:
        vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS), max_features=150)
        tfidf_matrix = vectorizer.fit_transform(chunks + [query])
        keyword_scores = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])[0]
    except:
        keyword_scores = np.zeros(len(chunks))
    
    # 4. Нормализация по длине (короткие точные фрагменты в приоритете)
    length_scores = np.array([1.0 / (1 + len(c.split()) / 100) for c in chunks])
    
    # 5. Приоритет норматива
    priority_scores = np.array([get_priority(meta['норматив']) / 10.0 for meta in chunk_metadata])
    
    # 6. Комбинированный рейтинг
    combined_scores = (
        0.45 * semantic_scores +
        0.25 * keyword_scores +
        0.15 * length_scores +
        0.15 * priority_scores
    )
    
    # 7. Сортировка
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = []
    for i in top_indices:
        if combined_scores[i] >= min_similarity:
            # Находим ключевые слова, которые совпали
            chunk_words = set(re.findall(r'[а-яa-z0-9]{3,}', chunks[i].lower()))
            query_words = set(re.findall(r'[а-яa-z0-9]{3,}', query.lower()))
            matched_words = chunk_words & query_words
            
            results.append({
                'text': chunks[i],
                'metadata': chunk_metadata[i],
                'score': combined_scores[i],
                'semantic': semantic_scores[i],
                'keyword': keyword_scores[i],
                'priority': priority_scores[i],
                'matched_words': matched_words,
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
    for f in txt_files:
        st.write(f"   - {f.name}")

    st.markdown("---")
    st.caption("🧠 Модель: allenai-specter (технические тексты)")
    st.caption("🔍 Гибридный поиск: смысл + слова + приоритет")
    st.caption("🔄 Авторасширение запроса синонимами")
    st.caption("📊 Приоритет: ПУЭ > ГОСТ > СО > СП > Инструкции")
    st.caption("🎯 Порог сходства: 30%")

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

            chunks = split_by_sentences(text, max_len=300, overlap=50)

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

        results = advanced_search(query, all_chunks, all_metadata, model)

        if results:
            st.success(f"✅ Найдено **{len(results)}** релевантных фрагментов")
            
            # Показываем расширенный запрос
            expanded = expand_query(query)
            if expanded != query:
                with st.expander("🔍 Расширенный запрос (синонимы)"):
                    st.write(f"Было: **{query}**")
                    st.write(f"Стало: **{expanded}**")
            
            st.markdown("---")

            for i, r in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1.2])

                    with col1:
                        st.markdown(f"**Результат {i}**")
                        display = r['text'][:700] + "..." if len(r['text']) > 700 else r['text']
                        st.write(display)
                        
                        # Показываем совпавшие ключевые слова
                        if r['matched_words']:
                            st.caption(f"🔑 Совпавшие термины: {', '.join(list(r['matched_words'])[:5])}")

                    with col2:
                        st.metric("Сходство", f"{r['score']*100:.1f}%")
                        st.caption(f"Смысл: {r['semantic']*100:.0f}%")
                        st.caption(f"Слова: {r['keyword']*100:.0f}%")
                        priority_label = "⭐" if r['priority'] > 0.7 else ""
                        st.caption(f"Приоритет: {priority_label} {r['priority']*100:.0f}%")

                    st.caption(f"📌 **Источник:** {r['metadata']['норматив']}")
                    st.divider()

            # Экспорт
            results_df = pd.DataFrame([
                {
                    'текст': r['text'][:300] + '...' if len(r['text']) > 300 else r['text'],
                    'норматив': r['metadata']['норматив'],
                    'сходство': f"{r['score']*100:.1f}%"
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
            st.info("💡 Совет: попробуйте использовать термины из нормативов: 'автоматический выключатель' вместо 'автомат'")

# --- Инструкция ---
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📈 Советы для точного поиска")
    st.markdown("""
    - **Используйте термины из нормативов**  
      ✅ «автоматический выключатель»  
      ❌ «автомат»
    - **Указывайте номер пункта**  
      ✅ «п. 1.7.126 заземление»
    - **Добавляйте норматив**  
      ✅ «ПУЭ сечение проводника»
    - **Не используйте стоп-слова**  
      («что», «как», «где», «почему»)
    """)