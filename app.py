import streamlit as st
import os
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Настройка страницы
st.set_page_config(page_title="Смысловой поиск по нормативам", layout="wide")
st.title("🧠 Смысловой поиск по нормативной документации")
st.markdown("**Ищет по смыслу, а не по точным словам**")
st.markdown("---")

# Загружаем модель (многозычная, оптимизирована для русского)
@st.cache_resource
def load_model():
    # Используем модель, специально обученную на русском языке
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

# Функция для извлечения текста из TXT
def extract_text_from_txt(txt_path):
    try:
        # Пробуем разные кодировки
        for encoding in ['utf-8', 'windows-1251', 'cp1251', 'koi8-r']:
            try:
                with open(txt_path, 'r', encoding=encoding) as f:
                    text = f.read()
                # Удаляем лишние пробелы и переносы
                text = re.sub(r'\s+', ' ', text).strip()
                return text
            except UnicodeDecodeError:
                continue
        return ""
    except Exception as e:
        return ""

# Функция для умной разбивки текста на фрагменты
def split_into_chunks(text, chunk_size=300, overlap=50):
    """
    Разбивает текст на перекрывающиеся фрагменты.
    chunk_size - размер фрагмента в словах
    overlap - перекрытие между фрагментами
    """
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Функция для семантического поиска с улучшенной фильтрацией
def search_semantic(query, chunks, chunk_metadata, model, top_k=15, min_similarity=0.4):
    if not chunks:
        return []
    
    # Создаем эмбеддинг для запроса
    query_embedding = model.encode([query])
    
    # Создаем эмбеддинги для всех фрагментов
    chunk_embeddings = model.encode(chunks)
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Получаем индексы топ-k наиболее похожих фрагментов
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Формируем результаты с порогом схожести
    results = []
    for i in top_indices:
        if similarities[i] >= min_similarity:
            results.append((chunks[i], chunk_metadata[i], similarities[i]))
    
    return results

# --- Интерфейс приложения ---

# Боковая панель с информацией
st.sidebar.header("📚 Библиотека нормативов")

# Путь к папке с документами
docs_folder = Path("./docs")

# Проверяем наличие папки docs
if not docs_folder.exists():
    st.error("❌ Папка 'docs' не найдена!")
    st.info("📖 Создайте папку 'docs' в репозитории и добавьте TXT-файлы с нормативами.")
    st.stop()

# Получаем список TXT-файлов (исключаем placeholder)
txt_files = [f for f in docs_folder.glob("*.txt") if "placeholder" not in f.name.lower()]

if not txt_files:
    st.warning("📁 В папке 'docs' нет TXT-файлов с нормативами")
    st.info("📤 Загрузите TXT-файлы через GitHub в папку 'docs' и обновите страницу.")
    
    with st.expander("📖 Как добавить нормативы"):
        st.markdown("""
        1. Конвертируйте RTF-файлы в TXT через WordPad
        2. Зайдите на **GitHub** в ваш репозиторий
        3. Нажмите **'Add file' → 'Upload files'**
        4. Выберите папку **'docs'**
        5. Перетащите TXT-файлы с нормативами
        6. Нажмите **'Commit changes'**
        7. Обновите эту страницу
        """)
    st.stop()

# Отображаем список загруженных файлов
with st.sidebar:
    st.write(f"📄 Всего нормативов: **{len(txt_files)}**")
    st.markdown("**Файлы:**")
    for f in txt_files:
        file_size = f.stat().st_size // 1024
        st.write(f"   - {f.name} ({file_size} КБ)")
    
    st.markdown("---")
    st.caption("💡 Используется улучшенная модель для русского языка")
    st.caption("⚙️ Минимальное сходство: 40%")

# Поле для поискового запроса
st.markdown("### ✏️ Введите ваш запрос")
query = st.text_input(
    "Поисковый запрос:",
    placeholder="например: время срабатывания автоматических выключателей",
    label_visibility="collapsed"
)

# Кнопка поиска
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔎 Искать по смыслу", type="primary", use_container_width=True)

# Обработка поиска
if search_button and query:
    if len(query) < 3:
        st.warning("⚠️ Введите минимум 3 символа для поиска")
        st.stop()
    
    with st.spinner(f"🧠 Обрабатываю {len(txt_files)} нормативов..."):
        all_chunks = []
        all_metadata = []
        failed_files = []
        
        # Прогресс-бар
        progress_bar = st.progress(0)
        
        # Обрабатываем каждый файл
        for idx, txt_file in enumerate(txt_files):
            progress_bar.progress((idx + 1) / len(txt_files))
            
            # Извлекаем текст
            text = extract_text_from_txt(txt_file)
            if not text:
                failed_files.append(txt_file.name)
                continue
            
            # Разбиваем на умные фрагменты
            chunks = split_into_chunks(text, chunk_size=300, overlap=50)
            
            # Сохраняем с метаданными
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'норматив': txt_file.stem.replace('.txt', '').replace('_', ' '),
                    'файл': txt_file.name
                })
        
        progress_bar.empty()
        
        # Если не удалось извлечь текст
        if failed_files:
            st.warning(f"⚠️ Не удалось прочитать {len(failed_files)} файлов")
        
        # Если нет текста для поиска
        if not all_chunks:
            st.error("❌ Не удалось извлечь текст из файлов. Проверьте формат TXT.")
            st.stop()
        
        # Выполняем семантический поиск
        results = search_semantic(query, all_chunks, all_metadata, model, top_k=15, min_similarity=0.4)
        
        # Вывод результатов
        if results:
            st.success(f"✅ Найдено **{len(results)}** релевантных фрагментов")
            st.markdown("---")
            
            # Показываем результаты
            for i, (chunk, meta, score) in enumerate(results, 1):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Результат {i}**")
                        # Показываем фрагмент с контекстом
                        display_text = chunk
                        if len(display_text) > 600:
                            display_text = display_text[:600] + "..."
                        st.write(display_text)
                    with col2:
                        score_percent = f"{score * 100:.1f}%"
                        st.metric("Сходство", score_percent)
                    
                    # Информация об источнике
                    st.caption(f"📌 **Источник:** {meta['норматив']}")
                    st.divider()
            
            # Кнопка для скачивания результатов в CSV
            results_df = pd.DataFrame([
                {
                    'текст': r[0][:300] + '...' if len(r[0]) > 300 else r[0],
                    'норматив': r[1]['норматив'],
                    'файл': r[1]['файл'],
                    'сходство': f"{r[2]*100:.1f}%"
                }
                for r in results
            ])
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать результаты в CSV",
                data=csv,
                file_name=f"результаты_поиска_{query[:20].replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.warning(f"😕 Ничего не найдено по запросу **'{query}'**")
            st.info("💡 Совет: попробуйте использовать более общие термины или проверьте, есть ли в нормативах нужная информация")

elif search_button and not query:
    st.warning("⚠️ Введите поисковый запрос")

# Информация в боковой панели
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Как улучшить точность поиска")
    st.markdown("""
    1. **Используйте конкретные термины** (например, 'время срабатывания автоматического выключателя')
    2. **Проверьте качество TXT** — если в файле много мусора, пересохраните через Word
    3. **Добавьте больше нормативов** — чем больше данных, тем точнее поиск
    4. **Удалите служебные файлы** (placeholder.txt) из папки docs
    """)
    
    st.markdown("---")
    st.caption("🔒 Данные хранятся только в вашем репозитории GitHub")
    st.caption("🧠 Модель: paraphrase-multilingual-MiniLM-L12-v2")