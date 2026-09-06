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

# Загружаем модель (кешируется для быстрой работы)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Функция для извлечения текста из TXT
def extract_text_from_txt(txt_path):
    try:
        # Пробуем разные кодировки (для русских букв)
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
        st.error(f"⚠️ Ошибка при чтении {txt_path.name}: {e}")
        return ""

# Функция для разбивки текста на фрагменты (чанки)
def split_into_chunks(text, chunk_size=500, overlap=100):
    """Разбивает текст на перекрывающиеся фрагменты для лучшего поиска"""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Функция для семантического поиска
def search_semantic(query, chunks, chunk_metadata, model, top_k=10):
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
    
    # Формируем результаты с порогом схожести 0.3 (30%)
    results = []
    for i in top_indices:
        if similarities[i] > 0.3:
            results.append((chunks[i], chunk_metadata[i], similarities[i]))
    
    return results

# --- Интерфейс приложения ---

# Боковая панель с информацией
st.sidebar.header("📚 Библиотека нормативов")

# Путь к папке с документами (в репозитории)
docs_folder = Path("./docs")

# Проверяем наличие папки docs
if not docs_folder.exists():
    st.error("❌ Папка 'docs' не найдена!")
    st.info("📖 Создайте папку 'docs' в репозитории и добавьте TXT-файлы с нормативами.")
    st.stop()

# Получаем список TXT-файлов
txt_files = list(docs_folder.glob("*.txt"))

if not txt_files:
    st.warning("📁 В папке 'docs' нет TXT-файлов")
    st.info("📤 Загрузите TXT-файлы через GitHub в папку 'docs' и обновите страницу.")
    
    # Инструкция
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
        file_size = f.stat().st_size // 1024  # размер в КБ
        st.write(f"   - {f.name} ({file_size} КБ)")
    
    st.markdown("---")
    st.caption("💡 Чтобы добавить новый норматив, загрузите TXT-файл в папку 'docs' на GitHub")
    st.caption("⚙️ Используется формат TXT (самый надежный)")

# Поле для поискового запроса
st.markdown("### ✏️ Введите ваш запрос")
query = st.text_input(
    "Поисковый запрос:",
    placeholder="например: сечение проводников заземления",
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
            
            # Разбиваем на чанки
            chunks = split_into_chunks(text)
            
            # Сохраняем с метаданными
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    'норматив': txt_file.stem,
                    'файл': txt_file.name
                })
        
        progress_bar.empty()
        
        # Если не удалось извлечь текст
        if failed_files:
            st.warning(f"⚠️ Не удалось прочитать {len(failed_files)} файлов: {', '.join(failed_files[:3])}")
        
        # Если нет текста для поиска
        if not all_chunks:
            st.error("❌ Не удалось извлечь текст из файлов. Проверьте формат TXT.")
            st.stop()
        
        # Выполняем семантический поиск
        results = search_semantic(query, all_chunks, all_metadata, model)
        
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
                        # Показываем первые 500 символов
                        display_text = chunk[:500] + "..." if len(chunk) > 500 else chunk
                        st.write(display_text)
                    with col2:
                        # Показываем процент сходства
                        score_percent = f"{score * 100:.1f}%"
                        st.metric("Сходство", score_percent)
                    
                    # Информация об источнике
                    st.caption(f"📌 **Источник:** {meta['норматив']} (файл: {meta['файл']})")
                    st.divider()
            
            # Кнопка для скачивания результатов в CSV
            results_df = pd.DataFrame([
                {
                    'текст': r[0][:200] + '...' if len(r[0]) > 200 else r[0],
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
            st.info("💡 Совет: попробуйте использовать более общие термины (например, 'заземление', 'сечение', 'проводник')")

elif search_button and not query:
    st.warning("⚠️ Введите поисковый запрос")

# Информация в боковой панели
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 Как это работает")
    st.markdown("""
    1. Конвертируйте RTF в TXT через WordPad
    2. Загрузите TXT-файлы в папку `docs` на GitHub
    3. Введите запрос на русском языке
    4. Приложение находит фрагменты по **смыслу**, а не по словам
    5. Результаты показывают процент сходства с запросом
    6. Можно скачать результаты в CSV
    """)
    
    st.markdown("---")
    st.caption("🔒 Данные хранятся только в вашем репозитории GitHub")
    st.caption("🧠 Модель: all-MiniLM-L6-v2 (бесплатно, локально)")