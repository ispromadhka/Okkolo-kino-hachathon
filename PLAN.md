# Video RAG — План работы на хакатон

## Задание

Система семантического поиска по видео. По текстовому запросу пользователя — найти точные временные отрезки в видео, где содержится ответ.

### Входные данные
- ~80 GB видеофайлов (MP4, WebM, MKV), каждое видео 2-3 минуты
- Аудиодорожки (MP3, Opus)
- Транскрипции от Whisper Tiny (низкое качество, особенно на русском)
- Табличные данные с разметкой: (запрос, релевантные фрагменты видео, ответ)

### Формат submission
```
query_id, video_file_1, start_1, end_1, video_file_2, start_2, end_2, ..., video_file_5, start_5, end_5
```
На каждый запрос — ровно 5 предсказаний (video_file + start + end), по убыванию релевантности.

### Метрики
- **SuccessRate@K (SR@K)** — среди top-K предсказаний есть хотя бы одно с правильным видео И IoU ≥ 0.5
- **VideoRecall@K (VR@K)** — среди top-K есть правильное видео (без учёта таймкодов)
- **FinalScore = (AvgSR + AvgVR) / 2**, где Avg = среднее по K=1,3,5

### Требования
- Мультиязычность: русский + английский
- Устойчивость к опечаткам и нечётким формулировкам
- Масштабируемость: большой объём видео
- Latency < 1 секунда на запрос

---

## Два контура

### Контур 1: Kaggle (70% времени) — максимизация FinalScore
### Контур 2: Бизнес-демо (30% времени) — Streamlit UI для жюри

---

## Архитектура

### OFFLINE — индексация (без лимита по времени, на DGX)

```
Для каждого видео (2-3 мин):

  1. PySceneDetect → границы сцен (5-20 сцен по 10-30 сек)

  2. FFmpeg → аудио (WAV 16kHz) + ключевые кадры (3-5 на сцену)

  3. GigaAM-v3 → транскрипт с пунктуацией и таймкодами
     (замена Whisper Tiny: WER 2-3% vs 30-40%)

  4. Qwen3-VL-4B → описание кадров на русском
     (визуальный контент: кто, что делает, обстановка, эмоции)

  5. EasyOCR → текст с экрана (титры, вывески, письма)

  6. Три уровня чанков:
     - Sentence-level: каждое предложение из ASR с точными таймкодами
       (максимальная точность для IoU)
     - Scene-level: PySceneDetect сцена + VLM описание + OCR
       (покрывает визуальные запросы)
     - Video-level: summary всего видео
       (гарантирует VR@K — 50% итогового скора)

  7. BGE-M3 → dense (1024d) + sparse vectors

  8. Qdrant → хранение с метаданными
     (video_file, start_time, end_time, chunk_type, text)
```

### ONLINE — поиск (<1 секунда)

```
Запрос (RU/EN)
  → BGE-M3 encode                           (~80ms)
  → Qdrant hybrid search (dense+sparse+RRF) (~30ms)
    → top-20 кандидатов
  → FlashRank reranker (ONNX, CPU)           (~150ms)
    → top-5
  → Post-processing:
    - Группировка по video_file
    - Merge overlapping segments
    - Window padding ±3-5 сек (для IoU)
  → Return: 5 × (video_file, start, end)
  ────────────────────────────────────────
  Итого: ~300-400ms ✓
```

**Никаких LLM при query-time.** Только embed → search → rerank.

---

## Стратегии для максимизации скора

### SR@K (правильное видео + IoU ≥ 0.5)

1. **Sentence-level chunking** — самая точная привязка к таймкодам. Если запрос цитирует диалог, sentence-level чанк даст точное совпадение.

2. **Window padding ±3-5 сек** — расширение предсказанного сегмента повышает IoU с ground truth.

3. **Merge overlapping** — если два чанка из одного видео рядом (например, sentence 5 и sentence 6), объединяем в один сегмент.

### VR@K (правильное видео, без таймкодов = 50% скора)

1. **Video-level чанки** — один summary на всё видео. Гарантирует, что правильное видео попадёт в top-5.

2. **Diversification** — если top-5 из одного видео, разбавить другими видео для VR@K.

### Общее

1. **GigaAM перетранскрибирование** — самый большой single improvement. Whisper Tiny на русском теряет треть слов.

2. **Contextual enrichment** — для 2-3 мин видео: LLM генерирует одно общее описание содержания. Каждая сцена получает контекст.

3. **Query expansion** (если останется время) — перевод запроса RU↔EN, переформулирование. Поиск по всем вариантам.

---

## Компоненты

| Компонент | Технология | Latency budget | Когда работает |
|-----------|-----------|---------------|---------------|
| Сегментация | PySceneDetect | — | Offline |
| ASR (русский) | GigaAM-v3 (240M) | — | Offline |
| ASR (английский) | faster-whisper large-v3-turbo | — | Offline |
| VLM описания | Qwen3-VL-4B | — | Offline |
| OCR | EasyOCR | — | Offline |
| Enrichment | Qwen3-14B | — | Offline |
| Эмбеддинги | BGE-M3 (1024d, dense+sparse) | ~80ms | Online (query) + Offline (index) |
| Векторная БД | Qdrant (in-memory, hybrid) | ~30ms | Online |
| Реранкер | FlashRank (ONNX, CPU) | ~150ms | Online |
| LLM ответ | Qwen3-14B | — | Только бизнес-демо |
| UI | Streamlit | — | Только бизнес-демо |

**FlashRank вместо bge-reranker-v2-m3** для online — ONNX на CPU, ультралёгкий, гарантированно укладывается в <1s.

---

## Дорожная карта

### Час 0-2: Setup
- [ ] Скачать данные (80GB) на DGX
- [ ] Развернуть venv, установить зависимости
- [ ] Скачать модели: GigaAM-v3, Qwen3-VL-4B, BGE-M3, FlashRank
- [ ] Изучить формат данных: таблица разметки, формат транскрипций

### Час 2-5: Перетранскрибирование + сегментация
- [ ] GigaAM-v3 перетранскрибировать все русские видео
- [ ] faster-whisper large-v3-turbo для английских видео
- [ ] PySceneDetect для всех видео → границы сцен
- [ ] FFmpeg → ключевые кадры (3-5 на сцену)

### Час 5-8: VLM описания + индексация
- [ ] Qwen3-VL-4B → описания кадров (параллельно на GPU)
- [ ] EasyOCR → текст с экрана
- [ ] Chunking: sentence-level + scene-level + video-level
- [ ] BGE-M3 эмбеддинги → Qdrant

### Час 8-10: Search pipeline + первый submit
- [ ] Qdrant hybrid search (dense + sparse + RRF)
- [ ] FlashRank reranker
- [ ] Post-processing: grouping, merging, padding
- [ ] Генерация submission файла
- [ ] **Первый Kaggle submit**

### Час 10-14: Итерации на скор
- [ ] Валидация на train данных: считаем SR@K, VR@K локально
- [ ] Подбор параметров: PySceneDetect threshold, padding window, RRF weights
- [ ] Добавить contextual enrichment (video-level summaries)
- [ ] Query expansion (RU↔EN перевод)
- [ ] **Улучшенные Kaggle submits**

### Час 14-18: Бизнес-демо
- [ ] Streamlit UI: поле запроса + видеоплеер с start/end
- [ ] Qwen3-14B: LLM объяснения "почему эта сцена"
- [ ] Multi-video selector
- [ ] Thumbnail previews из keyframes
- [ ] Query Router (тип запроса → стратегия)

### Час 18-20: Финал
- [ ] Финальный Kaggle submit
- [ ] Презентация: архитектура + метрики + live demo
- [ ] Подготовить "wow" запросы

---

## Структура кода

```
video-rag/
├── config.py                     # Все параметры
├── ingest.py                     # CLI: обработка всех видео
├── search.py                     # CLI: поиск по запросу
├── submit.py                     # Генерация Kaggle submission
├── evaluate.py                   # Локальный подсчёт SR@K, VR@K
│
├── pipeline/
│   ├── scene_detector.py         # PySceneDetect
│   ├── frame_extractor.py        # FFmpeg → keyframes
│   ├── transcriber.py            # GigaAM-v3 / faster-whisper
│   ├── frame_describer.py        # Qwen3-VL-4B
│   ├── ocr_extractor.py          # EasyOCR
│   ├── enricher.py               # Qwen3-14B summaries
│   ├── chunker.py                # Sentence/Scene/Video level chunks
│   └── indexer.py                # BGE-M3 → Qdrant
│
├── search/
│   ├── retriever.py              # Qdrant hybrid + FlashRank
│   └── postprocessor.py          # Merge, padding, dedup, top-5
│
├── app.py                        # Streamlit UI (бизнес-демо)
└── requirements.txt
```

---

## Позиционирование для жюри

**Нарратив**: "Мы набрали top-N на Kaggle, затем обернули тот же движок в production-ready интерфейс. Система понимает видео на трёх уровнях: диалоги (GigaAM), визуал (VLM), текст на экране (OCR). Гибридный поиск + реранкинг укладывается в <1 секунду."

**Демо-сценарий:**
1. Диалоговый запрос → sentence-level точность
2. Визуальный запрос → VLM описание сработало
3. Запрос с опечаткой → sparse search (BM25) + dense (семантика) покрывают
4. Kaggle метрики: "наш FinalScore = X"
5. Ablation: transcript-only → +VLM → +enrichment → +reranker
