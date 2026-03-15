# Attempt 4: Dynamic HyDE + Zero Padding — Score 0.518 (1st place)

## Score progression

| Version | Score | Key change |
|---------|-------|------------|
| v4 | 0.367 | Baseline: Whisper Tiny + BGE-M3 + 90s windows + ±10s padding |
| v8 | 0.377 | faster-whisper large-v3-turbo retranscription |
| v11 | 0.470 | Answer augmentation from train data (+0.09) |
| v12 | 0.500 | HyDE query expansion (+0.03) |
| v15 | 0.508 | P75 boundary truncation for windows, ±5s for answers |
| **v19** | **0.518** | **Dynamic HyDE weight + zero padding for answers** |

## What makes v19 the best

### 1. Dynamic HyDE weight
Instead of fixed 0.6/0.4 mixing of query and answer embeddings:
```
ans_weight = clamp(0.2 + (similarity - 0.7) * (0.5 / 0.3), 0.2, 0.7)
search_vec = (1 - ans_weight) * query_emb + ans_weight * answer_emb
```
- similarity = 0.7 (threshold) → ans_weight = 0.2 (mostly query)
- similarity = 0.85 → ans_weight = 0.45 (balanced)
- similarity = 1.0 → ans_weight = 0.7 (mostly answer)

More confident train match → trust answer embedding more.

### 2. Zero padding for answer_aug chunks
Answer augmentation chunks already have exact ground truth timestamps from training data. Adding padding (±5s or ±10s) only reduces IoU. Zero padding = maximum IoU.

### 3. P75 boundary truncation for windows
Window chunks (90s) are too wide for IoU ≥ 0.5. Center the prediction and trim to P75 of train fragment distribution (94s). This matches the typical ground truth fragment length.

## Architecture

```
OFFLINE:
  Audio (436 files)
    → faster-whisper large-v3-turbo (8 GPU parallel, beam_size=5)
    → new_transcripts.pkl

  Transcripts → 90s/30s sliding windows → 5010 chunks
  Train answers → answer augmentation → +4466 chunks
  Total: 9476 chunks → BGE-M3 1024d → numpy index

  Train questions → BGE-M3 → train question index (for HyDE matching)

ONLINE (<100ms/query):
  Query → BGE-M3 encode
  → Find similar train question (cosine)
  → Dynamic HyDE: mix with answer embedding (variable weight)
  → Cosine search top-10 → dedup
  → Adaptive boundary:
      answer_aug → exact timestamps (zero padding)
      windows → center ± P75/2 (47s each side)
  → 5 × (video_stem, start, end)
```

## What didn't work (lessons learned)

| Approach | Score | Why it failed |
|----------|-------|---------------|
| Multi-scale 171K chunks | 0.289 | Too many sentence chunks flooded results |
| Cascaded coarse+fine | 0.328 | Lost video diversity in top-5 |
| bge-reranker-v2-m3 | 0.484 | Cross-encoder re-ordered away from best temporal matches |
| question_en/ru augmentation | 0.477 | Questions too similar, duplicated chunks |
| Hybrid dense+sparse RRF | 0.404 | Sparse search disrupted ranking |
| Dynamic boundary by score | 0.508 | Score doesn't correlate with optimal window width on test |

## Key insight

The biggest gains came from **bridging the semantic gap** between queries and transcripts:
- Answer augmentation (+0.09): answers use query-like vocabulary
- HyDE (+0.03): shifts search vector toward answer space
- Dynamic HyDE (+0.01): better calibration of mixing weights

Chunking and boundary strategies gave smaller gains (+0.008-0.018). The retrieval quality matters more than boundary precision.

## Files

```
attempt4/
├── run_v19.py                 # Complete pipeline (score 0.518)
├── config.py                  # Parameters
├── evaluate.py                # Local evaluation (SR@K, VR@K)
├── retranscribe_parallel.py   # 8-GPU parallel ASR
├── retranscribe_fw.py         # Single-GPU ASR fallback
├── merge_transcripts.py       # Merge per-GPU transcripts
├── pipeline/
│   ├── chunker.py             # Sliding window chunker
│   └── indexer.py             # BGE-M3 numpy index
└── search/
    └── retriever.py           # Dynamic HyDE + adaptive boundary
```

## Usage

```bash
# Symlink data
ln -sf /path/to/data data
ln -sf /path/to/new_transcripts.pkl new_transcripts.pkl

# Run pipeline
python run_v19.py
# Output: submission_v19.csv

# Evaluate locally
python evaluate.py --sample 200
```

## Requirements
```
sentence-transformers
faster-whisper
numpy
pandas
tqdm
static-ffmpeg
```

---

## Рекомендации для улучшения

### Что показали эксперименты — уроки

Прежде чем рекомендовать что-то новое, стоит зафиксировать что провалилось и ПОЧЕМУ:

- **Cross-encoder reranking (0.484)** — ухудшил скор, потому что переупорядочил результаты в пользу семантически близких, но ВРЕМЕННÓ неточных чанков. Reranker оптимизирует text relevance, а наша метрика — IoU по таймкодам. Answer augmentation чанки с точными таймкодами проигрывают window чанкам по text quality, но выигрывают по IoU. Reranker этого не знает.
- **BM25 + RRF (0.404)** — sparse search вносит шум. BM25 хорош для exact match, но здесь запросы семантические, а не лексические. Fusion размыла сильный dense signal.
- **Multi-scale chunking (0.289)** — мелкие чанки затопили результаты. При 171K чанков top-10 забит мелкими кусками из одного видео → VR@K обвалился.

**Вывод:** в этой задаче побеждает НЕ усложнение pipeline, а улучшение качества ОДНОГО dense retrieval шага. Все успешные улучшения (answer aug, HyDE) работали именно через усиление query embedding.

---

### Приоритет 1: Fine-tuning BGE-M3 на train data (ожидаемый прирост +0.03-0.07)

Это самое мощное неиспользованное улучшение. Все текущие gains (answer aug, HyDE) — это обходные пути чтобы query embedding был ближе к нужным чанкам. Fine-tuning делает то же самое, но на уровне самой модели.

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer("BAAI/bge-m3")
examples = []

for _, row in train_df.iterrows():
    # Позитив 1: вопрос → текст чанка из [start, end]
    chunk_text = get_chunk_text(row["video_file"], row["start"], row["end"])
    if chunk_text and len(chunk_text) > 20:
        examples.append(InputExample(texts=[row["question_en"], chunk_text]))
    
    # Позитив 2: вопрос → answer_en
    if pd.notna(row.get("answer_en")):
        examples.append(InputExample(texts=[row["question_en"], row["answer_en"]]))

# Hard negatives: чанки из ТОГО ЖЕ видео, но другого окна
# Это учит модель различать правильный и неправильный момент в одном видео
for _, row in train_df.iterrows():
    wrong_chunks = get_other_chunks_same_video(row["video_file"], row["start"], row["end"])
    for neg_text in wrong_chunks[:2]:
        examples.append(InputExample(texts=[row["question_en"], chunk_text, neg_text]))

loss = losses.MultipleNegativesRankingLoss(model)
# С hard negatives: losses.TripletLoss(model)
model.fit([(DataLoader(examples, batch_size=16, shuffle=True), loss)],
          epochs=3, warmup_steps=100, output_path="bge-m3-finetuned")
```

**Почему это должно сработать:** answer augmentation дал +0.09 по сути добавив "подсказки" в индекс. Fine-tuning встроит это знание прямо в модель — каждый query embedding будет изначально ближе к правильным чанкам без нужды в augmentation.

**Риск:** overfitting на train. Митигация: оставить 20% train для валидации, early stopping.

**Время:** 2-3 часа (подготовка данных + 3 epochs + переиндексация).

---

### Приоритет 2: GigaAM-v3 для русских видео (ожидаемый прирост +0.01-0.03)

faster-whisper large-v3-turbo хорош, но для русского GigaAM-v3 значительно лучше. Sber'овская модель обучена на 700K часов русской речи, WER на ~50% ниже Whisper на русском. Модель v3_e2e_rnnt выдаёт пунктуированный нормализованный текст.

```python
import gigaam
model = gigaam.load_model("v3_e2e_rnnt")
# Для коротких аудио (<25с):
transcription = model.transcribe(audio_path)
# Для длинных:
utterances = model.transcribe_longform(long_audio_path)
```

**Стратегия:** определить язык каждого видео (через Whisper detect language) → русские → GigaAM-v3, английские → faster-whisper large-v3-turbo.

**Время:** 1-2 часа (установка + перетранскрибирование русских видео).

---

### Приоритет 3: SigLIP 2 visual search как дополнительный канал (ожидаемый прирост +0.01-0.03)

BM25 RRF провалился, но визуальный канал — другое дело. BM25 конкурирует с dense на ТОМ ЖЕ типе данных (текст). SigLIP 2 добавляет ДРУГУЮ модальность — визуал.

Запросы типа "тёмная сцена", "человек у окна", "на фоне гор" не содержатся в транскриптах вообще. Visual search — единственный способ их покрыть.

**ВАЖНО:** не использовать RRF как с BM25. Вместо этого — fallback strategy:

```python
def search(query, query_emb, text_index, visual_index, chunks):
    # Шаг 1: стандартный dense text search (как в v19)
    text_results = text_search(query_emb, text_index, top_k=10)
    
    # Шаг 2: если top-1 score НИЗКИЙ — добавить visual результаты
    if text_results[0]["score"] < CONFIDENCE_THRESHOLD:
        visual_results = visual_search(query, visual_index, top_k=5)
        # Вставить visual результаты на позиции 3-5 (не заменять top text)
        combined = text_results[:2] + merge_unique(visual_results, text_results[2:])
        return combined[:5]
    
    return text_results[:5]
```

Это НЕ ломает текущий pipeline (в отличие от RRF), а добавляет visual только когда текст не уверен.

**Модель:** `google/siglip2-so400m-patch14-384` — нативная поддержка русского.

**Время:** 3 часа (извлечение кадров + encoding + интеграция).

---

### Приоритет 4: Исправление опечаток в запросах (ожидаемый прирост +0.005-0.015)

Условие задачи: запросы могут содержать опечатки. BGE-M3 частично робастна, но "дракка" вместо "драка" может сместить embedding.

```python
# Вариант 1: Yandex.Speller API (бесплатный, для русского)
import requests
def fix_typos_ru(text):
    r = requests.get("https://speller.yandex.net/services/spellservice.json/checkText",
                     params={"text": text, "lang": "ru"})
    result = text
    for change in reversed(r.json()):  # reversed чтобы не сбивать индексы
        s = change["pos"]
        e = s + change["len"]
        result = result[:s] + change["s"][0] + result[e:]
    return result

# Вариант 2: TextBlob для английского
from textblob import TextBlob
def fix_typos_en(text):
    return str(TextBlob(text).correct())
```

**Время:** 30 минут.

---

### Приоритет 5: Увеличить train coverage через synthetic queries (ожидаемый прирост +0.01-0.02)

У тебя 4466 answer_aug чанков = 4466 вопросов покрыты HyDE. Но тестовых запросов 812 — и не все из них похожи на train вопросы (similarity > 0.7). Для "непокрытых" запросов HyDE не срабатывает.

```python
# Сгенерировать дополнительные вопросы из answer_en через LLM
# Это разрешено: "Использование LLM для предобработки не запрещено"
from transformers import pipeline
gen = pipeline("text2text-generation", model="google/flan-t5-base")

for _, row in train_df.iterrows():
    answer = row["answer_en"]
    # Генерировать 3-5 альтернативных вопросов
    prompts = [
        f"Generate a question that this text answers: {answer}",
        f"Rephrase as a search query: {row['question_en']}",
    ]
    for p in prompts:
        alt_question = gen(p, max_length=64)[0]["generated_text"]
        # Добавить alt_question → answer embedding mapping в HyDE index
```

Это расширяет "покрытие" HyDE — больше тестовых запросов найдут match > 0.7.

**Время:** 2 часа.

---

### Приоритет 6: Tune Dynamic HyDE детальнее (ожидаемый прирост +0.005-0.01)

Текущая формула линейная: `0.2 + (sim - 0.7) * (0.5 / 0.3)`. Можно попробовать:

```python
# Вариант 1: Квадратичная (более агрессивная на высоких similarity)
ans_weight = 0.2 + 0.5 * ((sim - 0.7) / 0.3) ** 2

# Вариант 2: Per-chunk-type weight
# Если best match — answer_aug чанк → доверять больше
# Если best match — window чанк → доверять меньше

# Вариант 3: Multi-HyDE — смешивать не с одним, а с top-3 similar answers
similar_answers = find_top_k_similar_train(query_emb, k=3)
avg_answer_emb = np.mean([ans["emb"] for ans in similar_answers], axis=0)
search_vec = (1 - w) * query_emb + w * avg_answer_emb
```

**Время:** 1-2 часа (grid search на val split).

---

### Итоговая таблица приоритетов

| # | Что | Время | Ожидаемый прирост | Риск | Пробовали? |
|---|-----|-------|-------------------|------|-----------|
| 1 | Fine-tune BGE-M3 | 2-3ч | +0.03-0.07 | Средний (overfitting) | ❌ Нет |
| 2 | GigaAM-v3 для RU | 1-2ч | +0.01-0.03 | Низкий | ❌ Нет |
| 3 | SigLIP 2 visual (fallback mode) | 3ч | +0.01-0.03 | Средний | ❌ Нет |
| 4 | Spellcheck запросов | 30м | +0.005-0.015 | Низкий | ❌ Нет |
| 5 | Synthetic queries для HyDE | 2ч | +0.01-0.02 | Низкий | ❌ Нет |
| 6 | Tune HyDE formula | 1-2ч | +0.005-0.01 | Низкий | Частично |

**Суммарный потенциал:** +0.07-0.18 → **score 0.59-0.70**