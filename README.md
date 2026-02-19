# Blackbird Dataset Preprocessing

Пайплайн предобработки музыкального датасета. Включает разделение вокала и музыки, детекцию битов, распознавание речи (ASR), а также набор бенчмарков для выбора оптимальных моделей на каждом этапе.

## Архитектура

```
MP3-датасет (WebDAV / локально)
        │
        ▼
┌───────────────────┐
│  audio_pipeline.py │  ← основной пайплайн
└───────┬───────────┘
        │
        ├─► Beat This!          → биты / даунбиты
        ├─► Windowed Roformer   → вокал / музыка (Opus 160 kbps)
        └─► NVIDIA Parakeet     → транскрипция + word-level timestamps
              │
              ▼
        output/<id>/<name>_beats.json + _lyrics.json + _voc.opus + _music.opus
```

## Скрипты

### Основной пайплайн

| Скрипт | Назначение |
|--------|-----------|
| `audio_pipeline.py` | Главный пайплайн: детекция битов (Beat This!), разделение вокала/музыки (Windowed Roformer), кодирование в Opus, ASR-транскрипция (Parakeet). Режимы: `beats`, `roformer-asr`, `all`. Поддерживает `--bench` для замера производительности по батчам |
| `process_remote_streaming.py` | Потоковая обработка датасета с удалённого Blackbird WebDAV-сервера: скачивание батчами, обработка, загрузка результатов обратно, переиндексация |

### Бенчмарки

| Скрипт | Модель | Оптимизация |
|--------|--------|-------------|
| `bench_beat_this.py` | Beat This! (Audio2Beats) | Baseline |
| `bench_beat_this_trt.py` | Beat This! (Audio2Beats) | torch.compile (TRT / Inductor) |
| `bench_beatnet.py` | BeatNet (DBN) | Baseline |
| `bench_roformer_orig.py` | MelBandRoformer (оригинальный) | Baseline |
| `bench_roformer_windowed.py` | Windowed Roformer (WSA) | Baseline |
| `bench_roformer_windowed_trt.py` | Windowed Roformer (WSA) | torch.compile (Inductor) |
| `bench_asr_parakeet.py` | NVIDIA Parakeet TDT 0.6B v3 | Baseline |
| `bench_asr_parakeet_trt.py` | NVIDIA Parakeet TDT 0.6B v3 | torch.compile (TRT / Inductor) |
| `bench_asr_qwen.py` | Qwen3-ASR-1.7B (vLLM) | Baseline |
| `bench_rmvpe.py` | RMVPE (F0 pitch) | Baseline |
| `bench_rmvpe_trt.py` | RMVPE (F0 pitch) | torch.compile (TRT / Inductor) |
| `bench_swift.py` | SwiftF0 (ONNX + CUDA) | ONNX Runtime CUDA |
| `bench_swift_trt.py` | SwiftF0 (ONNX + TRT) | ONNX Runtime TensorRT FP16 |

### Бенчмарки — качество ASR (WER)

Все WER-скрипты сканируют тестовый датасет с парами `<name>.txt` (эталонный текст) и `<name>_vocal.mp3`, транскрибируют аудио целевой моделью и вычисляют Word Error Rate.

| Скрипт | Модель |
|--------|--------|
| `bench_wer_parakeet.py` | NVIDIA Parakeet TDT v3 |
| `bench_wer_parakeet_trt.py` | NVIDIA Parakeet TDT v3 (torch.compile) |
| `bench_wer_qwen.py` | Qwen3-ASR-1.7B |
| `bench_wer_whisper.py` | OpenAI Whisper Large v3 |
| `bench_wer_whisper_turbo.py` | OpenAI Whisper Large v3 Turbo |
| `bench_wer_liteasr.py` | LiteASR (lite-whisper-large-v3-turbo) |

### Управление датасетом

| Скрипт | Назначение |
|--------|-----------|
| `download_lyrics.py` | Загрузка текстов песен из LRCLIB, lyrics.ovh и Genius по ID3-тегам. Параллельные воркеры, rate-limiting |
| `split_dataset.py` | Разделение датасета на N частей примерно равного размера (bin-packing по артистам). `--dry-run` для предпросмотра |
| `make_filelists_for_transfer.py` | Генерация filelist первых 1.5 TB для `rsync --files-from` |
| `scan_mp3_duration.py` | Подсчёт суммарной длительности всех MP3 в директории |
| `setup_remote_server.sh` | Настройка удалённого сервера: nginx + WebDAV, схема Blackbird-датасета, индексация, запуск сервера на порту 8085 |

### Утилиты

| Скрипт | Назначение |
|--------|-----------|
| `read_pitch.py` | Загрузка и вывод сохранённого pitch-тензора (`.pt`) |

## Конфигурация и данные

| Файл | Описание |
|------|----------|
| `scheme.json` | Схема Blackbird-датасета: 8 типов компонентов (mp3, info, lyrics, beats, pitch, vocal, vocal_dereverb, music) |
| `bench_preprocessing.html` | HTML-отчёт с результатами бенчмарков |

## Формат выходных данных

Для каждого входного MP3 пайплайн генерирует:

- **`<name>_beats.json`** — биты и даунбиты (секунды)
- **`<name>_lyrics.json`** — ASR-транскрипция с сегментами и word-level timestamps
- **`<name>_voc.opus`** — вокальная дорожка (Opus 160 kbps)
- **`<name>_music.opus`** — музыкальная дорожка (Opus 160 kbps)
