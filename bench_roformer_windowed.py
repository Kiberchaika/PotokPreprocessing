import os
import time
import logging
import torch
# Импортируем функции из предоставленного main.py
from main import load_model, load_audio, demix, save_audio

# Set CUDA to only use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_ckpt_if_missing(path, url):
    """Скачивает чекпоинт, если его нет."""
    if not os.path.exists(path):
        logger.info(f"Чекпоинт не найден. Скачивание {path}...")
        try:
            torch.hub.download_url_to_file(url, path)
        except Exception as e:
            logger.error(f"Не удалось скачать чекпоинт: {e}")
            raise

def main():
    input_file = "/home/k4/Python/windowed-roformer/input.mp3"
    output_file = "/home/k4/Python/windowed-roformer/output_wsa.wav"
    
    # Конфигурация модели и чекпоинта
    # Используем дефолтный чекпоинт из README
    ckpt_name = "mbr-win10-sink8.ckpt"
    ckpt_url = "https://huggingface.co/smulelabs/windowed-roformer/resolve/main/mbr-win10-sink8.ckpt"
    
    # Параметры инференса (как в config.yaml)
    sample_rate = 44100
    chunk_size = 8  # Секунд на чанк (для обработки длинных файлов)
    batch_size = 4  # Размер батча для чанков
    
    # Проверка входного файла
    if not os.path.exists(input_file):
        logger.error(f"Файл {input_file} не найден. Пожалуйста, положите input.mp3 в папку со скриптом.")
        return

    # Скачивание модели
    try:
        download_ckpt_if_missing(ckpt_name, ckpt_url)
    except Exception:
        return

    # Определение устройства
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Используемое устройство: {device}")

    # Загрузка модели (один раз перед циклом)
    logger.info("Инициализация модели Windowed Roformer...")
    try:
        # load_model из main.py
        model = load_model(ckpt_name, device)
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        logger.error("Убедитесь, что установлен PyTorch >= 2.5 для поддержки FlexAttention.")
        return

    # Загрузка и предобработка аудио (чтобы не включать I/O диска в замер времени инференса)
    logger.info("Загрузка аудио в память...")
    audio_tensor = load_audio(input_file, sample_rate)

    logger.info("Начинаем бенчмарк (10 прогонов)...")
    execution_times = []

    for i in range(10):
        logger.info(f"--- Запуск {i + 1}/10 ---")
        
        # Начало замера времени
        start_time = time.perf_counter()
        
        try:
            # Запуск инференса (demix из main.py)
            separated_audio = demix(
                model,
                audio_tensor,
                device=device,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"Ошибка во время инференса: {e}")
            return

        # Конец замера
        end_time = time.perf_counter()
        
        # Перевод в миллисекунды
        elapsed_ms = (end_time - start_time) * 1000
        execution_times.append(elapsed_ms)
        
        logger.info(f"Время выполнения прогона {i + 1}: {elapsed_ms:.2f} мс")
        
        # Сохраняем файл (перезаписываем), чтобы убедиться, что результат есть
        # Операция сохранения вынесена из таймера, так как нас интересует скорость модели
        save_audio(separated_audio, output_file, sample_rate)

    # Итоговая статистика
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        logger.info("\n=== Результаты бенчмарка (Windowed Roformer) ===")
        logger.info(f"Среднее время: {avg_time:.2f} мс")
        logger.info(f"Минимальное:   {min_time:.2f} мс")
        logger.info(f"Максимальное:  {max_time:.2f} мс")
        logger.info("==============================================")

if __name__ == "__main__":
    main()