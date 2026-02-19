import os
import time
import logging
import sys

# Set CUDA to only use GPU 0 (как в оригинале - "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Add current directory to path for imports
sys.path.append('/home/k4/Projects/BirdsMilkDatasetPreprocessing')
sys.path.append('/home/k4/Projects/BirdsMilkDatasetPreprocessing/Music-Source-Separation-Training')

from audio_separator import setup_models, process_audio, MODELS


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    input_file = "/home/k4/Python/windowed-roformer/input.mp3"
    output_file = "/home/k4/Python/windowed-roformer/output_vocal.wav" # Файл будет перезаписываться каждый раз
    
    # Проверка наличия входного файла
    if not os.path.exists(input_file):
        logger.error(f"Файл {input_file} не найден. Пожалуйста, положите input.mp3 в папку со скриптом.")
        return

    logger.info("Initializing models...")
    
    # Инициализируем ТОЛЬКО vocal модель (MelBandRoformer)
    setup_models({
        'vocal': {
            'config_url': 'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/config_kimmel_unwa_ft.yaml',
            'ckpt_url': 'https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT/resolve/main/kimmel_unwa_ft.ckpt',
            'model_type': 'mel_band_roformer'
        }
    })

    logger.info("Модель загружена. Начинаем бенчмарк (10 прогонов)...")

    execution_times = []

    for i in range(10):
        logger.info(f"--- Запуск {i + 1}/10 ---")
        
        # Начало замера времени (используем perf_counter для высокой точности)
        start_time = time.perf_counter()
        
        try:
            # Запускаем инференс (только vocal extraction)
            process_audio(
                MODELS['vocal'],
                input_file,
                output_file,
                extract_instrumental=False
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

    # Итоговая статистика
    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        min_time = min(execution_times)
        max_time = max(execution_times)
        
        logger.info("\n=== Результаты бенчмарка ===")
        logger.info(f"Среднее время: {avg_time:.2f} мс")
        logger.info(f"Минимальное:   {min_time:.2f} мс")
        logger.info(f"Максимальное:  {max_time:.2f} мс")
        logger.info("============================")

if __name__ == "__main__":
    main()