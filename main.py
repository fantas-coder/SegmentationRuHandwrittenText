from ultralytics import YOLO
import cv2
import logging
from pathlib import Path
import matplotlib.pyplot as plt


# ========== НАСТРОЙКИ ==========
MODEL_PATH = 'models/tuomiokirja_lines_05122023.pt'
SOURCE_PATH = 'images/img7.png'
OUTPUT_DIR = 'cropped_boxes'
CONF_THRESHOLD = 0.3                                    # Порог уверенности (0-1)
OVERLAP_THRESHOLD = 0.35                                # Порог пересечения (50% площади каждого бокса)
IMG_FORMAT = 'png'                                      # Формат изображений (png/jpg)
SCALE_COEFF = 2                                         # Коэффициент растяжения
SCALE_BBOX = 0.01                                       # Процент увеличения ббокса
# ================================


def setup_logging():
    """Функция для логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('../processing.log'), logging.StreamHandler()]
    )


def preprocess_image(image):
    """Растягиваем картинку в оттенках серого по Y"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray, None, fx=1, fy=SCALE_COEFF, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)


def calculate_intersection(box_a, box_b):
    """Вычисляет площадь пересечения двух bounding boxes"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_boxes(boxes):
    """Фильтрует дублирующиеся bounding boxes"""
    filtered = []
    for current_box in sorted(boxes, key=lambda x: x.conf, reverse=True):
        current_coords = current_box.xyxy[0].int().tolist()
        current_area = (current_coords[2] - current_coords[0]) * (current_coords[3] - current_coords[1])

        keep = True
        for idx, kept_box in enumerate(filtered):
            kept_coords = kept_box.xyxy[0].int().tolist()
            kept_area = (kept_coords[2] - kept_coords[0]) * (kept_coords[3] - kept_coords[1])

            # Вычисляем площадь пересечения
            inter_area = calculate_intersection(current_coords, kept_coords)

            # Проверяем условие пересечения
            if inter_area > OVERLAP_THRESHOLD * current_area and inter_area > OVERLAP_THRESHOLD * kept_area:
                # # Выбор большего бокса
                # if current_area > kept_area:
                #     filtered[idx] = current_box  # Замена на больший бокс
                keep = False
                break

        if keep:
            filtered.append(current_box)
    return filtered


def main():
    # Логирование
    setup_logging()

    try:
        model = YOLO(MODEL_PATH)
        logging.info(f'Model "{MODEL_PATH}" loaded')
    except Exception as e:
        logging.error(f'Error loading model: {str(e)}')
        return

    # Создаем выходную директорию
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Обрабатываем исходный путь
    source_path = Path(SOURCE_PATH)
    if not source_path.exists():
        logging.error(f'Path {source_path} does not exist')
        return

    image_paths = []
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_paths = list(source_path.glob('*.*'))

    for img_path in image_paths:
        if img_path.suffix.lower()[1:] not in ['jpg', 'jpeg', 'png']:
            continue

        logging.info(f'Processing: {img_path}')
        original_img = cv2.imread(str(img_path))
        processed_img = preprocess_image(original_img)
        plt.show()

        results = model.predict(source=processed_img, iou=0.2, agnostic_nms=True)

        for result in results:
            # Получаем исходное изображение
            annotated_img = original_img.copy()
            source_filename = img_path.stem

            # Фильтрация боксов
            boxes = [box for box in result.boxes if box.conf >= CONF_THRESHOLD]
            filtered_boxes = filter_boxes(boxes)

            # Сортировка боксов: сверху-вниз, слева-направо
            sorted_boxes = sorted(filtered_boxes,
                                  key=lambda b: (
                                      b.xyxy[0][1].item(),  # Сортировка по Y (верхняя граница)
                                      b.xyxy[0][0].item()  # Затем по X (левая граница)
                                  ))

            # Получаем информацию о bounding boxes
            for i, box in enumerate(sorted_boxes):
                if box.conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())     # Координаты bbox в формате (x1, y1, x2, y2)
                class_id = int(box.cls[0].item())                   # Confidence score
                class_name = model.names.get(class_id, 'unknown')   # Class ID (индекс класса)
                y1 //= SCALE_COEFF                                   # Возвращение к исходному формату y1
                y2 //= SCALE_COEFF                                   # Возвращение к исходному формату y2
                x1 = int(x1 - SCALE_BBOX * x1)
                y1 = int(y1 - SCALE_BBOX * y1)
                x2 = int(x2 + SCALE_BBOX * x2)
                y2 = int(y2 + SCALE_BBOX * y2)

                # TODO Написать функцию разделяющую ббокс строки на ббоксы слов
                # word_bboxes = get_word_bboxes(x1, y1, x2, y2)

                # TODO Сделать цикл вырезания каждого слова из полученного массива выше
                # for x1, y1, x2, y2 in word_bboxes:
                # Вырезаем изображение
                try:
                    cropped_img = original_img[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        logging.warning(f'Empty region detected in {img_path}')
                        continue

                    # Создаём файл
                    output_name = f"{source_filename}_{class_name}_{i}_conf{box.conf[0]:.2f}.{IMG_FORMAT}"
                    output_path = output_dir / output_name

                    # Save the cropped image
                    cv2.imwrite(str(output_path), cropped_img,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100] if IMG_FORMAT == 'jpg' else [])
                    logging.info(f'Saved: {output_path}')

                    # Draw on annotated image
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                except Exception as e:
                    logging.error(f'Error processing box {i} in {img_path}: {str(e)}')

                # Save annotated image
                annotated_path = output_dir / f"{source_filename}_annotated.{IMG_FORMAT}"
                cv2.imwrite(str(annotated_path), annotated_img)
                logging.info(f'Saved annotated: {annotated_path}')


if __name__ == '__main__':
    main()

# Box precision 0.912
# Box recall 0.888
# Box mAP50 0.949
# Box mAP50-95 0.701
