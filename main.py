from ultralytics import YOLO
import cv2
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu


# ========== НАСТРОЙКИ ==========
MODEL_PATH = 'models/tuomiokirja_lines_05122023.pt'
SOURCE_PATH = 'images/img25.jpg'
OUTPUT_DIR = 'cropped_boxes'
CONF_THRESHOLD = 0.3                                    # Порог уверенности (0-1)
OVERLAP_THRESHOLD = 0.35                                # Порог пересечения (50% площади каждого бокса)
IMG_FORMAT = 'png'                                      # Формат изображений (png/jpg)
SCALE_COEFF = 2                                         # Коэффициент растяжения
SCALE_BBOX = 0.01                                       # Процент увеличения ббокса
SPACE_THRESHOLD_COEFF = 0.0025
MIN_SPACE_WIDTH = 0.03
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


def get_word_bboxes(img, x1, y1, x2, y2):
    # Обрезаем область строки
    crop = img[y1:y2, x1:x2]

    # Преобразуем в градации серого и бинаризуем
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Применяем морфологическое закрытие для объединения символов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Считаем горизонтальную проекцию
    projection = np.sum(closed, axis=0)
    # projection = cv2.medianBlur(projection.astype(np.uint8), 3)

    # Находим пробелы (адаптивный порог: % от максимальной высоты строки)
    height = y2 - y1
    space_threshold = SPACE_THRESHOLD_COEFF * height * 255  # % от максимально возможной суммы
    # median_projection = np.mean(projection[projection > 0])
    # space_threshold = 0.3 * median_projection  # 10% от медианной интенсивности символов

    # plt.plot(projection)
    # plt.axhline(y=space_threshold, color='r')
    # plt.show()

    space_indices = np.where(projection < space_threshold)[0]

    # Определяем группы непрерывных пробелов
    width = x2 - x1
    spaces = []
    if len(space_indices) > 0:
        start = space_indices[0]
        end = start
        for idx in space_indices[1:]:
            if idx - end == 1:
                end = idx
            else:
                # Проверяем ширину пробела перед добавлением
                if (end - start + 1) >= width * MIN_SPACE_WIDTH:
                    spaces.append((start, end))
                start = end = idx
        if (end - start + 1) >= width * MIN_SPACE_WIDTH:
            spaces.append((start, end))

    # Вычисляем границы слов
    word_boxes = []
    prev_end = 0
    for (s_start, s_end) in spaces:
        if s_start > prev_end:
            word_boxes.append((prev_end, 0, s_start, closed.shape[0]))
        prev_end = s_end

    # Добавляем последнее слово
    if prev_end < closed.shape[1]:
        word_boxes.append((prev_end, 0, closed.shape[1], closed.shape[0]))

    # Фильтруем мелкие артефакты и преобразуем в глобальные координаты
    width = x2 - x1
    min_word_width = width * 0.01  # Минимальная ширина слова в пикселях
    filtered = []
    for (wx1, wy1, wx2, wy2) in word_boxes:
        if wx2 - wx1 >= min_word_width:
            global_x1 = int((x1 + wx1) * 0.99)
            global_y1 = int((y1 + wy1) * 0.99)
            global_x2 = int((x1 + wx2) * 1.01)
            global_y2 = int((y1 + wy2) * 1.01)
            filtered.append((global_x1, global_y1, global_x2, global_y2))

            cv2.rectangle(img, (global_x1, global_y1), (global_x2, global_y2), (0, 255, 0), 2)

    return filtered


def sort_word_bboxes(word_bboxes, line_overlap_threshold=0.7):
    """
    Сортирует bounding boxes по правилам:
    1. По вертикали (по y1)
    2. Для объектов в одной строке - по горизонтали (по x1)

    :param word_bboxes: список кортежей (x1, y1, x2, y2)
    :param line_overlap_threshold: порог перекрытия для определения одной строки (0.5 = 50%)
    :return: отсортированный список координат
    """
    # Рассчитываем высоты bounding boxes
    bboxes_with_height = [(box, box[3] - box[1]) for box in word_bboxes]

    # Сортируем по вертикали (основной критерий - y1)
    sorted_boxes = sorted(bboxes_with_height, key=lambda x: x[0][1])

    # Группируем по строкам
    lines = []
    for box, h in sorted_boxes:
        y_center = (box[1] + box[3]) / 2  # Центр по вертикали
        matched = False

        # Проверяем существующие линии
        for line in lines:
            # Берем эталонный bbox из линии
            ref_box = line[0][0]
            line_height = ref_box[3] - ref_box[1]

            # Порог перекрытия с учетом высоты линии
            threshold = line_height * line_overlap_threshold

            # Проверяем вертикальное перекрытие
            if ref_box[1] - threshold <= y_center <= ref_box[3] + threshold:
                line.append((box, h))
                matched = True
                break

        # Если не нашли подходящую линию, создаем новую
        if not matched:
            lines.append([(box, h)])

    # Сортируем каждую линию по x1 и формируем итоговый список
    result = []
    for line in lines:
        # Сортировка внутри линии
        line_sorted = sorted(line, key=lambda x: x[0][0])
        # Оставляем только координаты
        result.extend([box for box, h in line_sorted])

    return result


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

                # TODO Написать функцию разделяющую ббокс строки на ббоксы слов
                word_bboxes = get_word_bboxes(annotated_img, x1, y1, x2, y2)
                word_bboxes = sort_word_bboxes(word_bboxes)

                # TODO Сделать цикл вырезания каждого слова из полученного массива выше
                for j, (x1, y1, x2, y2) in enumerate(word_bboxes):
                    # Вырезаем изображение
                    try:
                        # Проверка координат
                        h, w = original_img.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x1 >= x2 or y1 >= y2:
                            logging.warning(f"Invalid bbox {i}_{j} in {img_path}: {x1},{y1},{x2},{y2}")
                            continue

                        cropped_img = original_img[y1:y2, x1:x2]
                        if cropped_img.size == 0:
                            logging.warning(f"Empty crop in {img_path} for bbox {i}_{j}")
                            continue

                        # Создаём файл
                        output_name = f"{source_filename}_{class_name}_{i}_{j}_conf{box.conf[0]:.2f}.{IMG_FORMAT}"
                        output_path = output_dir / output_name

                        # Save the cropped image
                        cv2.imwrite(str(output_path), cropped_img,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100] if IMG_FORMAT == 'jpg' else [])
                        logging.info(f"Saved word {i}_{j} to {output_path}")

                    except Exception as e:
                        logging.error(f'Error processing box {i}_{j} in {img_path}: {str(e)}')

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
