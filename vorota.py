import cv2
import numpy as np
import math

# Параметры рамки (в метрах)
OUTER_FRAME_WIDTH = 0.6  # Внешняя ширина рамки
OUTER_FRAME_HEIGHT = 0.4  # Внешняя высота рамки
INNER_FRAME_WIDTH = 0.5  # Внутренняя ширина рамки (вырез)
INNER_FRAME_HEIGHT = 0.3  # Внутренняя высота рамки (вырез)
FOCAL_LENGTH = 700  # Фокусное расстояние камеры (примерное значение)

# Функция для расчета расстояния
def calculate_distance(point1, point2, real_distance, focal_length):
    pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))
    return (real_distance * focal_length) / pixel_distance

# Функция для вычисления углов до центра объекта
def calculate_angle(cx, frame_width):
    fov = 60  # Поле зрения камеры (примерное значение, градусов)
    angle_per_pixel = fov / frame_width
    offset = cx - frame_width / 2  # Смещение центра объекта относительно центра кадра
    return offset * angle_per_pixel

# Функция для нахождения угловых точек рамки
def find_corners(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) >= 4:  # Ищем хотя бы 4 угловых точки
        return [tuple(pt[0]) for pt in approx]
    return []

# Открываем видеопоток
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование изображения в HSV для фильтрации красного цвета
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([25, 145, 85])
    upper_red1 = np.array([5, 255, 255])
    lower_red2 = np.array([165, 120, 70])
    upper_red2 = np.array([185, 255, 255])

    # Маска для красного цвета
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Поиск контуров
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    visible_outer_corners = []
    visible_inner_corners = []

    for contour in contours:
        corners = find_corners(contour)
        if len(corners) >= 4:  # Если найдено хотя бы 4 угловых точки
            # Определяем, внешний это контур или внутренний (по площади)
            area = cv2.contourArea(contour)
            outer_area = OUTER_FRAME_WIDTH * OUTER_FRAME_HEIGHT
            inner_area = INNER_FRAME_WIDTH * INNER_FRAME_HEIGHT
            if area > outer_area / 4:  # Примерный порог для внешнего контура
                visible_outer_corners.extend(corners)
            elif area < inner_area:  # Примерный порог для внутреннего контура
                visible_inner_corners.extend(corners)

    # Расчет расстояния и углов на основе видимых точек
    if visible_outer_corners and visible_inner_corners:
        # Упрощенно используем первую пару точек для вычисления расстояния
        if len(visible_outer_corners) >= 2 and len(visible_inner_corners) >= 2:
            p1, p2 = visible_outer_corners[:2]
            q1, q2 = visible_inner_corners[:2]

            # Расчет расстояний для внешнего и внутреннего контура
            outer_width_distance = calculate_distance(
                p1, p2, OUTER_FRAME_WIDTH, FOCAL_LENGTH
            )
            outer_height_distance = calculate_distance(
                p1, q1, OUTER_FRAME_HEIGHT, FOCAL_LENGTH
            )
            inner_width_distance = calculate_distance(
                q1, q2, INNER_FRAME_WIDTH, FOCAL_LENGTH
            )
            inner_height_distance = calculate_distance(
                q1, p2, INNER_FRAME_HEIGHT, FOCAL_LENGTH
            )

            # Среднее расстояние для учета толщины
            average_distance = (outer_width_distance + outer_height_distance +
                                inner_width_distance + inner_height_distance) / 4

            # Расчет угла на основе внешнего контура
            cx = (p1[0] + p2[0]) / 2
            angle = calculate_angle(cx, frame.shape[1])
            # Отображение результатов
            cv2.putText(frame, f"Distance: {average_distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            cv2.putText(frame, f"Angle: {angle:.2f}deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Рисование видимых угловых точек
        for pt in visible_outer_corners:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        for pt in visible_inner_corners:
            cv2.circle(frame, pt, 5, (255, 0, 0), -1)

            # Показ изображения
        cv2.imshow("Frame", frame)

        # Выход из программы по нажатию "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
