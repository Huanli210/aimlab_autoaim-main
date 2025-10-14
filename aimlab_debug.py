import cv2
import numpy as np
import win32gui
import win32api
from mss import mss
import config
import logging
import time
from pynput import mouse
import torch

class BoxInfo:
    def __init__(self, box, distance, confidence):
        self.box = box
        self.distance = distance
        self.confidence = confidence

# 获取窗口区域
def capture_screen(region_width, region_height):
    # 获取屏幕大小
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    # 计算屏幕中心
    screen_center_x = screen_width // 2
    screen_center_y = screen_height // 2

    # 计算截取区域的左上角坐标
    left = screen_center_x - region_width // 2
    top = screen_center_y - region_height // 2

    # 使用 mss 截取指定区域
    with mss() as sct:
        monitor = {"left": left, "top": top, "width": region_width, "height": region_height}
        img = sct.grab(monitor)  # 截取屏幕区域
        frame = np.array(img)  # 转换为 numpy 数组
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 转换为 BGR 格式
        return frame

# 加载YOLOv5模型
def load_yolo_model(model_path):
    # 清理路径字符串，去除可能存在的多余空格和引号
    cleaned_path = model_path.strip().strip('\'"')
    
    try:
        # 推荐使用 torch.hub 加载本地 YOLOv5 模型
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=cleaned_path)
        logging.info(f"YOLOv5 model loaded successfully from {cleaned_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLOv5 model with torch.hub: {e}")
        # 如果 torch.hub 加载失败，尝试使用 ultralytics YOLO 接口
        try:
            from ultralytics import YOLO
            model = YOLO(cleaned_path)
            logging.info(f"Successfully loaded model with ultralytics YOLO interface from {cleaned_path}")
            return model
        except Exception as e2:
            logging.error(f"Failed to load model with ultralytics YOLO interface: {e2}")
            return None

# 获取屏幕中心点坐标
def get_screen_center(img):
    screen_center_x = img.shape[1] // 2
    screen_center_y = img.shape[0] // 2
    return screen_center_x, screen_center_y

# 使用YOLO检测目标
def detector_yolo(frame, model, screen_center_x, screen_center_y):
    results = model(frame)
    closest_box_info = None
    closest_distance = float('inf')

    # 根据 `results` 的类型来处理返回结果
    # 兼容 torch.hub 加载的 YOLOv5 模型
    if hasattr(results, 'pandas'):
        predictions = results.pandas().xyxy[0]
        for _, row in predictions.iterrows():
            confidence = row['confidence']
            if confidence < config.confidence_threshold:
                continue

            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5

            if distance < closest_distance:
                closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                closest_distance = distance
    # 兼容 ultralytics YOLO() 接口
    else:
         for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                if confidence < config.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5

                if distance < closest_distance:
                    closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                    closest_distance = distance

    return closest_box_info

# 火控
def should_fire(img, fire_switch, screen_center_y, screen_center_x, fire_k, closest_box_info):
    if fire_switch == 0:
        if closest_box_info and closest_box_info.distance < fire_k:
            logging.info("Target detected within fire range, firing.")
            click_mouse_lift()
            return True
        else:
            return False
    elif fire_switch == 1:
        center_pixel_value = img[screen_center_y, screen_center_x]
        is_center_white = np.array_equal(center_pixel_value, [255, 255, 255])
        
        if is_center_white:
            logging.info("Fire detected at center point.")
            click_mouse_lift()
            return True
        else:
            return False

# 调试部分
def debug_yolo(frame, closest_box_info, screen_center_x, screen_center_y, delay_time):
    if closest_box_info:
        x, y, w, h = closest_box_info.box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # 显示置信度
        if hasattr(closest_box_info, 'confidence') and closest_box_info.confidence is not None:
            confidence_text = f"{closest_box_info.confidence:.2f}"
            cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(frame, (screen_center_x, screen_center_y), 5, (0, 255, 0), -1)

    delay_text = f"Delay: {delay_time:.2f} ms"
    cv2.putText(frame, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO Detection", frame)

# 鼠标移动控制
def control_mouse_move(closest_box_info, screen_center_x, screen_center_y ):
    global controlling_mouse

    if closest_box_info:
        target_x_frame = closest_box_info.box[0]
        target_y_frame = closest_box_info.box[1]
        vector_x = target_x_frame - screen_center_x
        vector_y = target_y_frame - screen_center_y

        threshold = 2

        if closest_box_info.distance > threshold:
            # 使用平滑係數來計算移動距離
            move_x = vector_x * config.mouse_smoothing
            move_y = vector_y * config.mouse_smoothing
            move_mouse_by(move_x, move_y)
        else:
            click_mouse_lift()
            logging.info("distance, clicking mouse")
    else:
        pass

# 鼠标移动控制
def move_mouse_by(delta_x, delta_y):
    config.driver_mouse_control.move_R(int(delta_x), int(delta_y))

# 鼠标左键点击
def click_mouse_lift():
    config.driver_mouse_control.click_Left_down()
    config.driver_mouse_control.click_Left_up()

# 鼠标右键点击
def click_mouse_right(x, y, button, pressed):
    global controlling_mouse
    if button == mouse.Button.right and pressed:
        logging.info("Right mouse button pressed. Stopping detection.")
        controlling_mouse = False

# 启动鼠标监听器
def start_mouse_listener():
    listener = mouse.Listener(on_click=click_mouse_right)
    listener.start()

def aimlab_debug():
    logging.info("aimlab_debug with YOLO start ... ")

    model = load_yolo_model(config.yolo_model_path)
    if model is None:
        logging.error("Failed to load YOLO model. Exiting.")
        return

    global controlling_mouse
    controlling_mouse = True

    while True:
        start_time = time.perf_counter()

        frame = capture_screen(config.roi_width, config.roi_height)
        if frame is None:
            continue

        screen_center_x, screen_center_y = get_screen_center(frame)

        closest_box_info = detector_yolo(frame, model, screen_center_x, screen_center_y)
        
        if closest_box_info:
            control_mouse_move(closest_box_info, screen_center_x, screen_center_y)
            should_fire(frame, config.fire_switch, screen_center_y, screen_center_x, config.fire_k, closest_box_info)
        
        end_time = time.perf_counter()

        delay_time = (end_time - start_time) * 1000
        
        debug_yolo(frame, closest_box_info, screen_center_x, screen_center_y, delay_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()