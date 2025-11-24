import cv2
import numpy as np
import win32gui
import config
import logging
import time
import torch
import queue
import threading
from pynput import keyboard
import dxcam

# --- PID 控制器 ---
class PID:
    def __init__(self, Kp, Ki, Kd, set_point=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.set_point = set_point
        self.integral = 0
        self.last_error = 0

    def update(self, current_value):
        error = self.set_point - current_value
        self.integral += error
        derivative = error - self.last_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error
        return output

    def reset(self):
        self.integral = 0
        self.last_error = 0

# --- 輔助類別 ---
class BoxInfo:
    def __init__(self, box, distance, confidence):
        self.box = box
        self.distance = distance
        self.confidence = confidence

# --- 全域變數 ---
running = True
frame_queue = queue.Queue(maxsize=2)

# --- 核心函式 (保留與調整) ---
def load_yolo_model(model_path):
    """從指定路徑載入 YOLOv5 模型。"""
    cleaned_path = model_path.strip().strip('"')
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=cleaned_path, force_reload=True)
        logging.info(f"YOLOv5 模型成功從 {cleaned_path} 載入")
        return model
    except Exception as e:
        logging.error(f"使用 torch.hub 載入 YOLOv5 模型時發生錯誤: {e}")
        try:
            from ultralytics import YOLO
            model = YOLO(cleaned_path)
            logging.info(f"使用 ultralytics YOLO 介面成功從 {cleaned_path} 載入模型")
            return model
        except Exception as e2:
            logging.error(f"使用 ultralytics YOLO 介面載入模型失敗: {e2}")
            return None

def detector_yolo(frame, model, screen_center_x, screen_center_y, last_target_box=None):
    """偵測物件，並對最後鎖定的目標套用「黏滯性」以防止準心抖動。"""
    results = model(frame)
    closest_box_info = None
    closest_distance = float('inf')

    # 定義一個閾值，用於判斷目標是否與前一個目標「相同」
    # 此處基於偵測框中心的距離。50 像素是一個初始參考值。
    STICKY_THRESHOLD = 50 

    def process_predictions(predictions):
        nonlocal closest_box_info, closest_distance
        for _, row in predictions.iterrows():
            confidence = row['confidence']
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5

            # 如果此目標與最後鎖定的目標很近，則套用黏滯係數
            if last_target_box is not None:
                last_center_x, last_center_y, _, _ = last_target_box
                dist_to_last = ((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2) ** 0.5
                if dist_to_last < STICKY_THRESHOLD:
                    distance *= config.target_stickiness

            if distance < closest_distance:
                closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                closest_distance = distance

    if hasattr(results, 'pandas'): # torch.hub 模型
        process_predictions(results.pandas().xyxy[0])
    else: # ultralytics 模型
        # 這部分需要針對 ultralytics 的結果格式進行調整才能應用黏滯性
        # 目前主要針對 pandas 格式實現了黏滯性。
         for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1

                distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5

                if last_target_box is not None:
                    last_center_x, last_center_y, _, _ = last_target_box
                    dist_to_last = ((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2) ** 0.5
                    if dist_to_last < STICKY_THRESHOLD:
                        distance *= config.target_stickiness

                if distance < closest_distance:
                    closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                    closest_distance = distance

    return closest_box_info

def debug_yolo(frame, closest_box_info, screen_center_x, screen_center_y, delay_time):
    """顯示帶有偵測資訊的除錯視窗。"""
    if closest_box_info:
        x, y, w, h = closest_box_info.box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        if hasattr(closest_box_info, 'confidence') and closest_box_info.confidence is not None:
            confidence_text = f"{closest_box_info.confidence:.2f}"
            cv2.putText(frame, confidence_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(frame, (screen_center_x, screen_center_y), 5, (0, 255, 0), -1)
    delay_text = f"delay: {delay_time:.2f} ms"
    cv2.putText(frame, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 根據 debug_window_scale 縮放畫面
    if config.debug_window_scale != 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * config.debug_window_scale)
        new_height = int(height * config.debug_window_scale)
        scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("YOLO detection", scaled_frame)
    else:
        cv2.imshow("YOLO detection", frame)

# --- 滑鼠控制函式 (使用專案的 DLL) ---
def move_mouse_by(delta_x, delta_y):
    config.driver_mouse_control.move_R(int(delta_x), int(delta_y))

def click():
    """執行滑鼠左鍵點擊。"""
    # DLL 有一個名為 'mouse_click' 的函式，1 為按下，0 為釋放
    config.driver_mouse_control.mouse_click(1)
    time.sleep(0.01) # 模擬點擊延遲
    config.driver_mouse_control.mouse_click(0)

def handle_firing(closest_box_info, screen_center_x, screen_center_y):
    """根據設定的開火模式決定是否開火。"""
    if not config.enable_auto_fire or not closest_box_info:
        return

    target_x, target_y, target_w, target_h = closest_box_info.box

    if config.fire_mode == 0: # 模式0: 準星在目標框內
        if abs(target_x - screen_center_x) < target_w / 2 and \
           abs(target_y - screen_center_y) < target_h / 2:
            click()
    
    elif config.fire_mode == 1: # 模式1: 準星靠近目標中心
        fire_radius = 10  # 可設定的開火半徑
        if closest_box_info.distance < fire_radius:
            click()

def control_mouse_move(closest_box_info, screen_center_x, screen_center_y, pid_x, pid_y):
    """使用 PID 計算向量並將滑鼠移向目標。"""
    if closest_box_info:
        target_x_frame = closest_box_info.box[0]
        target_y_frame = closest_box_info.box[1]
        
        # 誤差是目標到畫面中心的距離
        error_x = target_x_frame - screen_center_x
        error_y = target_y_frame - screen_center_y

        # PID 控制器的設定點為 0，因為我們希望將誤差（距離）最小化
        # 我們將負誤差傳遞給 PID 更新函式，因為我們希望滑鼠朝著目標移動
        # 從而有效地減少誤差。
        move_x = pid_x.update(-error_x)
        move_y = pid_y.update(-error_y)

        threshold = 2
        if closest_box_info.distance > threshold:
            # 套用整體平滑因子
            final_move_x = move_x * config.pid_smooth
            final_move_y = move_y * config.pid_smooth
            move_mouse_by(final_move_x, final_move_y)

# --- 多執行緒與主迴圈 ---

def on_press(key):
    """pynput 監聽器，按下 ESC 鍵時停止程式。"""
    global running
    if key == keyboard.Key.esc:
        logging.warning("偵測到 ESC 鍵。正在關閉程式。")
        running = False
        return False

def capture_thread_func():
    """使用 dxcam 擷取遊戲視窗並將影格放入佇列。
    此函式在獨立的執行緒中執行。
    """
    global running
    logging.info("擷取執行緒已啟動。")
    
    try:
        hwnd = win32gui.FindWindow(None, config.WINDOW_TITLE)
        if not hwnd:
            logging.error(f"找不到視窗: '{config.WINDOW_TITLE}'。正在結束擷取執行緒。")
            running = False
            return
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        region = (left, top, right, bottom)
        
        camera = dxcam.create(region=region)
        camera.start(target_fps=60)
        logging.info(f"DXCAM 已為視窗 '{config.WINDOW_TITLE}' 啟動，區域為 {region}")
    except Exception as e:
        logging.error(f"初始化 DXCAM 失敗: {e}。正在結束擷取執行緒。")
        running = False
        return

    while running:
        frame = camera.get_latest_frame()
        if frame is None:
            time.sleep(0.001)
            continue
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame_bgr)
        
    camera.stop()
    logging.info("擷取執行緒已停止。")

def yolo_thread_func(model):
    """從佇列處理影格，在中央 ROI 上執行 YOLO 偵測，並控制滑鼠。
    此函式在獨立的執行緒中執行。
    """
    global running
    logging.info("YOLO 執行緒已啟動。")

    # 初始化 PID 控制器
    pid_x = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    pid_y = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    
    last_target_box = None # 用於儲存最後一個目標框的變數

    if config.debug:
        cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("YOLO detection", cv2.WND_PROP_TOPMOST, 1)

    while running:
        try:
            start_time = time.perf_counter()
            frame = frame_queue.get(timeout=1)
            
            # 1. 定義瞄準中心 (全畫面的中心)
            aim_center_x = frame.shape[1] // 2
            aim_center_y = frame.shape[0] // 2

            # 2. 定義偵測的投資回報率 (ROI)
            roi_left = aim_center_x - config.roi_width // 2
            roi_top = aim_center_y - config.roi_height // 2
            roi_right = roi_left + config.roi_width
            roi_bottom = roi_top + config.roi_height
            
            # 3. 將影格裁剪至 ROI
            detection_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

            # 4. 在裁剪後的影格中進行偵測，並套用黏滯性
            detection_center_x = config.roi_width // 2
            detection_center_y = config.roi_height // 2
            local_box_info = detector_yolo(detection_frame, model, detection_center_x, detection_center_y, last_target_box)
            
            global_box_info = None
            if local_box_info:
                # 5. 將座標轉換回全畫面的空間
                global_box_coords = (
                    local_box_info.box[0] + roi_left, # center_x
                    local_box_info.box[1] + roi_top,  # center_y
                    local_box_info.box[2],            # width
                    local_box_info.box[3]             # height
                )
                global_distance = ((global_box_coords[0] - aim_center_x) ** 2 + (global_box_coords[1] - aim_center_y) ** 2) ** 0.5
                global_box_info = BoxInfo(global_box_coords, global_distance, local_box_info.confidence)
                
                # 使用找到的目標的局部座標更新 last_target_box
                last_target_box = local_box_info.box
            else:
                # 如果沒有找到目標，則重置 last_target_box
                last_target_box = None

            # 6. 使用全域座標控制滑鼠與開火
            if config.control_mose and global_box_info:
                control_mouse_move(global_box_info, aim_center_x, aim_center_y, pid_x, pid_y)
                handle_firing(global_box_info, aim_center_x, aim_center_y)
            else:
                # 如果沒有找到目標，則重置 PID 控制器
                pid_x.reset()
                pid_y.reset()
            
            end_time = time.perf_counter()
            delay_time = (end_time - start_time) * 1000

            # 7. 在完整畫面上顯示除錯資訊
            if config.debug:
                cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 2) # 黃色 ROI 框
                debug_yolo(frame, global_box_info, aim_center_x, aim_center_y, delay_time)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
                    
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"YOLO 執行緒發生錯誤: {e}", exc_info=True)
            time.sleep(1)

    if config.debug:
        cv2.destroyAllWindows()
    logging.info("YOLO 執行緒已停止。")

def aimlab_debug():
    """初始化並執行所有執行緒的主函式。"""
    logging.basicConfig(level=config.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("YOLO 偵錯模式啟動中...")

    model = load_yolo_model(config.yolo_model_path)
    if model is None:
        logging.error("載入 YOLO 模型失敗。正在結束程式。")
        return

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    logging.info("執行緒將在 3 秒後啟動... 按下 ESC 鍵可停止程式。")
    time.sleep(3)

    capture_proc = threading.Thread(target=capture_thread_func, daemon=True)
    yolo_proc = threading.Thread(target=yolo_thread_func, args=(model,), daemon=True)
    
    capture_proc.start()
    yolo_proc.start()

    try:
        yolo_proc.join()
    except KeyboardInterrupt:
        logging.warning("收到鍵盤中斷訊號。正在關閉程式。")
        global running
        running = False
    
    listener.stop()
    logging.info("程式已關閉。")
