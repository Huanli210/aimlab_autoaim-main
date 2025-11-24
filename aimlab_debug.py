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
        self.integral_max = 50  # 積分飽和限制

    def update(self, current_value):
        error = self.set_point - current_value
        self.integral += error
        # 限制積分值以防止飽和
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
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

class TrackingState:
    """管理目標追蹤狀態。"""
    def __init__(self, max_lost_frames=30):
        self.is_tracking = False
        self.last_target_box = None
        self.lost_frames = 0
        self.max_lost_frames = max_lost_frames  # 最多丟失多少幀後才放棄追蹤
        self.tracking_start_time = None
        self.last_detection_time = None
    
    def update_detection(self, target_box):
        """更新檢測結果。"""
        self.last_target_box = target_box
        self.is_tracking = True
        self.lost_frames = 0
        self.last_detection_time = time.time()
        if self.tracking_start_time is None:
            self.tracking_start_time = time.time()
    
    def update_no_detection(self):
        """沒有檢測到目標時的更新。"""
        self.lost_frames += 1
        if self.lost_frames >= self.max_lost_frames:
            self.is_tracking = False
            self.last_target_box = None
            self.tracking_start_time = None
    
    def reset(self):
        """重置追蹤狀態。"""
        self.is_tracking = False
        self.last_target_box = None
        self.lost_frames = 0
        self.tracking_start_time = None
        self.last_detection_time = None
    
    def get_tracking_confidence(self):
        """取得追蹤信心度 (基於連續追蹤的時間)。"""
        if self.tracking_start_time is None:
            return 0.0
        elapsed = time.time() - self.tracking_start_time
        return min(elapsed / 1.0, 1.0)  # 1 秒內達到最大信心度

# --- 全域變數 ---
running = False
frame_queue = queue.Queue(maxsize=2)
detection_paused = False  # 檢測暫停標誌
preview_enabled = False  # 預覽視窗顯示標誌
pid_x = None
pid_y = None

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

def detector_yolo(frame, model, screen_center_x, screen_center_y, tracking_state, search_radius=None):
    """偵測物件，並對追蹤中的目標套用「黏滯性」以防止準心抖動。
    
    Args:
        frame: 輸入影格
        model: YOLO 模型
        screen_center_x: 螢幕中心 X 座標
        screen_center_y: 螢幕中心 Y 座標
        tracking_state: 追蹤狀態對象
        search_radius: 搜索半徑。若為 None，則使用整個畫面。
    
    Returns:
        檢測到的目標資訊 (BoxInfo)，或 None
    """
    results = model(frame)
    closest_box_info = None
    closest_distance = float('inf')

    # 定義黏滯性閾值
    STICKY_THRESHOLD = 50 
    frame_height, frame_width = frame.shape[:2]

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

            # 如果正在追蹤目標，則套用黏滯性
            if tracking_state.is_tracking and tracking_state.last_target_box is not None:
                last_center_x, last_center_y, _, _ = tracking_state.last_target_box
                dist_to_last = ((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2) ** 0.5
                if dist_to_last < STICKY_THRESHOLD:
                    # 根據追蹤信心度調整黏滯性強度
                    confidence_factor = tracking_state.get_tracking_confidence()
                    stickiness = config.target_stickiness * (1.0 - confidence_factor * 0.5)
                    distance *= stickiness

            # 檢查是否在搜索半徑內
            if search_radius is not None:
                if distance > search_radius:
                    continue

            if distance < closest_distance:
                closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                closest_distance = distance

    if hasattr(results, 'pandas'): # torch.hub 模型
        process_predictions(results.pandas().xyxy[0])
    else: # ultralytics 模型
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

                if tracking_state.is_tracking and tracking_state.last_target_box is not None:
                    last_center_x, last_center_y, _, _ = tracking_state.last_target_box
                    dist_to_last = ((center_x - last_center_x) ** 2 + (center_y - last_center_y) ** 2) ** 0.5
                    if dist_to_last < STICKY_THRESHOLD:
                        confidence_factor = tracking_state.get_tracking_confidence()
                        stickiness = config.target_stickiness * (1.0 - confidence_factor * 0.5)
                        distance *= stickiness

                if search_radius is not None:
                    if distance > search_radius:
                        continue

                if distance < closest_distance:
                    closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                    closest_distance = distance

    return closest_box_info

def debug_yolo(frame, closest_box_info, screen_center_x, screen_center_y, delay_time, frame_count):
    """顯示帶有偵測資訊的除錯視窗。使用幀計數控制更新頻率。"""
    # 檢查預覽是否開啟
    if not preview_enabled:
        return
    
    # 延遲建立視窗（只在預覽開啟時）
    try:
        existing_window = cv2.getWindowProperty("YOLO detection", cv2.WND_PROP_VISIBLE)
        if existing_window < 0:  # 視窗不存在
            cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("YOLO detection", cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        try:
            cv2.namedWindow("YOLO detection", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("YOLO detection", cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
    
    # 每 3 幀才更新一次視窗，以提高流暢度
    if frame_count % 3 != 0:
        return
    
    display_frame = frame.copy() if config.debug_window_scale != 1.0 else frame
    
    # 只繪製準星圓點，不繪製目標框和目標紅點
    cv2.circle(display_frame, (screen_center_x, screen_center_y), 5, (0, 255, 0), -1)
    delay_text = f"delay: {delay_time:.2f} ms"
    cv2.putText(display_frame, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 縮放畫面用於顯示
    if config.debug_window_scale != 1.0:
        height, width = display_frame.shape[:2]
        new_width = int(width * config.debug_window_scale)
        new_height = int(height * config.debug_window_scale)
        display_frame = cv2.resize(display_frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    cv2.imshow("YOLO detection", display_frame)

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

        # 設定死區（Deadzone）：在這個範圍內不移動，以確保準心能精確停留
        deadzone = 1.5
        
        if abs(error_x) < deadzone:
            error_x = 0
        if abs(error_y) < deadzone:
            error_y = 0

        # PID 控制器的設定點為 0，因為我們希望將誤差（距離）最小化
        # 我們將負誤差傳遞給 PID 更新函式，因為我們希望滑鼠朝著目標移動
        # 從而有效地減少誤差。
        move_x = pid_x.update(-error_x)
        move_y = pid_y.update(-error_y)

        threshold = 1.5  # 降低閾值以提高精準度
        if closest_box_info.distance > threshold:
            # 套用整體平滑因子
            final_move_x = move_x * config.pid_smooth
            final_move_y = move_y * config.pid_smooth
            move_mouse_by(final_move_x, final_move_y)

# --- 多執行緒與主迴圈 ---

def on_press(key):
    """pynput 監聽器，根據配置文件中的按鍵設置。"""
    global running, detection_paused
    try:
        # 檢查退出鍵
        if config.exit_key.lower() == "end":
            if key == keyboard.Key.end:
                logging.warning(f"偵測到 End 鍵。正在關閉程式。")
                running = False
                return False
        elif config.exit_key.lower() == "esc":
            if key == keyboard.Key.esc:
                logging.warning(f"偵測到 ESC 鍵。正在關閉程式。")
                running = False
                return False
        
        # 檢查暫停鍵
        if hasattr(key, 'char'):
            if key.char and key.char.lower() == config.pause_key.lower():
                detection_paused = not detection_paused
                status = "已暫停" if detection_paused else "已恢復"
                logging.info(f"檢測已{status}。")
    except AttributeError:
        pass

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
    持續追蹤模式：即使目標暫時丟失，也會繼續搜索。
    """
    global running, pid_x, pid_y
    logging.info("YOLO 執行緒已啟動 (持續追蹤模式)。")

    # 初始化 PID 控制器
    pid_x = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    pid_y = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    
    # 初始化追蹤狀態 (20 幀以平衡連續性與切換)
    tracking_state = TrackingState(max_lost_frames=20)
    frame_count = 0
    window_created = False  # 追蹤視窗是否已建立

    while running:
        try:
            start_time = time.perf_counter()
            frame = frame_queue.get(timeout=1)
            frame_count += 1
            
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

            # 4. 進行偵測 (如果未暫停)
            detection_center_x = config.roi_width // 2
            detection_center_y = config.roi_height // 2
            
            if detection_paused:
                # 檢測暫停：跳過 YOLO 偵測
                local_box_info = None
            else:
                # 正常進行偵測
                local_box_info = detector_yolo(
                    detection_frame, 
                    model, 
                    detection_center_x, 
                    detection_center_y, 
                    tracking_state,
                    search_radius=None
                )
            
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
                
                # 更新追蹤狀態
                tracking_state.update_detection(local_box_info.box)
                logging.debug(f"目標已檢測到，距離: {global_distance:.2f}px，追蹤幀數: {tracking_state.lost_frames}")
            else:
                # 沒有找到新目標
                if tracking_state.is_tracking:
                    # 追蹤中但暫時丟失：保持最後已知位置進行預測移動
                    tracking_state.update_no_detection()
                    if tracking_state.last_target_box:
                        # 使用最後已知位置
                        last_box = tracking_state.last_target_box
                        global_box_coords = (
                            last_box[0] + roi_left,
                            last_box[1] + roi_top,
                            last_box[2],
                            last_box[3]
                        )
                        global_distance = ((global_box_coords[0] - aim_center_x) ** 2 + (global_box_coords[1] - aim_center_y) ** 2) ** 0.5
                        global_box_info = BoxInfo(global_box_coords, global_distance, 0.5)  # 信心度降低
                        logging.debug(f"目標丟失，使用預測位置。丟失幀數: {tracking_state.lost_frames}/{tracking_state.max_lost_frames}")
                else:
                    # 未追蹤狀態下沒有找到目標
                    pass

            # 6. 使用全域座標控制滑鼠與開火
            if config.control_mose and global_box_info:
                control_mouse_move(global_box_info, aim_center_x, aim_center_y, pid_x, pid_y)
                # 只在有高信心度的檢測時才開火
                if local_box_info:  # 只有實際偵測時才開火
                    handle_firing(global_box_info, aim_center_x, aim_center_y)
            else:
                # 沒有目標時重置 PID 控制器
                pid_x.reset()
                pid_y.reset()
            
            end_time = time.perf_counter()
            delay_time = (end_time - start_time) * 1000

            # 7. 在完整畫面上顯示除錯資訊
            if config.debug:
                cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 2) # 黃色 ROI 框
                
                # 顯示追蹤狀態
                if detection_paused:
                    status_text = "PAUSED"
                    status_color = (0, 0, 255)  # 紅色表示暫停
                elif tracking_state.is_tracking:
                    status_text = f"TRACKING (lost: {tracking_state.lost_frames}/{tracking_state.max_lost_frames})"
                    status_color = (0, 255, 0)
                else:
                    status_text = "IDLE"
                    status_color = (0, 165, 255)
                cv2.putText(frame, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                debug_yolo(frame, global_box_info, aim_center_x, aim_center_y, delay_time, frame_count)
                cv2.waitKey(1)
                    
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
