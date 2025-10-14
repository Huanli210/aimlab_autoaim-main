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

# --- PID Controller ---
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

# --- Helper Class ---
class BoxInfo:
    def __init__(self, box, distance, confidence):
        self.box = box
        self.distance = distance
        self.confidence = confidence

# --- Global Variables ---
running = True
frame_queue = queue.Queue(maxsize=2)

# --- Core Functions (Retained and Adapted) ---
def load_yolo_model(model_path):
    """Loads the YOLOv5 model from the specified path."""
    cleaned_path = model_path.strip().strip('"')
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=cleaned_path, force_reload=True)
        logging.info(f"YOLOv5 model loaded successfully from {cleaned_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading YOLOv5 model with torch.hub: {e}")
        try:
            from ultralytics import YOLO
            model = YOLO(cleaned_path)
            logging.info(f"Successfully loaded model with ultralytics YOLO interface from {cleaned_path}")
            return model
        except Exception as e2:
            logging.error(f"Failed to load model with ultralytics YOLO interface: {e2}")
            return None

def detector_yolo(frame, model, screen_center_x, screen_center_y):
    """Detects objects in the frame using YOLO and finds the closest one to the center."""
    results = model(frame)
    closest_box_info = None
    closest_distance = float('inf')

    # Process results based on model type
    if hasattr(results, 'pandas'): # torch.hub model
        predictions = results.pandas().xyxy[0]
        for _, row in predictions.iterrows():
            confidence = row['confidence']
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            distance = ((center_x - screen_center_x) ** 2 + (center_y - screen_center_y) ** 2) ** 0.5

            if distance < closest_distance:
                closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                closest_distance = distance
    else: # ultralytics model
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

                if distance < closest_distance:
                    closest_box_info = BoxInfo((center_x, center_y, width, height), distance, confidence)
                    closest_distance = distance

    return closest_box_info

def debug_yolo(frame, closest_box_info, screen_center_x, screen_center_y, delay_time):
    """Displays the debug window with detection info."""
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
    delay_text = f"Delay: {delay_time:.2f} ms"
    cv2.putText(frame, delay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("YOLO Detection", frame)

# --- Mouse Control Functions (Using project's DLL) ---
def move_mouse_by(delta_x, delta_y):
    config.driver_mouse_control.move_R(int(delta_x), int(delta_y))

def click_mouse_lift():
    config.driver_mouse_control.click_Left_down()
    config.driver_mouse_control.click_Left_up()

def control_mouse_move(closest_box_info, screen_center_x, screen_center_y, pid_x, pid_y):
    """Calculates vector and moves the mouse towards the target using PID."""
    if closest_box_info:
        target_x_frame = closest_box_info.box[0]
        target_y_frame = closest_box_info.box[1]
        
        # The error is the distance from the target to the screen center
        error_x = target_x_frame - screen_center_x
        error_y = target_y_frame - screen_center_y

        # The PID controller's set_point is 0, as we want to minimize the error (distance)
        # We pass the negative error to the PID update function because we want to move the mouse
        # towards the target, effectively reducing the error.
        move_x = pid_x.update(-error_x)
        move_y = pid_y.update(-error_y)

        threshold = 2
        if closest_box_info.distance > threshold:
            # Apply an overall smoothing factor
            final_move_x = move_x * config.pid_smooth
            final_move_y = move_y * config.pid_smooth
            move_mouse_by(final_move_x, final_move_y)
        else:
            click_mouse_lift()
            logging.info("Target within click threshold, clicking mouse.")

def should_fire(img, fire_switch, screen_center_y, screen_center_x, fire_k, closest_box_info):
    """Determines if the conditions to fire are met."""
    if fire_switch == 0: # Edge-based firing
        if closest_box_info and closest_box_info.distance < fire_k:
            logging.info("Target detected within fire range, firing.")
            click_mouse_lift()
            return True
    elif fire_switch == 1: # Center-pixel-based firing
        center_pixel_value = img[screen_center_y, screen_center_x]
        if np.array_equal(center_pixel_value, [255, 255, 255]):
            logging.info("Fire condition met at center point.")
            click_mouse_lift()
            return True
    return False

# --- Threading and Main Loop ---
def on_press(key):
    """pynput listener to stop the program on ESC key press."""
    global running
    if key == keyboard.Key.esc:
        logging.warning("ESC key detected. Shutting down.")
        running = False
        return False

def capture_thread_func():
    """
    Captures the game window using dxcam and puts frames into a queue.
    This runs in a separate thread.
    """
    global running
    logging.info("Capture thread started.")
    
    try:
        hwnd = win32gui.FindWindow(None, config.WINDOW_TITLE)
        if not hwnd:
            logging.error(f"Could not find window: '{config.WINDOW_TITLE}'. Exiting capture thread.")
            running = False
            return
        
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        region = (left, top, right, bottom)
        
        camera = dxcam.create(region=region)
        camera.start(target_fps=60)
        logging.info(f"DXCAM started for window '{config.WINDOW_TITLE}' with region {region}")
    except Exception as e:
        logging.error(f"Failed to initialize DXCAM: {e}. Exiting capture thread.")
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
    logging.info("Capture thread stopped.")

def yolo_thread_func(model):
    """
    Processes frames from the queue, runs YOLO detection on a central ROI, and controls the mouse.
    This runs in a separate thread.
    """
    global running
    logging.info("YOLO thread started.")

    # Initialize PID controllers
    pid_x = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    pid_y = PID(Kp=config.pid_kp, Ki=config.pid_ki, Kd=config.pid_kd)
    
    if config.debug:
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("YOLO Detection", cv2.WND_PROP_TOPMOST, 1)

    while running:
        try:
            start_time = time.perf_counter()
            frame = frame_queue.get(timeout=1)
            
            # 1. Define Aiming Center (center of full frame)
            aim_center_x = frame.shape[1] // 2
            aim_center_y = frame.shape[0] // 2

            # 2. Define ROI for detection
            roi_left = aim_center_x - config.roi_width // 2
            roi_top = aim_center_y - config.roi_height // 2
            roi_right = roi_left + config.roi_width
            roi_bottom = roi_top + config.roi_height
            
            # 3. Crop frame to ROI
            detection_frame = frame[roi_top:roi_bottom, roi_left:roi_right]

            # 4. Detect in the cropped frame
            detection_center_x = config.roi_width // 2
            detection_center_y = config.roi_height // 2
            local_box_info = detector_yolo(detection_frame, model, detection_center_x, detection_center_y)
            
            global_box_info = None
            if local_box_info:
                # 5. Translate coordinates back to full frame space
                global_box_coords = (
                    local_box_info.box[0] + roi_left, # center_x
                    local_box_info.box[1] + roi_top,  # center_y
                    local_box_info.box[2],            # width
                    local_box_info.box[3]             # height
                )
                global_distance = ((global_box_coords[0] - aim_center_x) ** 2 + (global_box_coords[1] - aim_center_y) ** 2) ** 0.5
                global_box_info = BoxInfo(global_box_coords, global_distance, local_box_info.confidence)

            # 6. Control mouse using global coordinates
            if config.control_mose and global_box_info:
                control_mouse_move(global_box_info, aim_center_x, aim_center_y, pid_x, pid_y)
                should_fire(frame, config.fire_switch, aim_center_y, aim_center_x, config.fire_k, global_box_info)
            else:
                # Reset PID controllers if no target is found
                pid_x.reset()
                pid_y.reset()
            
            end_time = time.perf_counter()
            delay_time = (end_time - start_time) * 1000

            # 7. Debug display on the full frame
            if config.debug:
                cv2.rectangle(frame, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 255), 2) # Yellow ROI box
                debug_yolo(frame, global_box_info, aim_center_x, aim_center_y, delay_time)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break
                    
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"An error occurred in YOLO thread: {e}", exc_info=True)
            time.sleep(1)

    if config.debug:
        cv2.destroyAllWindows()
    logging.info("YOLO thread stopped.")

def aimlab_debug():
    """
    Main function to initialize and run all threads.
    """
    logging.basicConfig(level=config.log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("aimlab_debug with YOLO starting up...")

    model = load_yolo_model(config.yolo_model_path)
    if model is None:
        logging.error("Failed to load YOLO model. Exiting.")
        return

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    logging.info("Starting threads in 3 seconds... Press ESC to stop.")
    time.sleep(3)

    capture_proc = threading.Thread(target=capture_thread_func, daemon=True)
    yolo_proc = threading.Thread(target=yolo_thread_func, args=(model,), daemon=True)
    
    capture_proc.start()
    yolo_proc.start()

    try:
        yolo_proc.join()
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received. Shutting down.")
        global running
        running = False
    
    listener.stop()
    logging.info("Program has shut down.")
