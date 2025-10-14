# Aim Lab 自動瞄準

本專案是針對遊戲 Aim Lab 的自動瞄準機器人，使用 OpenCV 和 YOLOv5 模型進行物件偵測。

## 功能

*   **物件偵測:** 使用 YOLOv5 模型偵測螢幕上的目標。
*   **自動瞄準:** 自動將滑鼠移動到偵測到的目標上。
*   **PID 控制器:** 利用 PID 控制器實現平滑且擬人化的滑鼠移動。
*   **目標黏滯性:** 防止在偵測到多個鄰近目標時準星抖動的功能。
*   **可配置:** 所有參數都可以透過 `config/config.yaml` 檔案進行配置。
*   **除錯模式:** 用於視覺化偵測過程的除錯模式。
*   **高效能:** 使用 `dxcam` 進行快速螢幕擷取，並使用多執行緒以防止效能問題。
*   **隱蔽的滑鼠控制:** 使用自訂的 DLL 進行滑鼠控制，以提高隱蔽性。

## 環境需求

*   Windows
*   Python 3
*   以下 Python 套件：
    *   `numpy`
    *   `opencv-python`
    *   `win32gui`
    *   `pynput`
    *   `mss`
    *   `torch`
    *   `dxcam`
    *   `pyyaml`
    *   `ultralytics`

## 使用方法

1.  **設定:**
    *   開啟 `config/config.yaml` 並根據需要調整參數。
    *   確保 `yolo_model_path` 指向您的 YOLOv5 模型檔案。
2.  **啟動 Aim Lab:**
    *   開啟 Aim Lab 並進入練習模式（例如 Gridshot）。
    *   按下 `ESC` 鍵暫停遊戲。
3.  **執行機器人:**
    *   執行 `aimlab_start.py` 指令碼：
        ```bash
        python aimlab_start.py
        ```
4.  **停止機器人:**
    *   按下 `ESC` 鍵停止指令碼。
    *   如果您失去滑鼠控制權，可以按 `Ctrl + Shift + Esc` 開啟工作管理員以重新獲得控制權。

## 參數設定

`config/config.yaml` 檔案包含機器人的所有參數。

*   **`logging.level`**: 日誌記錄等級（例如 `INFO`, `DEBUG`）。
*   **`device.path`**: `MouseControl.dll` 檔案的路徑。
*   **`settings.WINDOW_TITLE`**: Aim Lab 視窗的標題。
*   **`settings.roi_width` / `settings.roi_height`**: 物件偵測的感興趣區域（ROI）大小。
*   **`settings.debug`**: 設定為 `true` 以啟用除錯模式。
*   **`settings.control_mose`**: 設定為 `true` 以啟用滑鼠控制。
*   **`settings.fire_switch`**: 開火模式（0 為基於邊緣，1 為基於中心像素）。
*   **`settings.fire_k`**: 基於邊緣開火的開火範圍。
*   **`settings.yolo_model_path`**: 您的 YOLOv5 模型檔案的路徑。
*   **`settings.mouse_smoothing`**: 滑鼠平滑係數。
*   **`settings.pid_kp`, `settings.pid_ki`, `settings.pid_kd`**: PID 控制器參數。
*   **`settings.pid_smooth`**: PID 控制移動的整體平滑係數。
*   **`settings.target_stickiness`**: 使準星「黏住」目標的係數。

## 免責聲明

本專案僅供教育目的使用。在線上遊戲中使用此工具可能會導致帳號被封鎖。請自行承擔風險。
