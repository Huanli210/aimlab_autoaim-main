# Aim Lab 自動瞄準輔助

本專案為針對遊戲 Aim Lab 設計的自動瞄準輔助工具，採用 YOLOv5 模型進行即時目標偵測，並透過 PID 控制器實現流暢的滑鼠移動與可選的自動開火功能。

## 主要功能

*   **即時目標偵測：** 運用 YOLOv5 模型，精準辨識畫面中的射擊目標。
*   **自動平滑瞄準：** 透過 PID 控制器，模擬人類玩家的瞄準習慣，實現平滑、自然的滑鼠軌跡。
*   **可選自動開火：** 提供多種開火模式，可設定為全自動或由玩家手動控制。
*   **目標黏滯性：** 當多個目標靠近時，能穩定準星，避免在不同目標間頻繁抖動。
*   **高度可配置性：** 所有關鍵參數皆可透過 `config/config.yaml` 檔案進行客製化，方便使用者微調。
*   **視覺化除錯：** 內建除錯模式，可即時顯示偵測框、中心點及延遲等資訊，方便開發與調整。
*   **高效能擷取：** 採用 `dxcam` 進行硬體加速螢幕擷取，並結合多執行緒架構，確保低延遲與高效能。
*   **底層滑鼠控制：** 直接呼叫自訂的 C++ DLL 函式庫來模擬滑鼠移動，提高隱蔽性與反應速度。

## 環境需求

*   作業系統：Windows
*   Python 版本：3.x
*   必要的 Python 套件：
    *   `numpy`
    *   `opencv-python`
    *   `win32gui`
    *   `pynput`
    *   `torch`
    *   `dxcam`
    *   `pyyaml`
    *   `ultralytics`

## 使用說明

1.  **參數設定：**
    *   開啟 `config/config.yaml` 檔案，根據您的需求調整內部參數。
    *   請務必確認 `yolo_model_path` 指向您已訓練好的 YOLOv5 模型檔案 (`.pt` 格式)。
2.  **啟動遊戲：**
    *   開啟 Aim Lab 並選擇任一訓練模式 (例如 Gridshot)。
3.  **執行程式：**
    *   在專案根目錄下，執行 `aimlab_start.py`：
        ```bash
        python aimlab_start.py
        ```
    *   程式啟動後會有 3 秒延遲，讓您有時間切換回遊戲視窗。
4.  **停止程式：**
    *   在程式執行期間，隨時可按下 `ESC` 鍵來安全地終止程式。
    *   若滑鼠失去控制，可使用 `Ctrl + Shift + Esc` 開啟工作管理員來中斷程式。

## 參數詳解 (`config.yaml`)

`config/config.yaml` 檔案包含了所有可調整的參數。

*   **`logging.level`**: 指定日誌輸出的詳細程度 (例如 `INFO`, `DEBUG`)。
*   **`device.path`**: `MouseControl.dll` 檔案的相對路徑。
*   **`settings.WINDOW_TITLE`**: Aim Lab 遊戲視窗的完整標題。
*   **`settings.roi_width` / `settings.roi_height`**: 畫面中心偵測區域的寬度與高度 (單位：像素)。程式只會在此區域內偵測目標。
*   **`settings.debug`**: 設定為 `true` 以啟用視覺化除錯視窗。
*   **`settings.control_mose`**: 設定為 `true` 來啟用自動瞄準功能。
*   **`settings.enable_auto_fire`**: 設定為 `true` 來啟用自動開火，`false` 則為手動開火。
*   **`settings.fire_mode`**: 自動開火模式。`0` = 當準星進入目標偵測框內時開火。`1` = 當準星靠近目標中心一定範圍內時開火。
*   **`settings.yolo_model_path`**: YOLOv5 模型檔案 (`.pt`) 的絕對路徑。
*   **`settings.mouse_smoothing`**: 滑鼠移動的平滑係數，值越小移動越平滑。
*   **`settings.pid_kp`, `settings.pid_ki`, `settings.pid_kd`**: PID 控制器的 P (比例)、I (積分)、D (微分) 增益參數，用於微調瞄準的反應速度與穩定性。
*   **`settings.pid_smooth`**: PID 輸出移動向量的整體平滑因子，用於緩和最終的滑鼠移動。
*   **`settings.target_stickiness`**: 目標黏滯係數 (0.0 - 1.0)。數值越低，準星越容易「黏」在當前目標上，避免在多個鄰近目標間晃動。

## 免責聲明

本專案僅供學術研究與教育目的使用。在任何線上遊戲中使用此類工具都可能違反服務條款，並有導致帳號被封鎖的風險。使用者需自行承擔所有後果。
