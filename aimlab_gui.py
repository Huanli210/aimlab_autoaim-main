import threading
import tkinter as tk
from tkinter import ttk
import logging
import time
import yaml
from pathlib import Path

import aimlab_debug
import config


class AimlabGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AimLab AutoAim 控制面板")
        self.root.geometry("360x200")

        # 狀態欄
        self.status_var = tk.StringVar(value="狀態：停止")
        status_label = ttk.Label(self.root, textvariable=self.status_var)
        status_label.pack(pady=8)

        # 控制按鈕區
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=6)

        self.start_btn = ttk.Button(btn_frame, text="啟動模型", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=6)

        self.pause_btn = ttk.Button(btn_frame, text="暫停/繼續", command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=1, padx=6)

        self.stop_btn = ttk.Button(btn_frame, text="停止", command=self.stop)
        self.stop_btn.grid(row=0, column=2, padx=6)

        # 參數面板按鈕
        params_btn = ttk.Button(self.root, text="參數", command=self.open_params_window)
        params_btn.pack(pady=4)

        # 按鍵設置按鈕
        hotkeys_btn = ttk.Button(self.root, text="按鍵設置", command=self.open_hotkeys_window)
        hotkeys_btn.pack(pady=4)

        # 預覽切換開關
        self.preview_var = tk.BooleanVar(value=False)
        preview_check = ttk.Checkbutton(self.root, text="顯示預覽", variable=self.preview_var,
                                        command=self.toggle_preview)
        preview_check.pack(pady=2)

        # 退出按鈕
        quit_btn = ttk.Button(self.root, text="離開", command=self.quit)
        quit_btn.pack(pady=4)

        # 內部狀態
        self.worker_thread = None
        self._stop_requested = False
        self.params_window = None
        self.hotkeys_window = None
        self.auto_apply = tk.BooleanVar(value=True)

        # YAML 路徑
        self.yaml_path = Path(r"config\config.yaml")

        # 週期更新 UI
        self.root.after(200, self._update_status_loop)

    def start(self):
        # 如果已經在運行就不重複啟動
        if self.worker_thread and self.worker_thread.is_alive():
            logging.info("已在運行中，忽略 Start。")
            return

        # 初始化 aimlab_debug 全域旗標
        aimlab_debug.running = True
        aimlab_debug.detection_paused = False

        # 在背景執行 aimlab_debug.aimlab_debug()
        self.worker_thread = threading.Thread(target=aimlab_debug.aimlab_debug, daemon=True)
        self.worker_thread.start()
        self.status_var.set("狀態：運行")
        logging.info("已啟動偵測。")

    def stop(self):
        logging.info("停止請求中...")
        aimlab_debug.running = False
        aimlab_debug.detection_paused = False
        # 嘗試等待線程結束
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
        self.status_var.set("狀態：停止")

    def toggle_pause(self):
        # 切換暫停狀態
        current = getattr(aimlab_debug, 'detection_paused', False)
        aimlab_debug.detection_paused = not current
        st_cn = "暫停" if aimlab_debug.detection_paused else "運行"
        self.status_var.set(f"狀態：{st_cn}")
        logging.info(f"檢測已切換為: {st_cn}")

    def toggle_preview(self):
        # 切換預覽視窗顯示
        aimlab_debug.preview_enabled = self.preview_var.get()
        state = "開啟" if aimlab_debug.preview_enabled else "關閉"
        logging.info(f"預覽已{state}")

    # -------------------- 參數面板 --------------------
    def open_params_window(self):
        if self.params_window and tk.Toplevel.winfo_exists(self.params_window):
            self.params_window.lift()
            return

        self.params_window = tk.Toplevel(self.root)
        self.params_window.title("參數設定")
        self.params_window.geometry("720x400")

        # 讀取當前設定
        self.load_config_values()

        # 參數說明字典
        param_descriptions = {
            'pid_kp': '比例增益 (Proportional)：控制對目標偏差的直接反應速度。\n值越大回應越快，但容易過衝。',
            'pid_ki': '積分增益 (Integral)：消除穩態誤差，使滑鼠逐漸調整至目標。\n通常設置較小避免震盪。',
            'pid_kd': '微分增益 (Derivative)：預測並減少過衝，增加穩定性。\n有助於平滑移動。',
            'pid_smooth': 'PID 平滑係數：對 PID 輸出進行低通濾波。\n值越大移動越平順（0.0-1.0）。',
            'target_stickiness': '目標黏滯性：控制追蹤時對上次位置的依賴程度。\n值越高目標切換越慢，但追蹤更穩定。',
            'mouse_smoothing': '滑鼠平滑係數：對滑鼠移動進行平滑處理。\n值越大移動越流暢（0.0-1.0）。'
        }

        # 主容器：左邊參數列表，右邊說明欄
        main_frame = ttk.Frame(self.params_window)
        main_frame.pack(fill='both', expand=True, padx=8, pady=8)

        # 左側參數框
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))

        frame = ttk.Frame(left_frame)
        frame.pack(fill='both', expand=True)

        # 右側說明框
        help_frame = ttk.LabelFrame(main_frame, text="說明小幫手", padding=8)
        help_frame.pack(side='right', fill='both', padx=(8, 0))

        self.help_text = tk.Text(help_frame, height=15, width=30, wrap='word', state='disabled')
        self.help_text.pack(fill='both', expand=True)

        row = 0
        def add_scale(label_text, var_name, from_, to_, resolution, row_idx):
            label_widget = ttk.Label(frame, text=label_text)
            label_widget.grid(row=row_idx, column=0, sticky='w')
            var = tk.DoubleVar()
            scale = ttk.Scale(frame, variable=var, from_=from_, to=to_, orient='horizontal')
            scale.grid(row=row_idx, column=1, sticky='we', padx=6)
            frame.columnconfigure(1, weight=1)

            entry = ttk.Entry(frame, width=8)
            entry.grid(row=row_idx, column=2, sticky='e', padx=(6,0))

            # 初始化值
            init_val = float(getattr(config, var_name, 0.0))
            var.set(init_val)
            entry.delete(0, tk.END)
            entry.insert(0, f"{init_val:.3f}")

            def on_scale_change(*args):
                v = float(var.get())
                # 更新 entry 顯示
                entry.delete(0, tk.END)
                entry.insert(0, f"{v:.3f}")
                # 更新 config 變數
                setattr(config, var_name, v)
                if self.auto_apply.get():
                    self.apply_config({var_name: v})

            def on_mouse_enter(event=None):
                # 當滑鼠進入時更新說明欄
                self.update_help_text(param_descriptions.get(var_name, ''))

            def on_mouse_leave(event=None):
                # 當滑鼠離開時清空說明欄
                pass

            var.trace_add('write', on_scale_change)

            def on_entry_apply(event=None):
                try:
                    v = float(entry.get())
                except Exception:
                    v = float(getattr(config, var_name, init_val))
                # clamp value
                v = max(min(v, to_), from_)
                var.set(v)
                # apply immediately if auto_apply
                setattr(config, var_name, v)
                if self.auto_apply.get():
                    self.apply_config({var_name: v})

            entry.bind('<Return>', on_entry_apply)
            entry.bind('<FocusOut>', on_entry_apply)

            # 綁定滑鼠進入/離開事件以更新說明欄
            label_widget.bind('<Enter>', on_mouse_enter)
            scale.bind('<Enter>', on_mouse_enter)
            entry.bind('<Enter>', on_mouse_enter)

            return {'scale': scale, 'var': var, 'entry': entry}

        # PID 與其他參數
        r = 0
        self.s_kp = add_scale("PID Kp (比例)", 'pid_kp', 0.0, 2.0, 0.01, r); r+=1
        self.s_ki = add_scale("PID Ki (積分)", 'pid_ki', 0.0, 0.5, 0.001, r); r+=1
        self.s_kd = add_scale("PID Kd (微分)", 'pid_kd', 0.0, 1.0, 0.001, r); r+=1
        self.s_smooth = add_scale("PID 平滑", 'pid_smooth', 0.0, 1.0, 0.01, r); r+=1
        self.s_sticky = add_scale("目標黏滯性", 'target_stickiness', 0.0, 1.0, 0.01, r); r+=1
        self.s_mouse = add_scale("滑鼠平滑", 'mouse_smoothing', 0.0, 1.0, 0.01, r); r+=1

        ttk.Checkbutton(frame, text="自動套用", variable=self.auto_apply).grid(row=r, column=0, sticky='w')
        ttk.Button(frame, text="立即套用", command=self.apply_now).grid(row=r, column=1, sticky='e')
        r += 1

        # Reload / Restore defaults buttons (below the frame)
        btn_frame2 = ttk.Frame(self.params_window)
        btn_frame2.pack(fill='x', padx=8, pady=6)
        ttk.Button(btn_frame2, text="從 YAML 重新讀取", command=self.reload_from_yaml).pack(side='left', padx=6)
        ttk.Button(btn_frame2, text="還原預設", command=self.restore_defaults).pack(side='left', padx=6)

    def update_help_text(self, text):
        """更新說明欄文字"""
        self.help_text.config(state='normal')
        self.help_text.delete('1.0', 'end')
        self.help_text.insert('1.0', text)
        self.help_text.config(state='disabled')

    def load_config_values(self):
        # 從 YAML 讀取設定並記錄為原始預設，並將值套用到 UI
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.original_config = data.get('settings', {}).copy()
        except Exception:
            # fallback to config module
            self.original_config = {
                'pid_kp': getattr(config, 'pid_kp', 0.45),
                'pid_ki': getattr(config, 'pid_ki', 0.02),
                'pid_kd': getattr(config, 'pid_kd', 0.1),
                'pid_smooth': getattr(config, 'pid_smooth', 0.7),
                'target_stickiness': getattr(config, 'target_stickiness', 0.1),
                'mouse_smoothing': getattr(config, 'mouse_smoothing', 0.1)
            }

        # 如果已經有 scales，設定它們為當前值
        try:
            for attr in ('s_kp', 's_ki', 's_kd', 's_smooth', 's_sticky', 's_mouse'):
                obj = getattr(self, attr, None)
                if obj is None:
                    continue
                key = {
                    's_kp': 'pid_kp', 's_ki': 'pid_ki', 's_kd': 'pid_kd',
                    's_smooth': 'pid_smooth', 's_sticky': 'target_stickiness', 's_mouse': 'mouse_smoothing'
                }[attr]
                val = float(getattr(config, key, self.original_config.get(key, 0.0)))
                if isinstance(obj, dict):
                    obj['var'].set(val)
                    obj['entry'].delete(0, tk.END)
                    obj['entry'].insert(0, f"{val:.3f}")
                else:
                    try:
                        obj.set(val)
                    except Exception:
                        pass
        except Exception as e:
            logging.debug(f"更新 UI 參數值時發生錯誤: {e}")

    def apply_now(self):
        # collect values and apply
        kv = {
            'pid_kp': getattr(config, 'pid_kp', 0.45),
            'pid_ki': getattr(config, 'pid_ki', 0.02),
            'pid_kd': getattr(config, 'pid_kd', 0.1),
            'pid_smooth': getattr(config, 'pid_smooth', 0.7),
            'target_stickiness': getattr(config, 'target_stickiness', 0.1),
            'mouse_smoothing': getattr(config, 'mouse_smoothing', 0.1)
        }
        self.apply_config(kv)

    def reload_from_yaml(self):
        """從 YAML 重載設定並更新 UI。"""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            settings = data.get('settings', {})
            # 更新 config 與 ui
            for k, v in settings.items():
                if hasattr(config, k):
                    setattr(config, k, v)
            # 更新 UI 顯示
            self.load_config_values()
            logging.info("已從 YAML 重載設定並更新 UI。")
        except Exception as e:
            logging.error(f"重載 YAML 失敗: {e}")

    def restore_defaults(self):
        """將設定還原為啟動時讀取的預設值，並寫回 YAML。"""
        try:
            defaults = getattr(self, 'original_config', {})
            if not defaults:
                logging.warning("沒有可用的預設設定可還原。")
                return
            # apply to config
            for k, v in defaults.items():
                setattr(config, k, v)
            # write back
            self.apply_config(defaults)
            # update UI
            self.load_config_values()
            logging.info("已還原為預設設定。")
        except Exception as e:
            logging.error(f"還原預設失敗: {e}")

    def apply_config(self, kv: dict):
        """寫回 YAML 並更新 config 與嘗試即時更新 aimlab_debug 的 PID 實例。"""
        try:
            # 讀取完整 YAML
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # 更新 settings
            settings = data.get('settings', {})
            for k, v in kv.items():
                if k in settings:
                    settings[k] = v
                else:
                    # 也可能是直接 top-level in settings
                    settings[k] = v
            data['settings'] = settings

            # 寫回 YAML
            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)

            # 更新 config 模組屬性
            for k, v in kv.items():
                setattr(config, k, v)

            # 嘗試即時更新 aimlab_debug 中的 PID 實例（若存在）
            try:
                if hasattr(aimlab_debug, 'pid_x') and aimlab_debug.pid_x is not None:
                    aimlab_debug.pid_x.Kp = config.pid_kp
                    aimlab_debug.pid_x.Ki = config.pid_ki
                    aimlab_debug.pid_x.Kd = config.pid_kd
                if hasattr(aimlab_debug, 'pid_y') and aimlab_debug.pid_y is not None:
                    aimlab_debug.pid_y.Kp = config.pid_kp
                    aimlab_debug.pid_y.Ki = config.pid_ki
                    aimlab_debug.pid_y.Kd = config.pid_kd
            except Exception:
                logging.debug("即時更新 PID 實例失敗 (尚未建立或不可用)")

            logging.info("設定已寫回 YAML 並更新(config)。")
        except Exception as e:
            logging.error(f"寫回設定失敗: {e}")

    # -------------------- 按鍵設置面板 --------------------
    def open_hotkeys_window(self):
        if self.hotkeys_window and tk.Toplevel.winfo_exists(self.hotkeys_window):
            self.hotkeys_window.lift()
            return

        self.hotkeys_window = tk.Toplevel(self.root)
        self.hotkeys_window.title("按鍵設置")
        self.hotkeys_window.geometry("400x250")

        frame = ttk.Frame(self.hotkeys_window)
        frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 按鍵設置項目
        hotkeys_settings = [
            ('exit_key', '退出鍵', '關閉應用程式'),
            ('pause_key', '暫停鍵', '暫停/繼續檢測'),
        ]

        row = 0
        self.hotkey_entries = {}

        for config_key, label_cn, description in hotkeys_settings:
            # 標籤
            label = ttk.Label(frame, text=f"{label_cn}：", font=('Arial', 10))
            label.grid(row=row, column=0, sticky='w', pady=8)

            # 說明
            desc_label = ttk.Label(frame, text=f"({description})", font=('Arial', 8), foreground='gray')
            desc_label.grid(row=row, column=1, columnspan=2, sticky='w', padx=(10, 0))
            row += 1

            # 輸入框
            entry = ttk.Entry(frame, width=15)
            current_val = getattr(config, config_key, '')
            entry.insert(0, current_val)
            entry.grid(row=row, column=0, sticky='ew', pady=(0, 15))

            # 保存按鈕
            def save_hotkey(ck=config_key, e=entry):
                self.save_hotkey(ck, e.get())

            save_btn = ttk.Button(frame, text="保存", command=save_hotkey)
            save_btn.grid(row=row, column=1, padx=5)

            # 重置按鈕
            def reset_hotkey(ck=config_key, e=entry):
                self.reset_hotkey(ck, e)

            reset_btn = ttk.Button(frame, text="重置", command=reset_hotkey)
            reset_btn.grid(row=row, column=2, padx=5)

            self.hotkey_entries[config_key] = entry
            row += 1

        # 底部按鈕
        btn_frame = ttk.Frame(self.hotkeys_window)
        btn_frame.pack(fill='x', padx=10, pady=10)

        ttk.Button(btn_frame, text="全部保存", command=self.save_all_hotkeys).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="全部重置", command=self.reset_all_hotkeys).pack(side='left', padx=5)

    def save_hotkey(self, config_key, value):
        """保存單個按鍵設置"""
        try:
            if not value:
                logging.warning("按鍵值不能為空")
                return

            # 更新 config 模組
            setattr(config, config_key, value)

            # 更新 YAML
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'hotkeys' not in data:
                data['hotkeys'] = {}
            data['hotkeys'][config_key] = value

            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)

            logging.info(f"按鍵設置 {config_key} 已保存: {value}")
        except Exception as e:
            logging.error(f"保存按鍵設置失敗: {e}")

    def reset_hotkey(self, config_key, entry):
        """重置單個按鍵設置為預設值"""
        try:
            # 讀取 YAML 原始預設值
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            original_val = data.get('hotkeys', {}).get(config_key, '')
            entry.delete(0, tk.END)
            entry.insert(0, original_val)
            setattr(config, config_key, original_val)
            logging.info(f"按鍵設置 {config_key} 已重置為: {original_val}")
        except Exception as e:
            logging.error(f"重置按鍵設置失敗: {e}")

    def save_all_hotkeys(self):
        """保存所有按鍵設置"""
        hotkeys_dict = {}
        for config_key, entry in self.hotkey_entries.items():
            value = entry.get()
            if value:
                hotkeys_dict[config_key] = value
                setattr(config, config_key, value)

        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'hotkeys' not in data:
                data['hotkeys'] = {}
            data['hotkeys'].update(hotkeys_dict)

            with open(self.yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)

            logging.info("所有按鍵設置已保存")
        except Exception as e:
            logging.error(f"保存按鍵設置失敗: {e}")

    def reset_all_hotkeys(self):
        """重置所有按鍵設置為預設值"""
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            original_hotkeys = data.get('hotkeys', {})
            for config_key, original_val in original_hotkeys.items():
                if config_key in self.hotkey_entries:
                    self.hotkey_entries[config_key].delete(0, tk.END)
                    self.hotkey_entries[config_key].insert(0, original_val)
                    setattr(config, config_key, original_val)

            logging.info("所有按鍵設置已重置為預設值")
        except Exception as e:
            logging.error(f"重置按鍵設置失敗: {e}")

    def quit(self):
        self.stop()
        self.root.quit()

    def _update_status_loop(self):
        # 定期更新狀態標籤（從 aimlab_debug 的全域變數讀取）
        if getattr(aimlab_debug, 'running', False):
            if getattr(aimlab_debug, 'detection_paused', False):
                self.status_var.set("狀態：暫停")
            else:
                self.status_var.set("狀態：運行")
        else:
            self.status_var.set("狀態：停止")

        self.root.after(200, self._update_status_loop)


def main():
    app = AimlabGUI()
    app.root.mainloop()


if __name__ == '__main__':
    main()
