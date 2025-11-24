import logging
import config
from aimlab_gui import main as gui_main

logging.basicConfig(level=getattr(logging, config.log_level))  # 設定日誌級別

if __name__ == '__main__':
	# 啟動 GUI 應用程式
	gui_main()