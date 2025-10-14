import logging
import aimlab_debug
import config

logging.basicConfig(level=getattr(logging, config.log_level))  # 設定日誌級別

aimlab_debug.aimlab_debug()