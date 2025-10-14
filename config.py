import ctypes
import yaml

# YAML
def read_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)  # 解析 YAML 檔案
    return data

yaml_file = r"config\config.yaml"  # 替換 YAML 路徑
yaml_data = read_yaml(yaml_file)

log_level = yaml_data['logging']['level'].upper()
dll_path = yaml_data['device']['path']
driver_mouse_control = ctypes.CDLL(dll_path)

WINDOW_TITLE = yaml_data['settings']['WINDOW_TITLE']
roi_width = yaml_data['settings']['roi_width']
roi_height = yaml_data['settings']['roi_height']
debug = yaml_data['settings']['debug']
control_mose = yaml_data['settings']['control_mose']
fire_switch = yaml_data['settings']['fire_switch']
fire_k = yaml_data['settings']['fire_k']
yolo_model_path = yaml_data['settings']['yolo_model_path']
mouse_smoothing = yaml_data['settings']['mouse_smoothing']
pid_kp = yaml_data['settings']['pid_kp']
pid_ki = yaml_data['settings']['pid_ki']
pid_kd = yaml_data['settings']['pid_kd']
pid_smooth = yaml_data['settings']['pid_smooth']
target_stickiness = yaml_data['settings']['target_stickiness']