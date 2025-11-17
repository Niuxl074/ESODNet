import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/ESODNet/runs/train/exp_yolov8n-FDPN-LSCD-DOTA-v1.0/weights/best.pt')
    model.predict(source='/ESODNet/dataset/DOTA-v1.0/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='exp_yolov8n-FDPN-LSCD_DOTA-v1.0',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )
