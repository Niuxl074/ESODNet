import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/CLADet/runs/train/exp_yolov8n-CLAD-LTFA-DOTA-v1.0/weights/best.pt')
    model.predict(source='/CLADet/dataset/DOTA-v1.0/images/test',
                  imgsz=640,
                  project='runs/detect',
                  name='exp_yolov8n-CLAD-LTFA_DOTA-v1.0',
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )
