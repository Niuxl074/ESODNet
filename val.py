import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/CLADet/runs/train/exp_yolov8n-CLAD-LTFA-DOTA-v1.0/weights/best.pt')
    model.val(data='/CLADet/ultralytics/cfg/datasets/DOTA-v1.0.yaml',
              split='val',
              imgsz=640,
              batch=8,
              # rect=False,
              save_json=True, 
              project='runs/val',
              name='exp_yolov8n-CLAD-LTFA-DOTA-v1.0',
              )



