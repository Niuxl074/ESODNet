import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/ps/ultralytics-202404066/ultralytics-main/ultralytics/cfg/models/v8/yolov8n-FDPN-LSCD1.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    # /home/ps/ultralytics-202404066/ultralytics-main/ultralytics/cfg/models/v8/yolov8-FDPN-LSCD1.yaml
    # /home/ps/ultralytics-202404066/ultralytics-main/ultralytics/cfg/models/v5/yolov5.yaml
    model.train(data='/home/ps/ultralytics-202404066/ultralytics-main/ultralytics/cfg/datasets/DOTA_split_ss.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=8,
                close_mosaic=200,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                # resume='/home/ps/ultralytics-202404066/ultralytics-main/runs/train/exp_v6x_dota/weights/last.pt', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp_yolov8n-FDPN-LSCD1_DOTA111',
                )



