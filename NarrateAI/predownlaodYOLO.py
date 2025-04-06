import urllib.request
import os

YOLO_WEIGHTS_PATH = os.path.join(os.getcwd(), "models_cache", "yolov5s.pt")
os.makedirs(os.path.dirname(YOLO_WEIGHTS_PATH), exist_ok=True)

if not os.path.exists(YOLO_WEIGHTS_PATH):
    print("Downloading YOLOv5 weights...")
    urllib.request.urlretrieve("https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt", YOLO_WEIGHTS_PATH)
    print("YOLOv5 weights downloaded!")

else:
    print("YOLOv5 weights already exist, skipping download.")
