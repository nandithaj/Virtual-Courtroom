from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from ultralytics import YOLO
import whisper
import os

CACHE_DIR = os.path.join(os.getcwd(), "models_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

print("Downloading BLIP model...")
BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)

print("Downloading BART model...")
BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=CACHE_DIR)
BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=CACHE_DIR)

print("Downloading YOLOv5s model...")
yolo_weights_path = os.path.join(CACHE_DIR, "yolov5s.pt")
if not os.path.exists(yolo_weights_path):
    YOLO("yolov5s.pt").export(weights=yolo_weights_path)

print("Downloading Whisper model...")
whisper.load_model("small")

print("✅ All models downloaded and cached in", CACHE_DIR)
