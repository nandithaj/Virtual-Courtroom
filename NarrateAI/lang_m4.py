from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import os
import cv2
import torch
import shutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from scenedetect import VideoManager, SceneManager, ContentDetector
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
#import numpy as np
from gtts import gTTS
import whisper
from googletrans import Translator

BASE_UPLOAD_FOLDER = r"D:\NEWTheNarrateAI\uploads"
PROCESSED_VIDEOS_FOLDER = r"D:\NEWTheNarrateAI\processed_videos"


shutil.rmtree(BASE_UPLOAD_FOLDER, ignore_errors=True)
os.makedirs(BASE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
# summarizer = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# yolo_model = YOLO("yolov5s.pt")
from ultralytics import YOLO
CACHE_DIR = os.path.join(os.getcwd(), "models_cache")  # 📌 Offline cache directory

# ✅ Load all models from local cache
def get_captioning_model():
    global processor, model
    if "processor" not in globals():
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=CACHE_DIR)
    return processor, model

def get_text_summarizer():
    global tokenizer, summarizer
    if "tokenizer" not in globals():
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=CACHE_DIR)
        summarizer = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", cache_dir=CACHE_DIR)
    return tokenizer, summarizer

def get_yolo_model():
    global yolo_model
    YOLO_WEIGHTS_PATH = os.path.join(CACHE_DIR, "yolov5su.pt")
    if "yolo_model" not in globals():
        yolo_model = YOLO(YOLO_WEIGHTS_PATH)  # ✅ Load YOLO from cache
    return yolo_model

def get_whisper_model():
    global whisper_model
    if "whisper_model" not in globals():
        whisper_model = whisper.load_model("small")  # ✅ Whisper automatically caches
    return whisper_model

# ✅ Set environment variables for offline mode (IMPORTANT!)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Right after defining `get_*` functions
processor, model = get_captioning_model()
tokenizer, summarizer = get_text_summarizer()
yolo_model = get_yolo_model()
whisper_model = get_whisper_model()


def detect_language(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["language"]

from scenedetect import open_video, SceneManager, ContentDetector

def detect_scenes(video_path):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=40.0))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()


def extract_first_frame(video_path, timestamp, output_folder):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    success, frame = video.read()
    frame_path = None
    if success:
        frame_path = os.path.join(output_folder, f"scene_{int(timestamp)}.jpg")
        cv2.imwrite(frame_path, frame)
    video.release()
    return frame_path

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
    return processor.batch_decode(caption_ids, skip_special_tokens=True)[0]

def detect_objects(image_path):
    results = yolo_model(image_path)
    image = Image.open(image_path)
    img_width, img_height = image.size
    detected_objects = []

    def get_human_readable_position(x_center, y_center):
        if x_center < img_width * 0.33:
            horizontal = "left"
        elif x_center > img_width * 0.66:
            horizontal = "right"
        else:
            horizontal = "center"

        if y_center < img_height * 0.33:
            vertical = "top"
        elif y_center > img_height * 0.66:
            vertical = "bottom"
        else:
            vertical = "middle"

        # 🌟 Human-friendly mapping
        position_map = {
            ("top", "left"): "in the top left corner",
            ("top", "center"): "at the top center",
            ("top", "right"): "in the top right corner",
            ("middle", "left"): "on the left side",
            ("middle", "center"): "in the center",
            ("middle", "right"): "on the right side",
            ("bottom", "left"): "in the bottom left corner",
            ("bottom", "center"): "at the bottom center",
            ("bottom", "right"): "in the bottom right corner",
        }
        return position_map.get((vertical, horizontal), "in the scene")

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            label = result.names[cls]
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            position_description = get_human_readable_position(x_center, y_center)
            detected_objects.append(f"a {label} {position_description}")

    return ", and ".join(detected_objects) if detected_objects else ""



# def detect_objects(image_path):
#     results = yolo_model(image_path)
#     image = Image.open(image_path)
#     img_width, img_height = image.size
#     detected_objects = []

#     for result in results:
#         for box in result.boxes:
#             cls = int(box.cls)  # Object class index
#             label = result.names[cls]  # Object name
#             x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

#             # Calculate center of the object
#             x_center = (x_min + x_max) / 2
#             y_center = (y_min + y_max) / 2

#             # Determine horizontal position
#             if x_center < img_width * 0.33:
#                 horizontal_position = "left"
#             elif x_center > img_width * 0.66:
#                 horizontal_position = "right"
#             else:
#                 horizontal_position = "center"

#             # Determine vertical position
#             if y_center < img_height * 0.33:
#                 vertical_position = "top"
#             elif y_center > img_height * 0.66:
#                 vertical_position = "bottom"
#             else:
#                 vertical_position = "middle"

#             # Generate description
#             position_description = f"A {label} is in the {vertical_position}-{horizontal_position} part of the scene."
#             detected_objects.append(position_description)

#     return " ".join(detected_objects) if detected_objects else "No objects detected."


# def generate_detailed_caption(image_path):
#     # Get the initial caption
#     blip_caption = generate_caption(image_path)
    
#     # Detect objects using YOLO
#     objects_detected = detect_objects(image_path)
    
#     # Create a detailed description
#     if objects_detected:
#         raw_description = f"The video shows {blip_caption}. In this scene, you can see {objects_detected}."
#     else:
#         raw_description = blip_caption

#     # Use BART to refine the description
#     inputs = tokenizer(raw_description, return_tensors="pt", max_length=1024, truncation=True)
#     summary_ids = summarizer.generate(**inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
#     refined_description = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return refined_description

def generate_detailed_caption(image_path):
    """
    Generates a natural, detailed caption by combining:
    - BLIP-generated visual caption
    - YOLO-detected objects with spatial positions
    - BART summarization for natural refinement
    """
    # 🧠 Step 1: Get BLIP caption
    blip_caption = generate_caption(image_path)

    # 🔍 Step 2: Detect objects with YOLO
    object_description = detect_objects(image_path)

    # 🧩 Step 3: Merge into a single paragraph
    if object_description:
        combined_description = (
            f"The image depicts {blip_caption}. "
            f"Visible in the scene are {object_description}."
        )
    else:
        combined_description = f"The image depicts {blip_caption}."

    # ✨ Step 4: Use BART to rewrite as a smoother, natural-language caption
    inputs = tokenizer(combined_description, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarizer.generate(
        **inputs,
        max_length=100,
        min_length=25,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )
    refined_caption = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return refined_caption.strip()







def translate_text(text, target_lang):
    translator = Translator()
    return translator.translate(text, dest=target_lang).text

def text_to_speech(text, output_path, lang):
    if lang == "en":
        lang_code = "en"
    elif lang in ["hi", "hin"]:
        lang_code = "hi"
    elif lang in ["ml", "mal"]:
        lang_code = "ml"
    else:
        raise ValueError(f"Unsupported language for TTS: {lang}")

    tts = gTTS(text=text, lang=lang_code)
    tts.save(output_path)
    return output_path


def format_timecode(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def build_srt(entries, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, (start, end, description) in enumerate(entries, start=1):
            start_timecode = format_timecode(start)
            end_timecode = format_timecode(end)
            f.write(f"{i}\n{start_timecode} --> {end_timecode}\n{description}\n\n")

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(PROCESSED_VIDEOS_FOLDER, video_name)
    os.makedirs(output_folder, exist_ok=True)
    
    scenes = detect_scenes(video_path)
    modified_clips = []
    adjusted_audio_clips = []
    srt_entries = []
    video = VideoFileClip(video_path)
    lang = detect_language(video_path)
    current_audio_time = 0

    for i, (start_time, end_time) in enumerate(scenes):
        start_sec = min(start_time.get_seconds(), video.duration)
        end_sec = min(end_time.get_seconds(), video.duration)
        
        frame_path = extract_first_frame(video_path, start_sec, BASE_UPLOAD_FOLDER)  # Save in uploads/
        if frame_path:
            caption = generate_detailed_caption(frame_path)
            translated_caption = translate_text(caption, lang)
            tts_path = os.path.join(BASE_UPLOAD_FOLDER, f"scene_{video_name}_{i+1}.mp3")  # Unique filename
            text_to_speech(translated_caption, tts_path, lang)
            tts_audio = AudioFileClip(tts_path)
            
            freeze_frame = ImageClip(frame_path, duration=tts_audio.duration).set_fps(24)
            modified_clips.append(freeze_frame)
            adjusted_audio_clips.append(tts_audio.set_start(current_audio_time))
            srt_entries.append((current_audio_time, current_audio_time + tts_audio.duration, translated_caption))
            current_audio_time += tts_audio.duration
            
            scene_clip = video.subclip(start_sec, end_sec)
            modified_clips.append(scene_clip)
            adjusted_audio_clips.append(scene_clip.audio.set_start(current_audio_time))
            current_audio_time += scene_clip.duration

    if not modified_clips:
        return {"success": False, "error": "No scenes were processed properly."}
    
    final_video = concatenate_videoclips(modified_clips)
    final_audio = CompositeAudioClip(adjusted_audio_clips)
    final_output_path = os.path.join(output_folder, "processed_video.mp4")
    final_video.set_audio(final_audio).write_videofile(final_output_path, codec="libx264", audio_codec="aac")
    
    srt_output_path = os.path.join(output_folder, "captions.srt")
    build_srt(srt_entries, srt_output_path)
    
    return {"success": True, "output_video": final_output_path, "srt_file": srt_output_path, "detected_language": lang}

if __name__ == '__main__':
    print("✅ All models are loaded offline from", CACHE_DIR)
    video_path = input("Enter the video file path: ")
    if os.path.exists(video_path):
        result = process_video(video_path)
        print(result)
    else:
        print("Error: Video file not found.")

def prcss(video_path):
    return process_video(video_path)