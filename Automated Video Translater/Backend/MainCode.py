import ffmpeg
import os
import librosa
import numpy as np
from pydub import AudioSegment
import speech_recognition as sr
from pydub.silence import split_on_silence
from pydub.effects import normalize
import shutil
import pyttsx3
from moviepy.editor import VideoFileClip, AudioFileClip
import subprocess
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import requests

API_KEY = "Add your google cloud api key"

SAMPLE_RATE = 22050
DURATION = 4  
N_MFCC = 64
TARGET_WIDTH = 216

def detect_dialogue_segments_with_pitch(audio_file, silence_thresh=-40, min_silence_duration=1.0, pitch_change_thresh=30):
    y, sr = librosa.load(audio_file, sr=None)
    

    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)

    sound_frames = db > silence_thresh
    times = librosa.frames_to_time(np.arange(len(sound_frames)), sr=sr, hop_length=hop_length)

    pitches, _ = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_values = []
    for i in range(pitches.shape[1]):
        pitch = pitches[:, i]
        pitch_value = pitch[pitch > 0]
        pitch_mean = np.mean(pitch_value) if len(pitch_value) else 0
        pitch_values.append(pitch_mean)

    dialogue_intervals = []
    start_time = None
    last_pitch = 0
    for i, is_sound in enumerate(sound_frames):
        time = times[i]
        if is_sound:
            pitch_diff = abs(pitch_values[i] - last_pitch)
            if start_time is None:
                start_time = time
                last_pitch = pitch_values[i]
            elif pitch_diff > pitch_change_thresh:
                end_time = time
                dialogue_intervals.append((start_time, end_time))
                start_time = time
                last_pitch = pitch_values[i]
        else:
            if start_time is not None:
                end_time = time
                if end_time - start_time > 0.2:
                    dialogue_intervals.append((start_time, end_time))
                start_time = None

    if not dialogue_intervals:
        dialogue_intervals.append((0.0, librosa.get_duration(y=y, sr=sr)))

    return dialogue_intervals

def trim_audio_by_segments(audio_file, output_dir, segments):
    audio = AudioSegment.from_file(audio_file)
    for idx, (start, end) in enumerate(segments):
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        clip = audio[start_ms:end_ms]
        clip.export(f"{output_dir}/dialogue_segment_{idx+1}.wav", format="wav")

def add_silence_to_audio(audio, silence_duration_ms=1000):

    silence = AudioSegment.silent(duration=silence_duration_ms)  
    return audio + silence  

def improve_audio_clarity(audio, volume_increase_db=10):

    audio = audio + volume_increase_db
    audio = audio.low_pass_filter(3000)  
    audio = normalize(audio)

    return audio

def split_audio(audio_file_path, chunk_length_ms=10000):

    audio = AudioSegment.from_wav(audio_file_path)
    audio = improve_audio_clarity(audio)
    audio = add_silence_to_audio(audio)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    chunks = split_audio(audio_file_path) 

    full_text = ""
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunk{i}.wav"
        chunk.export(chunk_path, format="wav")  

        with sr.AudioFile(chunk_path) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="ml-IN")
                full_text += text + " "
            except sr.UnknownValueError:
                print(f"Chunk {i}: Google Web Speech API could not understand the audio.")
            except sr.RequestError as e:
                print(f"Chunk {i}: Could not request results from Google Web Speech API; {e}")

    print("Malayalam Text: ", full_text.strip())
    return full_text.strip()

def load_model_and_encoder(model_dir):

    model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model'))
    le = LabelEncoder()
    le.classes_ = np.load(os.path.join(model_dir, 'label_encoder_classes.npy'), allow_pickle=True)
    return model, le

def preprocess_audio(file_path):

    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    if len(audio) < SAMPLE_RATE * DURATION:
        audio = np.pad(audio, (0, max(0, SAMPLE_RATE * DURATION - len(audio))), mode='constant')
    else:
        audio = audio[:SAMPLE_RATE * DURATION]
    
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    if mfcc.shape[1] < TARGET_WIDTH:
        mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_WIDTH - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :TARGET_WIDTH]
    
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    
    return mfcc

def predict_gender(file_path, model, label_encoder):

    mfcc = preprocess_audio(file_path)

    pred_probs = model.predict(mfcc, verbose=0)
    pred_class = label_encoder.inverse_transform([np.argmax(pred_probs)])[0]
    confidence = np.max(pred_probs)
    
    return pred_class, confidence

def detect_gender(audio_file, model, label_encoder):

        gender, confidence = predict_gender(audio_file, model, label_encoder)
        print(f"\nSingle file prediction:")
        print(f"File: {os.path.basename(audio_file)}")
        print(f"Predicted gender: {gender}")
        print(f"Confidence: {confidence:.2%}")
        if gender == "Male":
            print("Identified Gender: Male")
            return 1
        else:
            print("Identified Gender: Female")
            return 2


def Mal2Eng(Maltext):
    
    target_lang = "en"
    url = f"https://translation.googleapis.com/language/translate/v2?key={API_KEY}"
    data = {"q": Maltext, "target": target_lang}
    response = requests.post(url, data=data)
    s=response.json()["data"]["translations"][0]["translatedText"]
    Engtext=s.replace("&#39;","'")
    return Engtext

def eng2aud(text,dialougue_len,gender,j):

    engine = pyttsx3.init()
    words = text.split()
    num_words = len(words)
    
    speech_rate=(60/dialougue_len)*num_words

    voices = engine.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice {index}: {voice.name} ({voice.id})")
    print("Chosen voice : Voice ",gender)

    engine.setProperty('voice', voices[gender].id)
    engine.setProperty('rate', speech_rate)  
    engine.setProperty('volume', 1.0)  

    engine.save_to_file(text, "Backend/final_dialogues/dialogue_segment_"+str(i+1)+".wav")

    engine.runAndWait()
    
if __name__ == '__main__':

    
    inputvideo="Frontend/uploads/input.mp4"
    print("Loaded Video :",inputvideo)
    ffmpeg.input(inputvideo).output("Backend/out.mp3").run(overwrite_output=True)
    print("Audio Extracted.")

#spleeter
    subprocess.run(["python", "Backend/Spleet.py"])

#Trim

    audio_file = "Backend/output/out/vocals.wav"
    output_dir = "Backend/output_dialogues"
    silence_thresh = -40
    min_silence_duration = 1
    pitch_change_thresh = 14000

    os.makedirs(output_dir, exist_ok=True)

    dialogue_segments = detect_dialogue_segments_with_pitch(audio_file, silence_thresh, min_silence_duration, pitch_change_thresh)

    trim_audio_by_segments(audio_file, output_dir, dialogue_segments)

    print(dialogue_segments)

    print(f"Saved {len(dialogue_segments)} dialogue segments in '{output_dir}'")

#conversion mal audio to english audio

    MODEL_DIR = 'Backend/GenderResnet2'
    model, label_encoder = load_model_and_encoder(MODEL_DIR)
    output_dir1 = "Backend/final_dialogues"
    os.makedirs(output_dir1, exist_ok=True)
    dialogue_data=[] #[len,nor0/male1/female2]
    for i in range(len(dialogue_segments)):
        source_path="Backend/output_dialogues/dialogue_segment_"+str(i+1)+".wav"
        dial_len= dialogue_segments[i][1]-dialogue_segments[i][0]
        if dial_len < 0.6:
            gender=0
            dialdata=[dial_len,gender]
            dialogue_data.append(dialdata)
            shutil.copy(source_path, os.path.join(output_dir1, "dialogue_segment_"+str(i+1)+".wav"))
            continue
        audio_file_path = "Backend/output_dialogues/dialogue_segment_"+str(i+1)+".wav"
        gender = detect_gender(audio_file_path, model, label_encoder)
        dialdata=[dial_len,gender]
        dialogue_data.append(dialdata)
        text_output = audio_to_text(audio_file_path)
        
        Eng_text=Mal2Eng(text_output)
        print("Translated text:", Eng_text)
        
        eng2aud(Eng_text,dial_len,gender-1,i+1)

    print(dialogue_data)

#merging accompaniments and audio

    accompaniment = AudioSegment.from_file(r"Backend/output/out/accompaniment.wav")

    for i in range(len(dialogue_segments)):
        vocal = AudioSegment.from_file(rf"Backend/final_dialogues/dialogue_segment_{i+1}.wav")
        timestamp = dialogue_segments[i][0] * 1000
        accompaniment = accompaniment.overlay(vocal, position=timestamp)
    accompaniment.export("Backend/combined_output.wav", format="wav")
    print("Vocals and accompaniment combined")

#audio video merging
    
    video = VideoFileClip(inputvideo)
    new_audio = AudioFileClip("Backend/combined_output.wav")
    video = video.set_audio(new_audio)
    video.write_videofile("Frontend/output/output.mp4", codec="libx264")
    print("Output Video Generated")