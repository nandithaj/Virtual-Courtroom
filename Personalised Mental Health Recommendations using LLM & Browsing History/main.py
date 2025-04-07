import subprocess
import sys
import json
import ollama
import csv
import os
import time
import cv2
import threading
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import pandas as pd
import numpy as np
import ipywidgets as widgets
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser
from IPython.display import display  # Fix for 'display' is not defined
from deepface import DeepFace


def run_notebook():
    """Executes the music recommendation and Spotify playback process."""
    try:
        # Load filtered_df_pca.csv
        filtered_df_pca = pd.read_csv('filtered_df_pca.csv')

        # Read detected emotion from the file
        with open("detected_emotion.txt", "r") as file:
            detected_emotion = file.read().strip()

        print(f"Detected Emotion: {detected_emotion}")

        # Filter songs based on detected mood
        detected_mood = detected_emotion
        filtered_by_mood = filtered_df_pca[filtered_df_pca['mood'] == detected_mood]

        # Ensure no duplicate songs
        filtered_by_mood = filtered_by_mood.drop_duplicates(subset=['song_name'])

        # Select 5 random songs
        random_songs = filtered_by_mood.sample(5)['song_name'].tolist()
        print("\nDetected mood is:", detected_mood)
        print("Random songs based on detected mood:")
        for i, song in enumerate(random_songs, 1):
            print(f"{i}. {song}")

        # Ask user to select a song by typing its number
        selected_index = int(input("\nType the number of the song you want to select: ")) - 1
        selected_song = random_songs[selected_index]
        print(f"\nYou selected: {selected_song}")

        # Recommend similar songs based on the selected song
        top_recommendation_uri = None
        if selected_song in filtered_by_mood['song_name'].values:
            selected_song_features = filtered_by_mood.loc[filtered_by_mood['song_name'] == selected_song, ['energy', 'valence']].values.flatten()
            all_songs_features = filtered_by_mood[['energy', 'valence']].values

            # Calculate similarity
            similarities = cosine_similarity([selected_song_features], all_songs_features)
            result_df = pd.DataFrame({'song_name': filtered_by_mood['song_name'], 'similarity': similarities.flatten(), 'uri': filtered_by_mood['uri']})
            result_df = result_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
            result_df = result_df[result_df['song_name'] != selected_song]
            top_3_recommendations = result_df.head(3)

            print("\nTop 3 recommendations based on energy and valence:")
            print(top_3_recommendations[['song_name', 'similarity']])

            # Get URI of the top recommended song
            top_recommendation_uri = top_3_recommendations.iloc[0]['uri']
            print(f"\nURI of top recommendation: {top_recommendation_uri}")

        # Open Spotify Web Player
        webbrowser.open("https://open.spotify.com/")

        # Replace with your actual Spotify credentials
        client_id = '9df87f74f32e4e68b42201a3c0195426'
        client_secret = 'dad01ae1d41c4b3ea83eb034f8c71c7a'
        redirect_uri = 'http://127.0.0.1:8888/callback'  # Updated to 127.0.0.1 for Windows compatibility

        # Set up the Spotify OAuth object with user authentication
        scope = 'user-modify-playback-state user-read-playback-state'
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret, redirect_uri=redirect_uri, scope=scope, cache_path=".spotipyoauthcache"))

        # Get the list of user's available devices
        devices = sp.devices()
        device_id = None

        # Check if there are available devices
        if devices['devices']:
            print("\nAvailable Spotify devices:")
            for device in devices['devices']:
                print(f"Device ID: {device['id']}, Name: {device['name']}, Type: {device['type']}")
            device_id = devices['devices'][0]['id']  # Use the first available device
        else:
            print("No active Spotify devices found.")

        # Validate and play the top recommended song
        if top_recommendation_uri and top_recommendation_uri.startswith("spotify:track:"):
            try:
                if device_id:
                    sp.start_playback(device_id=device_id, uris=[top_recommendation_uri])
                    print("Playing top recommended song...")
                    return f"Playing song: {top_recommendation_uri}"
                else:
                    print("No active device available to play the song.")
                    return "No active device available to play the song."
            except spotipy.exceptions.SpotifyException as e:
                print("Spotify playback error:", str(e))
                return f"Spotify playback error: {str(e)}"
        else:
            print("Invalid or missing track URI.")
            return "Invalid or missing track URI."

    except Exception as e:
        print("Unexpected error:", str(e))
        return f"Unexpected error: {str(e)}"



def detect_emotion_from_text(text):
    """Detects emotion using Ollama based on the given text."""
    prompt = (
        "Analyze the following text and determine the predominant emotion. "
        "Respond with only one of these options: happy, sad, angry, or calm.\n\n"
        f"{text}\n\nEmotion:"
    )
    
    allowed_emotions = {"happy", "sad", "angry", "calm"}
    
    while True:  # Keep asking until we get a valid emotion
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        emotion = response['message']['content'].strip().lower()
        
        # Extract the emotion if it contains any of the allowed emotions
        for allowed in allowed_emotions:
            if allowed in emotion:
                return allowed  # Return the detected emotion immediately
        
        print(f"Unexpected response: {emotion}. Re-prompting...")  # For debugging


def test3():
    file_path = "browsing_history.csv"  # CSV file with headings: Title, URL, Last Visit Time
    emotions = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            if not any(reader):
                print("The file is empty.")
                return
            
            # Move the reader back to the start after the empty check
            file.seek(0)
            reader = csv.DictReader(file)

            # Process each row based on the Title column
            for row in reader:
                title = row.get("Title", "").strip()
                if title:
                    emotion = detect_emotion_from_text(title)
                    emotions.append(emotion)
                    print(f"Title: {title} -> Detected Emotion: {emotion}")
                else:
                    print("Skipped: Empty or missing title")
        
        # Determine the most frequent emotion
        if emotions:
            dominant_emotion = max(set(emotions), key=emotions.count)
            #dominant_emotion.strip()
            print(f"\nDominant Emotion: {dominant_emotion}")
            return dominant_emotion
        else:
            print("No emotions detected.")
    
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

frame_width = 640
frame_height = 480
fps = 20.0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
video_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

recording = True  # Flag to control video capturing
start_time = time.time()
last_frame_time = start_time  # Track time for saving frames

def capture_video():
    global recording, last_frame_time
    while recording:
        ret, img = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        video_writer.write(img)

        '''# Save a frame every 30 seconds
        current_time = time.time()
        if current_time - last_frame_time >= 30:  # Adjusted to 30 seconds
            cv2.imwrite(f'frame_{int(current_time)}.jpg', img)
            last_frame_time = current_time'''

        cv2.imshow('Video Stream', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recording = False
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def get_response(prompt):
    command = f'echo "{prompt}" | ollama run llama2'
    stream = os.popen(command)
    response = stream.read().strip()
    return response if response else "I'm here for you. Can you tell me more?"

def print_word_by_word(text):
    for word in text.split():
        sys.stdout.write(word + " ")
        sys.stdout.flush()
        time.sleep(0.2)
    print()

def store_user_response(user_input):
    with open("user_responses.txt", "a") as file:
        file.write(user_input + "\n")

def chat():
    global recording
    print("Chatbot: Hi there! How are you feeling today?")

    with open("user_responses.txt", "w") as file:
        pass 
    
    video_thread = threading.Thread(target=capture_video)
    video_thread.start()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            recording = False
            print("Chatbot: Take care! I'm here whenever you need me. 😊")
            break
        store_user_response(user_input)
        print("Chatbot:", end=" ", flush=True)
        response = get_response(user_input)
        print_word_by_word(response)
    
    video_thread.join()

def test1():
    file_path = "user_responses.txt"  # Change this to your file name
    
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            if not text.strip():
                print("The file is empty.")
                return
            
            emotion = detect_emotion_from_text(text)
            #emotion.strip()
            print("Detected Emotion:", emotion)
            return emotion
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except Exception as e:
        print("An error occurred:", e)

def describe_face_emotion(image):
    """Uses DeepFace to describe the facial expressions."""
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return f"A person showing {emotion} facial expressions."
    except Exception:
        return "No clear facial expression detected."

def analyze_emotion_with_ollama(description):
    """Uses Ollama to classify emotions based on text descriptions."""
    prompt = (
        "Classify the following facial expression into 'Happy', 'Sad', 'Calm', or 'Angry'. "
        "Respond with only one of these options:\n\n"
        f"{description}\n\nEmotion:"
    )
    
    allowed_emotions = {"happy", "sad", "angry", "calm"}
    
    while True:  # Keep asking until we get a valid emotion
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        emotion = response['message']['content'].strip().lower()
        
        # Check if the emotion is one of the allowed emotions
        if emotion in allowed_emotions:
            return emotion  # Return the detected emotion immediately
        
        print(f"Unexpected response: {emotion}. Re-prompting...")  # For debugging

# Load the video
def test2():
    video_path = "output.mp4"
    cap = cv2.VideoCapture(video_path)

    emotions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 150th frame to improve performance
        if frame_count % 100 == 0:
            description = describe_face_emotion(frame)
            emotion = analyze_emotion_with_ollama(description)
            
            if emotion:
                emotions.append(emotion)
                print(f"Frame {frame_count}: {emotion}")
        
        frame_count += 1

    cap.release()

    # Determine the most frequent emotion
    if emotions:
        dominant_emotion = max(set(emotions), key=emotions.count)
        print(f"\nDominant Emotion: {dominant_emotion}")
        return dominant_emotion
    else:
        print("No emotions detected.")

def apply_rules(emotion_a, emotion_b):
    # Handle None cases
    if not emotion_a:
        return emotion_b
    if not emotion_b:
        return emotion_a
    
    # Emotion decision rules based on the image
    if (emotion_a, emotion_b) in [('happy', 'happy')]:
        return 'happy'
    elif (emotion_a, emotion_b) in [('happy', 'calm'), ('calm', 'happy')]:
        return 'happy'
    elif (emotion_a, emotion_b) in [('happy', 'sad'), ('sad', 'happy')]:
        return 'calm'
    elif (emotion_a, emotion_b) in [('happy', 'angry'), ('angry', 'happy')]:
        return 'sad'
    elif (emotion_a, emotion_b) in [('sad', 'sad')]:
        return 'sad'
    elif (emotion_a, emotion_b) in [('sad', 'angry'), ('angry', 'sad')]:
        return 'angry'
    elif (emotion_a, emotion_b) in [('sad', 'calm'), ('calm', 'sad')]:
        return 'sad'
    elif (emotion_a, emotion_b) in [('angry', 'angry')]:
        return 'angry'
    elif (emotion_a, emotion_b) in [('angry', 'calm'), ('calm', 'angry')]:
        return 'angry'
    elif (emotion_a, emotion_b) in [('calm', 'calm')]:
        return 'calm'
    else:
        return emotion_a  # Default to the first emotion if no rule matches
    

def main():
    # Step 1: Ask for browsing history analysis
    print("Do you want to provide your browsing history for emotion analysis? (yes/no)")
    if input().strip().lower() == 'yes':
        print("\nAnalyzing browsing history for emotions...")
        emotion1 = test3()
        print(f"Browsing History Emotion: {emotion1}")
    else:
        emotion1 = None

    # Step 2: Ask for chat interaction
    print("\nDo you want to chat? (yes/no)")
    if input().strip().lower() == 'yes':
        print("\nStarting chat...")
        chat()

        # Analyze emotions from chat responses
        print("\nAnalyzing chat responses for emotions...")
        emotion2 = test1()
        print(f"Chat Emotion: {emotion2}")

        # Analyze emotions from video frames
        print("\nAnalyzing video for emotions...")
        #emotion3 = run_script("test2.py")
        emotion3 = test2()
        print(f"Video Emotion: {emotion3}")
    else:
        emotion2 = emotion3 = None
    
    # Step 3: Determine dominant emotion from all sources
    # Define a function to apply emotion decision rules using if-else

# Extract and filter emotions
    emotions = [e for e in [emotion1, emotion2, emotion3] if e]

    if emotions:
        # Handle cases when emotion1 is None
        if not emotion1:
            dominant_emotion = apply_rules(emotion2, emotion3)
        else:
            # Combine emotion2 and emotion3 to form emotion4
            emotion4 = apply_rules(emotion2, emotion3)
            # Combine emotion1 and emotion4 to form dominant_emotion
            dominant_emotion = apply_rules(emotion1, emotion4)

        print(f"\nOverall Dominant Emotion: {dominant_emotion}")
        
        # Save dominant emotion to a file for the notebook to read
        with open("detected_emotion.txt", "w") as file:
            file.write(dominant_emotion)
    else:
        print("No emotions detected.")
        return

    # Step 4: Run llm.ipynb to get song recommendations based on the dominant emotion
    print("\nRecommending songs based on dominant emotion...")
    run_notebook()

if __name__ == "__main__":
    main()

#pip install -r requirements.txt
