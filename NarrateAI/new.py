from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from flask import Flask, render_template, send_from_directory
import lang_m4
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'upload_files'  # Changed to store files in 'upload_files'
# PROCESSED_FOLDER = 'uploads'    # Intermediate and output files
PROCESSED_FOLDER = r"D:\NEWTheNarrateAI\processed_videos"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('frontend1.html')  # Ensure 'frontend.html' exists inside 'templates/'


# ===========================
# 📌 1️⃣ API: Add Files to Directory
# ===========================
@app.route('/add_files', methods=['POST'])
def add_files():
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files provided"})

    files = request.files.getlist('files')
    for file in files:
        file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    return jsonify({"success": True, "message": "Files added successfully"})

# ===========================
# 📌 2️⃣ API: List Files in Directory
# ===========================
@app.route('/list_files', methods=['GET'])
def list_files():
    files = os.listdir(UPLOAD_FOLDER)
    return jsonify({"files": files})

# ===========================
# 📌 3️⃣ API: Process Selected File
# ===========================
@app.route('/upload_selected', methods=['POST'])
def upload_selected():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({"success": False, "error": "No file selected"})

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({"success": False, "error": "File not found"})

    # Process the selected video
    result = lang_m4.prcss(filepath)
    
    return jsonify(result)

@app.route('/get_video/<filename>')
def get_video(filename):
    folder_path = os.path.join(PROCESSED_FOLDER, filename)
    video_file = "processed_video.mp4"  
    video_path = os.path.join(folder_path, video_file)
    if os.path.exists(video_path):
        return send_from_directory(folder_path, video_file)
    return jsonify({"success": False, "error": "Video not found"}), 404




# ===========================
# 📌 6️⃣ Start Flask Server
# ===========================
if __name__ == '__main__':
    app.run(debug=False, threaded=True)

