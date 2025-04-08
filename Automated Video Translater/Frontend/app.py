from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Define file paths
    input_path = os.path.join(UPLOAD_FOLDER, "input.mp4")
    output_path = os.path.join(OUTPUT_FOLDER, "output.mp4")


    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Deleted old processed video: {output_path}")

    # Save the uploaded video
    file.save(input_path)
    print(f"Video saved at {input_path}")

    # Start video processing (Example: Convert video to grayscale using FFmpeg)
    print("Processing video...")
    try:
       subprocess.run(["python", "Backend/Code/MainCode.py"])
    except subprocess.CalledProcessError as e:
        print("Error processing video:", e)
        return jsonify({'error': 'Video processing failed'}), 500

    # Check if output video exists
    if os.path.exists(output_path):
        print(f"Processed video ready: {output_path}")
        return jsonify({'video_url': '/output/output.mp4'})
    else:
        return jsonify({'error': 'Processing failed'}), 500

@app.route('/output/<filename>')
def get_processed_video(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_from_directory(OUTPUT_FOLDER, filename)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
