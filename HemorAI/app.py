from flask import Flask, request, jsonify, send_from_directory,make_response
from flask_cors import CORS
import os
from middleware import predict_image_cnn, predict_image_densenet, predict_image_resenet,predict_image_vgg16
from middleware import gradcam_cnn, gradcam_densenet, gradcam_resnet, gradcam_vgg16

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"  # Publicly accessible folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    [c,d,r,v]=predict_image_cnn(filepath),predict_image_densenet(filepath),predict_image_resenet(filepath),predict_image_vgg16(filepath)
    gradcam_cnn(filepath)
    gradcam_densenet(filepath)
    gradcam_resnet(filepath)
    gradcam_vgg16(filepath)
    
    processed_filenames = ['http://127.0.0.1:5000/static/cnn_gradcam.jpg', 'http://127.0.0.1:5000/static/densenet_gradcam.jpg', 'http://127.0.1:5000/static/resnet_gradcam.jpg', 'http://127.0.1:5000/static/vgg16_gradcam.jpg'] 

    return jsonify({"efficient-Net": [processed_filenames[0],c],"densenet": [processed_filenames[1],d],"resnet": [processed_filenames[2],r],"vgg": [processed_filenames[3],v]})

# Serve static files (processed images)
@app.route("/static/<path:filename>")
def serve_static(filename):
    # response = make_response(send_from_directory('static', filename))
    # response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    # return response
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
