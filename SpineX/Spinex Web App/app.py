from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import io
from PIL import Image
import os
import logging
from keras.applications.efficientnet import preprocess_input

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 👈 Enable CORS for all routes



# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Load both scoliosis classification models
model_path_1 = os.path.join(os.path.dirname(__file__), "scoliosisCNN.keras")
model_path_2 = os.path.join(os.path.dirname(__file__), "scoliosisTL.keras")

model_1 = tf.keras.models.load_model(model_path_1)
model_2 = tf.keras.models.load_model(model_path_2)

def calculate_cobb_angle(points):
    """
    Compute the Cobb angle given three key points.
    """
    try:
        x1, y1 = points[0]["x"], points[0]["y"]
        x2, y2 = points[1]["x"], points[1]["y"]
        x3, y3 = points[2]["x"], points[2]["y"]

        # Compute slopes
        m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
        m2 = (y3 - y2) / (x3 - x2) if x3 != x2 else float('inf')

        if m1 == float('inf') or m2 == float('inf'):
            angle_rad = np.pi / 2
        else:
            angle_rad = abs(np.arctan((m2 - m1) / (1 + m1 * m2)))

        return round(np.degrees(angle_rad), 2)
    except Exception as e:
        app.logger.error(f"Error in calculate_cobb_angle: {e}")
        return None

@app.route("/api/cobb-angle", methods=["POST"])
def cobb_angle():
    try:
        data = request.json
        points = data.get("points", [])

        if len(points) != 5:
            return jsonify({"error": "Exactly 5 points required."}), 400

        points.insert(3, points[2])  # Duplicate third point

        # Calculate Cobb angles
        top_cobb_angle = calculate_cobb_angle(points[:3])
        bottom_cobb_angle = calculate_cobb_angle(points[3:])
        max_cobb_angle = max(top_cobb_angle, bottom_cobb_angle) if top_cobb_angle and bottom_cobb_angle else None

        if max_cobb_angle is None:
            return jsonify({"error": "Failed to calculate Cobb angle."}), 500

        # Determine curve type
        if top_cobb_angle >= 15 and bottom_cobb_angle >= 15:
            curve_type = "S-shaped curve"
        elif top_cobb_angle >= 15 or bottom_cobb_angle >= 15:
            curve_type = "C-shaped curve"
        else:
            curve_type = "No significant curve detected"

        # Determine severity
        if 10 <= max_cobb_angle < 20:
            severity = (
                "Mild scoliosis: Cobb angle of 10–20 degrees.\n"
                "Usually observed without treatment; exercise may be recommended."
            )
        elif 20 <= max_cobb_angle <= 40:
            severity = (
                "Moderate scoliosis: Cobb angle of 20–40 degrees.\n"
                "Bracing may be required to stop progression."
            )
        elif max_cobb_angle > 40:
            severity = (
                "Severe scoliosis: Cobb angle above 40 degrees.\n"
                "Spinal fusion surgery may be required."
            )
        else:
            severity = "No scoliosis detected."

        return jsonify({
            "top_angle": top_cobb_angle,
            "bottom_angle": bottom_cobb_angle,
            "max_angle": max_cobb_angle,
            "curve_type": curve_type,
            "severity": severity
        })

    except Exception:
        return jsonify({"error": "Server error"}), 500


def preprocess_image(image):
    try:
        img = Image.open(io.BytesIO(image))
        img = img.resize((224, 224))  # Resize to match model input size
        img = np.array(img) / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img
    except Exception as e:
        app.logger.error(f"Error in preprocess_image: {e}")
        return None

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"].read()
        img = preprocess_image(file)

        if img is None:
            return jsonify({"error": "Image processing failed"}), 500

        # Model 1 (Uses Argmax for classification)
        predictions_1 = model_1.predict(img)
        predicted_class_1 = np.argmax(predictions_1, axis=1)[0]
        confidence_1 = np.max(predictions_1)  # Confidence between 0 and 1
        label_1 = "Scoliosis" if predicted_class_1 == 1 else "Normal"

        # Model 2 (Binary classification)
        pred_2 = model_2.predict(img)[0][0]
        label_2 = "Scoliosis" if pred_2 > 0.5 else "Normal"

        # Log results
        app.logger.debug(f"Model 1 Label: {label_1}, Confidence: {confidence_1}")
        app.logger.debug(f"Model 2 Label: {label_2}, Confidence: {pred_2}")

        # Decision logic
        if label_1 == label_2:
            final_label = label_1
            final_confidence = max(confidence_1, float(pred_2))
        else:
            if confidence_1 > pred_2:
                final_label = label_1
                final_confidence = confidence_1
            else:
                final_label = label_2
                final_confidence = float(pred_2)

        # Log final decision
        app.logger.info(f"Final Prediction: {final_label}, Confidence: {final_confidence}")

        return jsonify({
            "prediction": final_label,
            "confidence": float(final_confidence)
        })
    except Exception as e:
        app.logger.error(f"Error in predict route: {e}")
        return jsonify({"error": "Server error"}), 500
    



# Load the posture classification model (change the model file name to 'lordkyph.keras')
posture_model_path = os.path.join(os.path.dirname(__file__), "lordkyph.keras")
try:
    posture_model = tf.keras.models.load_model(posture_model_path)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error(f"Error loading model: {e}")

# Define class labels for posture classification
posture_class_labels = ['Kyphosis', 'Lordosis', 'Normal']

def preprocess_posture(image):
    """
    Process the image for model prediction.
    """
    try:
        app.logger.info("Processing image for model prediction.")
        
        # Ensure the image is opened and converted to RGB
        img = Image.open(io.BytesIO(image)).convert('RGB')
        app.logger.debug("Image converted to RGB.")

        # Resize to the required input shape (224x224)
        img = img.resize((224, 224))
        app.logger.debug("Image resized to 224x224.")

        # Convert the image to numpy array and preprocess it
        img_array = np.array(img)
        app.logger.debug(f"Image array shape before preprocessing: {img_array.shape}")
        
        # EfficientNet preprocessing (if applicable)
        img_array = preprocess_input(img_array)
        app.logger.debug("Image preprocessed.")

        # Add batch dimension to the image
        img_array = np.expand_dims(img_array, axis=0)
        app.logger.debug(f"Image array shape after adding batch dimension: {img_array.shape}")

        return img_array
    except Exception as e:
        app.logger.error(f"Error in preprocess_posture: {e}")
        return None

@app.route("/api/predict-posture", methods=["POST"])
def predict_posture():
    """
    Predict the posture classification based on the uploaded image.
    """
    try:
        if "file" not in request.files:
            app.logger.error("No file provided.")
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"].read()
        app.logger.info(f"Received file with size: {len(file)} bytes.")
        
        img_array = preprocess_posture(file)

        if img_array is None:
            app.logger.error("Image processing failed.")
            return jsonify({"error": "Image processing failed"}), 500

        # Predict using posture model
        app.logger.info("Making prediction using the posture model.")
        predictions = posture_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class]) * 100  # Convert to percentage

        # Get class label
        predicted_label = posture_class_labels[predicted_class]

        app.logger.info(f"Posture Prediction: {predicted_label}, Confidence: {confidence:.2f}%")

        return jsonify({
            "posture": predicted_label,
            "confidence": confidence
        })
    except Exception as e:
        app.logger.error(f"Error in predict_posture route: {e}")
        return jsonify({"error": "Server error"}), 500
    


from flask import request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

@app.route('/api/segment-image', methods=['POST'])
def segment_image():
    try:
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Secure the filename and save the uploaded file to a temp uploads folder
        filename = secure_filename(file.filename)
        uploads_folder = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_folder, exist_ok=True)
        image_path = os.path.join(uploads_folder, filename)
        file.save(image_path)

        # Load the image in grayscale
        xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if xray_image is None:
            return jsonify({'error': 'Invalid image format or corrupted file'}), 400

        height, width = xray_image.shape

        # CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(xray_image)

        # Step 1: Vertical mask (focus on center)
        mask = np.zeros_like(enhanced_image)
        cv2.rectangle(mask, (int(width * 0.3), 0), (int(width * 0.7), height), 255, -1)
        central_region = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)

        # Step 2: Thresholding
        _, thresholded = cv2.threshold(central_region, 200, 255, cv2.THRESH_BINARY)

        # Step 3: Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Step 4: Ensure vertical continuity
        nonzero_indices = np.transpose(np.nonzero(cleaned))
        for y, x in nonzero_indices:
            if np.sum(cleaned[max(0, y - 10):min(height, y + 10), x]) == 0:
                cleaned[y, x] = 255

        # Step 5: Vertical dilation
        vertical_kernel = np.ones((15, 3), np.uint8)
        spine_full_height = cv2.dilate(cleaned, vertical_kernel, iterations=3)

        # Step 6: Filter small components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(spine_full_height, connectivity=8)
        min_height = int(height * 0.6)
        filtered_mask = np.zeros_like(labels, dtype=np.uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_HEIGHT] > min_height:
                filtered_mask[labels == i] = 255

        # Step 7: Save the final filtered mask
        segmented_dir = os.path.join(os.path.dirname(__file__), 'segmented')
        os.makedirs(segmented_dir, exist_ok=True)
        output_filename = f"segmented_{filename}"
        output_path = os.path.join(segmented_dir, output_filename)
        cv2.imwrite(output_path, filtered_mask)

        return jsonify({
            "message": "Segmentation successful",
            "output_filename": output_filename
        })

    except Exception as e:
        app.logger.error(f"Segmentation error: {e}")
        return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)    


# import cv2
# import numpy as np
# from scipy.interpolate import UnivariateSpline

# logging.basicConfig(level=logging.INFO)

# def segment_spine(image):
#     """Perform spine segmentation using thresholding and morphological operations."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_image = clahe.apply(gray)
    
#     height, width = enhanced_image.shape
#     mask = np.zeros_like(enhanced_image)
#     cv2.rectangle(mask, (int(width * 0.3), 0), (int(width * 0.7), height), 255, -1)
#     central_region = cv2.bitwise_and(enhanced_image, enhanced_image, mask=mask)
    
#     _, thresholded = cv2.threshold(central_region, 200, 255, cv2.THRESH_BINARY)
#     kernel = np.ones((5, 5), np.uint8)
#     cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
#     cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
#     vertical_kernel = np.ones((15, 3), np.uint8)
#     spine_full_height = cv2.dilate(cleaned, vertical_kernel, iterations=3)
    
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(spine_full_height, connectivity=8)
#     min_height = int(height * 0.6)
#     filtered_mask = np.zeros_like(labels, dtype=np.uint8)
#     for i in range(1, num_labels):
#         if stats[i, cv2.CC_STAT_HEIGHT] > min_height:
#             filtered_mask[labels == i] = 255
    
#     final_spine = cv2.bitwise_and(enhanced_image, enhanced_image, mask=filtered_mask)
#     return final_spine

# def skeletonize_image(image):
#     """Skeletonize the input binary image using OpenCV."""
#     # Convert the image to binary (assuming the image is already a binary mask)
#     _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
#     # Apply morphological transformations for skeletonization
#     skeleton = np.zeros(binary.shape, np.uint8)
#     size = np.size(binary)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
#     while True:
#         temp = cv2.morphologyEx(binary, cv2.MORPH_OPEN, element)
#         temp = cv2.subtract(binary, temp)
#         skeleton = cv2.bitwise_or(skeleton, temp)
#         binary = cv2.subtract(binary, temp)
        
#         if cv2.countNonZero(binary) == 0:
#             break
    
#     return skeleton

# def extract_keypoints(segmented_image):
#     """Extract keypoints from the segmented spine image using skeletonization."""
    
#     # Step 1: Skeletonization
#     skeleton = skeletonize_image(segmented_image)
#     coords = np.column_stack(np.where(skeleton > 0))
#     if len(coords) == 0:
#         raise ValueError("No skeleton points detected!")
    
#     # Step 2: Sort by y-axis (top to bottom)
#     coords = coords[np.argsort(coords[:, 0])]
#     y_vals = coords[:, 0]
#     x_vals = coords[:, 1]
    
#     # Step 3: Fit a smooth curve
#     unique_y, unique_indices = np.unique(y_vals, return_index=True)
#     unique_x = x_vals[unique_indices]
#     spline = UnivariateSpline(unique_y, unique_x, k=3, s=1)
    
#     # Step 4: Sample points along the curve
#     num_points = 50
#     sample_y = np.linspace(unique_y[0], unique_y[-1], num_points)
#     sample_x = spline(sample_y)
    
#     # Step 5: Compute deviations from the centerline
#     mid_x = np.mean(sample_x)
#     deviations = np.abs(sample_x - mid_x)
    
#     # Step 6: Identify most deviated points (Point 2 & 4)
#     half_idx = len(sample_y) // 2
#     point_2_idx = min(np.argmax(deviations[:half_idx]) + 2, half_idx - 1)
#     point_4_idx = np.argmax(deviations[half_idx:]) + half_idx
    
#     # Step 7: Find transition point (Point 3)
#     point_3_idx = (point_2_idx + point_4_idx) // 2
    
#     # Step 8: Find start (Point 1) and end (Point 5)
#     vertebrae_above = 15  # Adjust to move Point 1 higher
#     point_1_idx = max(0, point_2_idx - vertebrae_above)
#     point_5_idx = min(len(sample_x) - 1, point_4_idx + 5)
    
#     # Adjust horizontal positioning
#     adjustment_value = 13
#     key_points = {
#         "Point 1 (Start)": (sample_x[point_1_idx], sample_y[point_1_idx]),
#         "Point 2 (Max Deviation 1)": (sample_x[point_2_idx], sample_y[point_2_idx]),
#         "Point 3 (Transition)": (sample_x[point_3_idx] - adjustment_value, sample_y[point_3_idx] + 20),
#         "Point 4 (Max Deviation 2)": (sample_x[point_4_idx] + adjustment_value, sample_y[point_4_idx]),
#         "Point 5 (End)": (sample_x[point_5_idx] - adjustment_value - 5, sample_y[point_5_idx])
#     }
    
#     return key_points

# @app.route("/api/segment-and-keypoints", methods=["POST"])
# def segment_and_keypoints():
#     try:
#         if "file" not in request.files:
#             return jsonify({"error": "No file provided"}), 400

#         file = request.files["file"]
#         logging.info(f"Received file: {file.filename}")
        
#         # Log file size
#         logging.info(f"File size: {len(file.read())} bytes")
        
#         file.seek(0)  # Reset file pointer after reading for logging
#         npimg = np.frombuffer(file.read(), np.uint8)
        
#         # Log the first 100 bytes of the file for debugging
#         logging.info(f"Raw file data (first 100 bytes): {npimg[:100]}")
        
#         image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

#         # Check if image is successfully decoded
#         if image is None:
#             logging.error("Error decoding image. Possibly an unsupported format.")
#             return jsonify({"error": "Error decoding image"}), 400
        
#         # Proceed with segmentation and keypoint extraction
#         logging.info("Image decoded successfully, starting segmentation and keypoint extraction.")
#         segmented = segment_spine(image)
#         keypoints = extract_keypoints(segmented)

#         return jsonify({"keypoints": keypoints})

#     except Exception as e:
#         logging.error(f"Error in segment_and_keypoints: {e}")
#         return jsonify({"error": "Server error"}), 500
