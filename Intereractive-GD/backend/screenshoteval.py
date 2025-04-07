import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Union
import base64

class ScreenshotEvaluator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
        # Load Haar cascade for frontal face detection
        self.frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def detect_faces(self, image: np.ndarray) -> Dict:
        """
        Detects frontal faces in an image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Dict: Dictionary containing face detection results
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect frontal faces
        faces = self.frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        return {
            "total_faces": len(faces),
            "frame_status": "Your frame looks good" if len(faces) == 1 else "Multiple people detected in frame"
        }
        
    def analyze_face(self, image_data: Union[str, bytes]) -> Dict:
        """
        Analyze a face in the given image data (either base64 string or file path) and return various metrics.
        
        Args:
            image_data (Union[str, bytes]): Either base64 encoded image data or file path
            
        Returns:
            Dict: Dictionary containing analysis results
        """
        try:
            # Handle both base64 and file path inputs
            if isinstance(image_data, str):
                if os.path.isfile(image_data):
                    # Handle file path
                    image = cv2.imread(image_data)
                else:
                    # Handle base64
                    image_bytes = base64.b64decode(image_data.split(',')[-1])  # Remove data URL prefix if present
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                # Handle raw bytes
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Unable to decode image"}
            
            # First check for multiple faces
            face_detection = self.detect_faces(image)
            if face_detection["total_faces"] != 1:
                return {
                    "error": "Multiple people detected in frame",
                    "face_detection": face_detection
                }
            
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return {"error": "No face detected"}
            
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Eye aspect ratio (EAR) calculation
            def eye_aspect_ratio(eye_points):
                A = np.linalg.norm(np.array([eye_points[1].x, eye_points[1].y]) - np.array([eye_points[5].x, eye_points[5].y]))
                B = np.linalg.norm(np.array([eye_points[2].x, eye_points[2].y]) - np.array([eye_points[4].x, eye_points[4].y]))
                C = np.linalg.norm(np.array([eye_points[0].x, eye_points[0].y]) - np.array([eye_points[3].x, eye_points[3].y]))
                return (A + B) / (2.0 * C)
            
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            left_eye = [face_landmarks[i] for i in left_eye_indices]
            right_eye = [face_landmarks[i] for i in right_eye_indices]
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            left_eye_status = "Open" if left_ear > 0.25 else "Closed"
            right_eye_status = "Open" if right_ear > 0.25 else "Closed"
            
            # Head position detection
            nose = face_landmarks[1]
            chin = face_landmarks[152]
            left_cheek = face_landmarks[234]
            right_cheek = face_landmarks[454]
            
            head_x = (left_cheek.x + right_cheek.x) / 2.0
            head_y = (nose.y + chin.y) / 2.0
            
            head_position = "Straight"
            if abs(nose.x - head_x) > 0.03:
                head_position = "Left" if nose.x < head_x else "Right"
            if abs(nose.y - head_y) > 0.03:
                head_position = "Down" if nose.y > head_y else "Up"
            
            return {
                "Left Eye Status": left_eye_status,
                "Right Eye Status": right_eye_status,
                "Head Position": head_position,
                "Left EAR": left_ear,
                "Right EAR": right_ear,
                "face_detection": face_detection
            }
        except Exception as e:
            return {"error": str(e)}
    
    def evaluate_screenshots(self, user_id: str, screenshot_data: List[str]) -> Dict:
        """
        Evaluate multiple screenshots for a user.
        
        Args:
            user_id (str): The ID of the user
            screenshot_data (List[str]): List of base64 encoded image data
            
        Returns:
            Dict: Dictionary containing evaluation results for all screenshots
        """
        results = {
            "user_id": user_id,
            "screenshots": [],
            "summary": {
                "total_screenshots": len(screenshot_data),
                "valid_screenshots": 0,
                "attention_metrics": {
                    "eyes_closed_count": 0,
                    "head_turned_count": 0
                }
            }
        }
        
        for image_data in screenshot_data:
            analysis = self.analyze_face(image_data)
            
            if "error" in analysis:
                results["screenshots"].append({
                    "error": analysis["error"]
                })
                continue
                
            results["summary"]["valid_screenshots"] += 1
            
            # Update attention metrics
            if analysis["Left Eye Status"] == "Closed" or analysis["Right Eye Status"] == "Closed":
                results["summary"]["attention_metrics"]["eyes_closed_count"] += 1
            if analysis["Head Position"] != "Straight":
                results["summary"]["attention_metrics"]["head_turned_count"] += 1
            
            results["screenshots"].append({
                "analysis": analysis
            })
        
        # Calculate attention scores
        if results["summary"]["valid_screenshots"] > 0:
            results["summary"]["attention_score"] = {
                "eyes_closed_ratio": results["summary"]["attention_metrics"]["eyes_closed_count"] / results["summary"]["valid_screenshots"],
                "head_turned_ratio": results["summary"]["attention_metrics"]["head_turned_count"] / results["summary"]["valid_screenshots"]
            }
        
        return results

# Example usage
if __name__ == "__main__":
    evaluator = ScreenshotEvaluator()
    # Example usage with a single image
    result = evaluator.analyze_face("face.jpg")
    print(result)
    
    # Example usage with multiple screenshots
    screenshots = ["screenshot1.jpg", "screenshot2.jpg"]
    evaluation = evaluator.evaluate_screenshots("user123", screenshots)
    print(evaluation)
