from flask import Blueprint, jsonify, request
from screenshoteval import ScreenshotEvaluator
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["gd"]

# Create blueprint
screenshot_bp = Blueprint('screenshot', __name__)

# Initialize evaluator
evaluator = ScreenshotEvaluator()

@screenshot_bp.route('/api/screenshots/evaluate/<user_id>', methods=['GET'])
def evaluate_user_screenshots(user_id):
    """
    Evaluate all screenshots for a specific user and return the analysis results.
    """
    try:
        # Get the user's screenshots from MongoDB
        user_data = db.user_speech.find_one({"user_id": user_id})
        if not user_data or "screenshots" not in user_data:
            return jsonify({
                "error": "No screenshots found for this user",
                "user_id": user_id
            }), 404

        # Get the image data of all screenshots
        screenshot_data = [screenshot["image_data"] for screenshot in user_data["screenshots"]]
        
        # Evaluate the screenshots
        evaluation_results = evaluator.evaluate_screenshots(user_id, screenshot_data)
        
        # Store the evaluation results in MongoDB
        db.screenshot_evaluations.update_one(
            {"user_id": user_id},
            {"$set": evaluation_results},
            upsert=True
        )
        
        return jsonify(evaluation_results), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "user_id": user_id
        }), 500

@screenshot_bp.route('/api/screenshots/evaluation/<user_id>', methods=['GET'])
def get_evaluation_results(user_id):
    """
    Get the stored evaluation results for a specific user.
    """
    try:
        evaluation = db.screenshot_evaluations.find_one({"user_id": user_id})
        if not evaluation:
            return jsonify({
                "error": "No evaluation results found for this user",
                "user_id": user_id
            }), 404
            
        # Remove MongoDB _id field
        evaluation.pop('_id', None)
        return jsonify(evaluation), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "user_id": user_id
        }), 500 