from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("WARNING: MONGO_URI environment variable not set")

client = MongoClient(MONGO_URI)
db = client["gd"]  # Database name

# Import blueprints AFTER defining the app
from auth import auth_bp
from llm1 import llm_bp as llm1_bp
from llm2 import llm_bp as llm2_bp
from user_data import user_data_bp
from routes.screenshot_routes import screenshot_bp
# from gd_routes import gd_bp  # Import the new blueprint

# Register blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(llm1_bp)
app.register_blueprint(llm2_bp)
app.register_blueprint(user_data_bp)
app.register_blueprint(screenshot_bp)
# app.register_blueprint(gd_bp)  # Register the new blueprint

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Simple test endpoint to verify the server is working
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({"message": "API is working"}), 200

@app.route('/test', methods=['GET'])
def test_route():
    return jsonify({"status": "ok", "message": "API server is running"}), 200

if __name__ == "__main__":
    print(f"Starting Flask server on port 8080...")
    app.run(debug=True, host="0.0.0.0", port=8080) 