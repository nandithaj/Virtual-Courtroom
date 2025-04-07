from flask import Blueprint, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import os
from bson import json_util
import json
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create blueprint instead of Flask app
auth_bp = Blueprint('auth', __name__)
CORS(auth_bp, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend communication

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    print("WARNING: MONGO_URI environment variable not set")

client = MongoClient(MONGO_URI)
db = client["gd"]  # Database name
users_collection = db["users"]  # Collection name

# Google OAuth Client ID
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
if not GOOGLE_CLIENT_ID:
    print("WARNING: GOOGLE_CLIENT_ID environment variable not set")

@auth_bp.route("/api/auth/google", methods=["POST"])
def google_signin():
    try:
        data = request.get_json()
        token = data.get("token")
        
        if not token:
            return jsonify({"success": False, "error": "No token provided"}), 400

        # Verify Google ID Token
        id_info = id_token.verify_oauth2_token(token, google_requests.Request(), GOOGLE_CLIENT_ID)

        if not id_info:
            return jsonify({"success": False, "error": "Invalid token"}), 400

        user_id = id_info["sub"]  # Google user ID
        user_email = id_info["email"]
        name = id_info.get("name", "Unknown")
        picture = id_info.get("picture", "")

        # Check if user exists in MongoDB
        user = users_collection.find_one({"user_id": user_id})

        if not user:
            # Create new user entry
            new_user = {
                "user_id": user_id,
                "email": user_email,
                "name": name,
                "photo_url": picture,
                "created_at": db.command("serverStatus")["localTime"],  # Timestamp
            }
            users_collection.insert_one(new_user)
            user = new_user  # Assign new user data
        
        # Convert MongoDB document to JSON-serializable format
        user_data = json.loads(json_util.dumps(user))
        
        return jsonify({"success": True, "user": user_data})

    except ValueError as e:
        print(f"Token validation error: {e}")
        return jsonify({"success": False, "error": "Invalid token"}), 401
    except Exception as e:
        print(f"Auth error: {e}")
        traceback.print_exc()  # More detailed error logging
        return jsonify({"success": False, "error": "Authentication failed", "details": str(e)}), 500

# Add a route to verify API is working
@auth_bp.route('/api/status', methods=['GET'])
def api_status():
    """Check API status."""
    return jsonify({
        "status": "online",
        "registered_blueprints": ["auth"]
    })

if __name__ == "__main__":
    auth_bp.run(debug=True, host="0.0.0.0", port=8080)
