from flask import Blueprint, request, jsonify
from bson import ObjectId
import json
from datetime import datetime
import base64
import traceback
import logging
import requests
import os
from dotenv import load_dotenv
import textstat
import language_tool_python
from collections import Counter

# Create a Blueprint for user data routes
user_data_bp = Blueprint('user_data', __name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "http://localhost:5173")
SITE_NAME = os.getenv("SITE_NAME", "Interactive-GD")

def get_user_speech_collection(db):
    """Returns the user_speech collection from the database."""
    return db["user_speech"]

def get_qwen_evaluation(prompt):
    """Get evaluation from Qwen model via OpenRouter API."""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            json={
                "model": "qwen/qwen2.5-vl-32b-instruct:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )
        
        if response.status_code == 200:
            try:
                result = response.json()
                # Check if result and result['choices'] exist and have content
                if not result or 'choices' not in result or not result['choices']:
                    logger.error("Invalid response from Qwen API: missing choices")
                    return {"success": False, "error": "Invalid response from Qwen API: missing choices"}
                
                if not result['choices'][0] or 'message' not in result['choices'][0]:
                    logger.error("Invalid response from Qwen API: missing message")
                    return {"success": False, "error": "Invalid response from Qwen API: missing message"}
                
                if not result['choices'][0]['message'] or 'content' not in result['choices'][0]['message']:
                    logger.error("Invalid response from Qwen API: missing content")
                    return {"success": False, "error": "Invalid response from Qwen API: missing content"}
                
                # Extract the response text from the completion
                evaluation_text = result['choices'][0]['message']['content']
                # Parse the JSON string from the response
                try:
                    return {"success": True, "data": json.loads(evaluation_text)}
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON from Qwen response: {e}")
                    return {"success": False, "error": f"Invalid JSON in Qwen response: {str(e)}"}
            except Exception as e:
                logger.error(f"Error processing Qwen API response: {e}")
                return {"success": False, "error": f"Error processing Qwen API response: {str(e)}"}
        else:
            error_msg = f"API request failed with status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        logger.error(f"Failed to get evaluation from Qwen: {str(e)}")
        return {"success": False, "error": f"Failed to get evaluation from Qwen: {str(e)}"}

@user_data_bp.route('/api/user/speech', methods=['POST'])
def store_speech():
    """Store user's speech text in MongoDB."""
    try:
        logger.info("Speech storage request received")
        
        try:
            from auth import db
            logger.info("Successfully imported db from auth")
        except ImportError as ie:
            logger.error(f"Failed to import db from auth: {ie}")
            return jsonify({"success": False, "error": "Server configuration error"}), 500
        
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        user_id = data.get("user_id")
        speech_text = data.get("text", "").strip()
        topic = data.get("topic", "").strip()
        
        logger.info(f"Processing speech for user {user_id}, topic: {topic}")
        
        if not user_id or not speech_text:
            logger.error("Missing required data")
            return jsonify({"success": False, "error": "Missing required data"}), 400
            
        collection = get_user_speech_collection(db)
        logger.info(f"Retrieved collection: {collection.name}")
        
        # Create speech entry with timestamp
        speech_entry = {
            "timestamp": datetime.utcnow(),
            "text": speech_text
        }
        
        try:
            # Find user's document
            user_speech_doc = collection.find_one({"user_id": user_id})
            logger.info(f"Found existing user document: {bool(user_speech_doc)}")
            
            if user_speech_doc:
                # Update existing document
                logger.info("Updating existing document")
                result = collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {"topic": topic},
                        "$push": {"speech_entries": speech_entry}
                    }
                )
                logger.info(f"Update result: matched={result.matched_count}, modified={result.modified_count}")
                
                if result.modified_count == 0:
                    logger.error("Failed to update document")
                    return jsonify({"success": False, "error": "Failed to update document"}), 500
            else:
                # Create new document
                logger.info("Creating new document")
                result = collection.insert_one({
                    "user_id": user_id,
                    "topic": topic,
                    "speech_entries": [speech_entry],
                    "screenshots": []
                })
                logger.info(f"Insert result: inserted_id={result.inserted_id}")
                
                if not result.inserted_id:
                    logger.error("Failed to insert document")
                    return jsonify({"success": False, "error": "Failed to insert document"}), 500
            
            # Verify the update/insert
            updated_doc = collection.find_one({"user_id": user_id})
            if not updated_doc:
                logger.error("Failed to verify document update")
                return jsonify({"success": False, "error": "Failed to verify document update"}), 500
            
            speech_entries_count = len(updated_doc.get('speech_entries', []))
            logger.info(f"Successfully stored speech entry. Total entries: {speech_entries_count}")
            
            return jsonify({
                "success": True, 
                "message": "Speech stored successfully",
                "entries_count": speech_entries_count
            })
            
        except Exception as db_error:
            logger.error(f"Database operation error: {db_error}")
            return jsonify({"success": False, "error": f"Database error: {str(db_error)}"}), 500
            
    except Exception as e:
        logger.error(f"Error storing speech: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/screenshot', methods=['POST'])
def store_screenshot():
    """Store screenshot image in MongoDB."""
    try:
        # Log the request
        logger.info("Screenshot upload request received")
        
        try:
            from auth import db
            logger.info("Successfully imported db from auth")
        except ImportError as ie:
            logger.error(f"Failed to import db from auth: {ie}")
            return jsonify({"success": False, "error": "Server configuration error"}), 500
        
        data = request.get_json()
        user_id = data.get("user_id")
        image_data = data.get("image_data")  # Base64 encoded image
        topic = data.get("topic", "")
        
        if not user_id or not image_data:
            return jsonify({"success": False, "error": "Missing required data"}), 400
        
        # Print the first 100 characters to debug (avoid logging entire image)
        print(f"Received image data from user {user_id}, length: {len(image_data)} chars")
        
        # Check if image data is too large
        if len(image_data) > 1024 * 1024 * 5:  # 5MB limit
            print("Image data exceeds size limit, compressing...")
            # We'll trim the data for now (in production you'd compress it instead)
            image_data = image_data[:1024 * 1024 * 2]  # Trim to 2MB
            
        collection = get_user_speech_collection(db)
        
        # Find user's document or create a new one
        user_speech_doc = collection.find_one({"user_id": user_id})
        
        screenshot_entry = {
            "timestamp": datetime.utcnow(),
            "image_data": image_data
        }
        
        try:
            if user_speech_doc:
                # Update existing document - update topic and push to screenshots array
                collection.update_one(
                    {"user_id": user_id},
                    {
                        "$set": {"topic": topic},
                        "$push": {"screenshots": screenshot_entry}
                    }
                )
            else:
                # Create new document with the desired structure
                collection.insert_one({
                    "user_id": user_id,
                    "topic": topic,
                    "speech_entries": [],
                    "screenshots": [screenshot_entry]
                })
                
            print(f"Successfully stored screenshot for user {user_id}")
            return jsonify({"success": True})
            
        except Exception as db_error:
            print(f"Database error when storing screenshot: {db_error}")
            if "document too large" in str(db_error).lower():
                # MongoDB has a 16MB document size limit
                return jsonify({
                    "success": False, 
                    "error": "Screenshot too large for database storage"
                }), 413
            raise
            
    except Exception as e:
        print(f"Error storing screenshot: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify API is working."""
    return jsonify({
        "success": True,
        "message": "User data API is working correctly"
    })

@user_data_bp.route('/api/user/<user_id>/data', methods=['GET'])
def get_user_data(user_id):
    """Get user's data including screenshots."""
    try:
        from auth import db
        
        if not user_id:
            return jsonify({"success": False, "error": "User ID required"}), 400
            
        collection = get_user_speech_collection(db)
        user_data = collection.find_one({"user_id": user_id})
        
        # Add debug logging
        logger.info(f"Type of user_data: {type(user_data)}")
        logger.info(f"Content of user_data: {user_data}")
        
        if not user_data:
            return jsonify({"success": False, "error": "User not found"}), 404
            
        # Ensure user_data is a dictionary
        if not isinstance(user_data, dict):
            logger.error(f"Invalid user_data type: {type(user_data)}")
            return jsonify({"success": False, "error": "Invalid data format in database"}), 500
            
        # Extract all speech entries with safe access
        try:
            speech_entries = user_data.get("speech_entries", [])
            if not isinstance(speech_entries, list):
                logger.error(f"Invalid speech_entries type: {type(speech_entries)}")
                return jsonify({"success": False, "error": "Invalid speech entries format"}), 500
                
            topic = user_data.get("topic")
            if not isinstance(topic, str):
                logger.error(f"Invalid topic type: {type(topic)}")
                return jsonify({"success": False, "error": "Invalid topic format"}), 500
        except Exception as e:
            logger.error(f"Error accessing user_data fields: {e}")
            return jsonify({"success": False, "error": "Error accessing user data"}), 500
        
        if not speech_entries:
            return jsonify({"success": False, "error": "No speech entries found"}), 404
            
        # Combine all speech entries into one text with validation
        try:
            full_speech = " ".join([entry.get("text", "") for entry in speech_entries if isinstance(entry, dict)])
        except Exception as e:
            logger.error(f"Error combining speech entries: {e}")
            return jsonify({"success": False, "error": "Error processing speech entries"}), 500
        
        # Convert MongoDB ObjectId to string
        user_data["_id"] = str(user_data["_id"])
        
        # Convert datetime objects to strings for JSON serialization
        if "screenshots" in user_data:
            for screenshot in user_data["screenshots"]:
                if isinstance(screenshot, dict):  # Ensure screenshot is a dictionary
                    if "timestamp" in screenshot:
                        screenshot["timestamp"] = screenshot["timestamp"].isoformat()
                    # Ensure image_data is a string
                    if "image_data" in screenshot and not isinstance(screenshot["image_data"], str):
                        screenshot["image_data"] = str(screenshot["image_data"])
        
        if "speech_entries" in user_data:
            for entry in user_data["speech_entries"]:
                if isinstance(entry, dict) and "timestamp" in entry:
                    entry["timestamp"] = entry["timestamp"].isoformat()
        
        return jsonify({
            "success": True,
            "data": user_data
        })
        
    except Exception as e:
        print(f"Error retrieving user data: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/test-speech', methods=['GET'])
def test_speech_storage():
    """Test speech storage API."""
    try:
        from auth import db
        
        # Test data
        test_user_id = "test_user_123"
        test_speech = "This is a test speech entry."
        test_topic = "Test Topic"
        
        # Get collection
        collection = get_user_speech_collection(db)
        
        # Create test entry
        speech_entry = {
            "timestamp": datetime.utcnow(),
            "text": test_speech
        }
        
        # Check if test user exists
        test_user = collection.find_one({"user_id": test_user_id})
        
        if test_user is not None:  # Changed from if test_user:
            # Update existing user
            result = collection.update_one(
                {"user_id": test_user_id},
                {
                    "$set": {"topic": test_topic},
                    "$push": {"speech_entries": speech_entry}
                }
            )
            operation = "updated"
        else:
            # Create new user
            result = collection.insert_one({
                "user_id": test_user_id,
                "topic": test_topic,
                "speech_entries": [speech_entry],
                "screenshots": []
            })
            operation = "created"
            
        # Get the result
        test_user = collection.find_one({"user_id": test_user_id})
        
        if test_user is not None:  # Changed from if test_user:
            # Remove ObjectId for JSON serialization
            test_user["_id"] = str(test_user["_id"])
            
            return jsonify({
                "success": True,
                "message": f"Test user {operation} successfully",
                "user_data": test_user
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to retrieve test user after operation"
            }), 500
            
    except Exception as e:
        logger.error(f"Error in test speech storage: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/<user_id>/gd-evaluation', methods=['GET'])
def evaluate_gd_performance(user_id):
    """Fetch user's GD speech and evaluate topic coverage using Qwen."""
    try:
        logger.info(f"Received GD evaluation request for user_id: {user_id}")
        
        from auth import db
        logger.info("Successfully imported db from auth")
        
        if not user_id:
            logger.error("No user_id provided")
            return jsonify({"success": False, "error": "User ID required"}), 400
            
        if not OPENROUTER_API_KEY:
            logger.error("OpenRouter API key not configured")
            return jsonify({
                "success": False, 
                "error": "OpenRouter API key not configured. Please check your environment variables."
            }), 500
            
        collection = get_user_speech_collection(db)
        logger.info(f"Retrieved collection: {collection.name}")
        
        # Log the query we're about to make
        logger.info(f"Querying MongoDB for user_id: {user_id}")
        user_data = collection.find_one({"user_id": str(user_id)})
        
        # Detailed logging of the response
        logger.info(f"Type of user_data: {type(user_data)}")
        logger.info(f"Content of user_data: {user_data}")
        
        if not user_data:
            logger.error(f"No data found for user_id: {user_id}")
            return jsonify({"success": False, "error": "User not found"}), 404
            
        # Ensure user_data is a dictionary
        if not isinstance(user_data, dict):
            logger.error(f"Invalid user_data type: {type(user_data)}")
            return jsonify({"success": False, "error": "Invalid data format in database"}), 500
            
        # Extract all speech entries with safe access
        try:
            speech_entries = user_data.get("speech_entries", [])
            logger.info(f"Type of speech_entries: {type(speech_entries)}")
            logger.info(f"Number of speech entries: {len(speech_entries)}")
            
            if not isinstance(speech_entries, list):
                logger.error(f"Invalid speech_entries type: {type(speech_entries)}")
                return jsonify({"success": False, "error": "Invalid speech entries format"}), 500
                
            topic = user_data.get("topic", "")
            logger.info(f"Type of topic: {type(topic)}")
            logger.info(f"Topic value: {topic}")
            
            if not isinstance(topic, str):
                logger.error(f"Invalid topic type: {type(topic)}")
                return jsonify({"success": False, "error": "Invalid topic format"}), 500
        except Exception as e:
            logger.error(f"Error accessing user_data fields: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({"success": False, "error": "Error accessing user data"}), 500
        
        if not speech_entries:
            logger.error("No speech entries found")
            return jsonify({"success": False, "error": "No speech entries found"}), 404
            
        # Combine all speech entries into one text with validation
        try:
            full_speech = " ".join([entry.get("text", "") for entry in speech_entries if isinstance(entry, dict)])
            logger.info(f"Combined speech length: {len(full_speech)}")
        except Exception as e:
            logger.error(f"Error combining speech entries: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({"success": False, "error": "Error processing speech entries"}), 500
        
        try:
            # Make request to OpenRouter API with Qwen model
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": SITE_URL,
                    "X-Title": SITE_NAME,
                },
                json={
                    "model": "qwen/qwen2.5-vl-32b-instruct:free",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are a JSON-only response evaluator.
CRITICAL INSTRUCTIONS:
1. Return ONLY a single, valid JSON object
2. NO text before or after the JSON
3. NO explanations or comments
4. NO markdown formatting
5. NO trailing commas in JSON
6. NO extra whitespace outside the JSON structure
7. Ensure all JSON strings are properly escaped
8. All numeric scores must be between 0.0 and 1.0"""
                        },
                        {
                            "role": "user",
                            "content": f"""Evaluate this Group Discussion speech.
Topic: "{topic}"
Speech: {full_speech}

Respond with this exact JSON structure (no other text):
{{
"topic_coverage":{{
"score":0.75,
"analysis":"Brief analysis of topic coverage",
"key_points_covered":["Key point 1","Key point 2"],
"missing_points":["Missing point 1","Missing point 2"]
}},
"depth_of_analysis":{{
"score":0.8,
"analysis":"Analysis of depth"
}},
"relevance":{{
"score":0.85,
"analysis":"Analysis of relevance"
}},
"structure":{{
"score":0.7,
"analysis":"Analysis of structure"
}},
"overall_score":0.78,
"summary":"Brief overall summary",
"suggestions":["Suggestion 1","Suggestion 2"]
}}"""
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                    "response_format": { "type": "json_object" }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Check if result and result['choices'] exist and have content
                if not result or 'choices' not in result or not result['choices']:
                    logger.error("Invalid response from Qwen API: missing choices")
                    return jsonify({
                        "success": False,
                        "error": "Invalid response from Qwen API: missing choices"
                    }), 500
                
                if not result['choices'][0] or 'message' not in result['choices'][0]:
                    logger.error("Invalid response from Qwen API: missing message")
                    return jsonify({
                        "success": False,
                        "error": "Invalid response from Qwen API: missing message"
                    }), 500
                
                if not result['choices'][0]['message'] or 'content' not in result['choices'][0]['message']:
                    logger.error("Invalid response from Qwen API: missing content")
                    return jsonify({
                        "success": False,
                        "error": "Invalid response from Qwen API: missing content"
                    }), 500
                
                # Extract and clean the evaluation text from the response
                evaluation_text = result['choices'][0]['message']['content']
                
                # Clean the response text
                evaluation_text = evaluation_text.strip()
                # Remove any potential markdown code block markers
                evaluation_text = evaluation_text.replace('```json', '').replace('```', '')
                evaluation_text = evaluation_text.strip()
                
                # Debug logging
                print("Raw API response:", result)
                print("Cleaned evaluation text:", evaluation_text)
                
                try:
                    # Parse the JSON string from the response
                    evaluation_result = json.loads(evaluation_text)
                    
                    # Validate the required fields and data types
                    required_fields = {
                        "topic_coverage": dict,
                        "depth_of_analysis": dict,
                        "relevance": dict,
                        "structure": dict,
                        "overall_score": (int, float),
                        "summary": str,
                        "suggestions": list
                    }
                    
                    for field, expected_type in required_fields.items():
                        if field not in evaluation_result:
                            raise ValueError(f"Missing required field: {field}")
                        if not isinstance(evaluation_result[field], expected_type):
                            raise ValueError(f"Invalid type for field {field}: expected {expected_type}")
                    
                    # Additional validation for nested fields
                    for section in ["topic_coverage", "depth_of_analysis", "relevance", "structure"]:
                        if "score" not in evaluation_result[section]:
                            raise ValueError(f"Missing score in {section}")
                        if not isinstance(evaluation_result[section]["score"], (int, float)):
                            raise ValueError(f"Invalid score type in {section}")
                        if evaluation_result[section]["score"] < 0 or evaluation_result[section]["score"] > 1:
                            raise ValueError(f"Score must be between 0 and 1 in {section}")
                    
                    # Store the evaluation result in MongoDB
                    collection.update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "gd_evaluation": {
                                    "timestamp": datetime.utcnow(),
                                    "evaluation": evaluation_result
                                }
                            }
                        }
                    )
                    
                    return jsonify({
                        "success": True,
                        "evaluation": evaluation_result
                    })
                    
                except json.JSONDecodeError as je:
                    print(f"JSON parsing error: {je}")
                    print(f"Problematic text: {evaluation_text}")
                    return jsonify({
                        "success": False,
                        "error": f"Invalid JSON response from model: {str(je)}"
                    }), 500
                except ValueError as ve:
                    print(f"Validation error: {ve}")
                    return jsonify({
                        "success": False,
                        "error": f"Invalid response structure: {str(ve)}"
                    }), 500
                
            else:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_details = response.json()
                    error_msg += f": {json.dumps(error_details)}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"Error getting Qwen evaluation: {e}")
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": f"Failed to get evaluation from Qwen: {str(e)}"
            }), 500
            
    except Exception as e:
        logger.error(f"Error evaluating GD performance: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/<user_id>/screenshots', methods=['GET'])
def get_user_screenshots(user_id):
    logger.info(f"Fetching screenshots for user_id: {user_id}")
    """Get all screenshots for a specific user."""
    try:
        logger.info(f"Fetching screenshots for user_id: {user_id}")
        
        from auth import db
        logger.info("Successfully imported db from auth")
        
        if not user_id:
            logger.error("No user_id provided")
            return jsonify({"success": False, "error": "User ID required"}), 400
            
        collection = get_user_speech_collection(db)
        logger.info(f"Retrieved collection: {collection.name}")
        
        # Log the query we're about to make
        query = {"user_id": str(user_id)}
        logger.info(f"Querying MongoDB with: {query}")
        
        # Find user's document
        user_data = collection.find_one(query)
        
        # Log the result
        logger.info(f"Query result: {user_data}")
        
        if not user_data:
            logger.error(f"No data found for user_id: {user_id}")
            # Let's check if the user exists in any format
            all_users = list(collection.find({}, {"user_id": 1}))
            logger.info(f"Available users in database: {all_users}")
            return jsonify({
                "success": False, 
                "error": "User not found",
                "debug_info": {
                    "searched_user_id": user_id,
                    "available_users": [user.get("user_id") for user in all_users]
                }
            }), 404
            
        # Get screenshots array from user data
        screenshots = user_data.get("screenshots", [])
        logger.info(f"Found {len(screenshots)} screenshots for user")
        
        # Convert datetime objects to ISO format strings for JSON serialization
        for screenshot in screenshots:
            if isinstance(screenshot, dict) and "timestamp" in screenshot:
                screenshot["timestamp"] = screenshot["timestamp"].isoformat()
        
        return jsonify({
            "success": True,
            "data": {
                "screenshots": screenshots
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching screenshots: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/speaking-time', methods=['POST'])
def store_speaking_time():
    """Store user's speaking time data in MongoDB."""
    try:
        logger.info("Speaking time storage request received")
        
        try:
            from auth import db
            logger.info("Successfully imported db from auth")
        except ImportError as ie:
            logger.error(f"Failed to import db from auth: {ie}")
            return jsonify({"success": False, "error": "Server configuration error"}), 500
        
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
        user_id = data.get("user_id")
        session_id = data.get("session_id")
        speaking_duration = data.get("speaking_duration", 0)  # in seconds
        total_duration = data.get("total_duration", 0)  # in seconds
        
        logger.info(f"Processing speaking time for user {user_id}, session: {session_id}")
        
        if not user_id or not session_id:
            logger.error("Missing required data")
            return jsonify({"success": False, "error": "Missing required data"}), 400
            
        collection = get_user_speech_collection(db)
        logger.info(f"Retrieved collection: {collection.name}")
        
        # Create speaking time entry
        speaking_time_entry = {
            "timestamp": datetime.utcnow(),
            "user_id": user_id,
            "session_id": session_id,
            "speaking_duration": speaking_duration,
            "total_duration": total_duration,
            "percentage": (speaking_duration / total_duration * 100) if total_duration > 0 else 0
        }
        
        # Update or insert the speaking time entry
        result = collection.update_one(
            {"user_id": user_id, "session_id": session_id},
            {"$set": speaking_time_entry},
            upsert=True
        )
        
        logger.info(f"Successfully stored speaking time entry")
        
        return jsonify({
            "success": True,
            "message": "Speaking time stored successfully",
            "percentage": speaking_time_entry["percentage"]
        })
        
    except Exception as e:
        logger.error(f"Error storing speaking time: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@user_data_bp.route('/api/user/speaking-stats/<user_id>', methods=['GET'])
def get_speaking_stats(user_id):
    """Get user's speaking time statistics."""
    try:
        logger.info(f"Getting speaking stats for user {user_id}")
        
        try:
            from auth import db
        except ImportError as ie:
            logger.error(f"Failed to import db from auth: {ie}")
            return jsonify({"success": False, "error": "Server configuration error"}), 500
        
        collection = get_user_speech_collection(db)
        
        # Get all speaking time entries for the user
        entries = list(collection.find(
            {"user_id": user_id},
            {"speaking_duration": 1, "total_duration": 1, "percentage": 1, "_id": 0}
        ))
        
        if not entries:
            return jsonify({
                "success": True,
                "average_percentage": 0,
                "total_sessions": 0
            })
        
        # Calculate average percentage
        total_percentage = sum(entry.get("percentage", 0) for entry in entries)
        average_percentage = total_percentage / len(entries)
        
        return jsonify({
            "success": True,
            "average_percentage": round(average_percentage, 2),
            "total_sessions": len(entries)
        })
        
    except Exception as e:
        logger.error(f"Error getting speaking stats: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# @user_data_bp.route('/api/users/list', methods=['GET'])
# def list_all_users():
    """List all users in the database."""
    try:
        from auth import db
        collection = get_user_speech_collection(db)
        
        # Get all users with their IDs
        users = list(collection.find({}, {"user_id": 1, "topic": 1}))
        
        # Convert ObjectId to string for JSON serialization
        for user in users:
            user["_id"] = str(user["_id"])
        
        return jsonify({
            "success": True,
            "data": {
                "users": users
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    
def evaluate_text_speech(text_list):
    tool = language_tool_python.LanguageTool('en-US')
    readability_scores = []
    grammar_scores = []
    repetitiveness_scores = []
    
    for text in text_list:
        if not text.strip():  # Check for empty text
            readability_scores.append(0)
            grammar_scores.append(0)
            repetitiveness_scores.append(0)
            continue
        
        # Readability Score (Flesch-Kincaid Index)
        readability = textstat.flesch_reading_ease(text)
        if readability > 70:
            readability_score = 100
        elif 50 <= readability <= 70:
            readability_score = 70
        else:
            readability_score = 40
        readability_scores.append(readability_score)
        
        # Grammar & Spelling Check
        matches = tool.check(text)
        num_errors = len(matches)
        if num_errors == 0:
            grammar_score = 100
        elif num_errors <= 5:
            grammar_score = 80
        elif num_errors <= 10:
            grammar_score = 60
        else:
            grammar_score = 40
        grammar_scores.append(grammar_score)
        
        # Repetitiveness Detection
        words = text.lower().split()
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 3}
        repetitiveness = len(repeated_words)
        
        if repetitiveness == 0:
            repetitiveness_score = 100
        elif repetitiveness <= 3:
            repetitiveness_score = 80
        elif repetitiveness <= 6:
            repetitiveness_score = 60
        else:
            repetitiveness_score = 40
        repetitiveness_scores.append(repetitiveness_score)
    
    # Final Percentage Scores
    final_readability = sum(readability_scores) / len(text_list)
    final_grammar = sum(grammar_scores) / len(text_list)
    final_repetitiveness = sum(repetitiveness_scores) / len(text_list)
    
    return {
        "final_readability_score": final_readability,
        "final_grammar_score": final_grammar,
        "final_repetitiveness_score": final_repetitiveness
    }
    
@user_data_bp.route('/api/grammar/<user_id>', methods=['GET'])
def get_grammar_scores(user_id):
    try:
        logger.info(f"Received request for grammar scores for user_id: {user_id}")
        
        # Get user's text data from the database
        from auth import db
        logger.info("Successfully imported db from auth")

        
        collection = get_user_speech_collection(db)
        logger.info(f"Querying speech collection for user_id: {user_id}")
        
        # Find the user document
        user_doc = collection.find_one({"user_id": str(user_id)})
        if not user_doc:
            logger.info(f"No user document found for user_id: {user_id}")
            return jsonify({
                "final_readability_score": 0,
                "final_grammar_score": 0,
                "final_repetitiveness_score": 0
            })
        
        # Extract texts from speech_entries
        user_texts = []
        if "speech_entries" in user_doc and isinstance(user_doc["speech_entries"], list):
            for entry in user_doc["speech_entries"]:
                if isinstance(entry, dict) and "text" in entry:
                    user_texts.append(entry["text"])
        
        logger.info(f"Extracted {len(user_texts)} texts from speech entries")
        
        if not user_texts:
            logger.info(f"No speech texts found for user_id: {user_id}")
            return jsonify({
                "final_readability_score": 0,
                "final_grammar_score": 0,
                "final_repetitiveness_score": 0
            })
        
        # Evaluate the texts
        logger.info("Starting text evaluation")
        scores = evaluate_text_speech(user_texts)
        logger.info(f"Evaluation complete. Scores: {scores}")
        
        return jsonify(scores)
    except Exception as e:
        logger.error(f"Error in grammar evaluation: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500
    

