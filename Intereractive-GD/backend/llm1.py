from flask import Blueprint, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
import io
from gtts import gTTS
import os
import requests
import logging
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize a Blueprint for LLM routes
llm_bp = Blueprint('llm1', __name__)
CORS(llm_bp, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend communication

def init_api_keys():
    try:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OPENROUTER_API_KEY environment variable is missing")
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Log the first few characters of the API key for debugging (not the full key for security)
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        logger.info(f"Initializing OpenRouter API with key: {masked_key}")
        
        # Initialize OpenAI client with OpenRouter
        global client
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        logger.info("Successfully configured OpenRouter API")
    except Exception as e:
        logger.error(f"Error in initializing API keys: {e}", exc_info=True)
        raise

# Initialize API keys when the blueprint is created
init_api_keys()

@llm_bp.route('/api/llm1/llm', methods=['POST'])
def get_llm_response():
    try:
        data = request.get_json()
        if not data or not data.get("text"):
            return jsonify({"success": False, "error": "No text provided"}), 400

        text = data.get("text")
        topic = data.get("topic", "")
        is_initial = data.get("is_initial_message", False)
        is_user_message = data.get("is_user_message", False)

        if is_initial:
            prompt = f"""You are starting a group discussion about "{topic}". Give a simple introduction in 40-50 words that sets the context and invites others to share their views. Use plain text without any special characters or emojis."""
        elif is_user_message:
            prompt = f"""You are in a group discussion about "{topic}". A participant just said: "{text}". Respond directly to their point in 40-50 words. Use plain text without any special characters or emojis. Keep your response simple and conversational."""
        else:
            prompt = f"""You are in a group discussion about "{topic}". Respond in 40-50 words to: {text}. Use plain text without any special characters or emojis. Keep your response simple and conversational."""

        # Log the request we're about to make
        logger.info(f"Sending request to OpenRouter with prompt: {prompt}")
        
        try:
            completion = client.chat.completions.create(
                model="google/gemma-3-4b-it:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Check if completion is None
            if completion is None:
                logger.error("Received None response from OpenRouter API")
                return jsonify({"success": False, "error": "No response received from LLM API"}), 500
                
            # Log the raw response for debugging
            logger.info(f"Received response from OpenRouter: {completion}")
            
            # First try the standard OpenAI format
            try:
                response_text = completion.choices[0].message.content.strip()
            except (AttributeError, IndexError) as e:
                logger.warning(f"Failed to get response in standard format: {e}")
                # Try alternative response formats
                if hasattr(completion, 'text'):
                    response_text = completion.text.strip()
                elif isinstance(completion, dict):
                    if 'choices' in completion and len(completion['choices']) > 0:
                        response_text = completion['choices'][0].get('message', {}).get('content', '').strip()
                    else:
                        response_text = completion.get('text', '').strip()
                else:
                    logger.error("Unable to extract response text from completion")
                    return jsonify({"success": False, "error": "Invalid response format from LLM API"}), 500
        except Exception as api_error:
            logger.error(f"Error calling OpenRouter API: {str(api_error)}", exc_info=True)
            return jsonify({"success": False, "error": f"API call failed: {str(api_error)}"}), 500

        if not response_text:
            logger.error("Empty response text from LLM API")
            return jsonify({"success": False, "error": "Empty response from LLM API"}), 500

        if len(response_text.split()) > 55:
            response_text = ' '.join(response_text.split()[:50]) + '...'

        logger.info(f"Response text: {response_text}")

        return jsonify({
            "success": True,
            "response": response_text,
            "model_used": "google/gemma-3-4b-it:free"
        })

    except Exception as e:
        logger.error(f"Error in get_llm_response: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

@llm_bp.route('/api/llm1/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data or not data.get("text"):
            return jsonify({"success": False, "error": "No text provided"}), 400

        text = data.get("text")
        tts = gTTS(text=text, lang='en', tld='com.au')
        
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        
        return send_file(audio_stream, mimetype="audio/mp3")
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# This is an alternative implementation if you want to try another option
@llm_bp.route('/api/tts/alt', methods=['POST'])
def alt_text_to_speech():
    """Alternative TTS using pyttsx3 with a sweet female voice."""
    try:
        import pyttsx3
        import tempfile
        import os
        
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        # Initialize the pyttsx3 engine
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        
        # Select a female voice (index may vary by system)
        # Usually the second voice (index 1) is female on most systems
        engine.setProperty('voice', voices[1].id)
        
        # Set a higher pitch for a sweeter sound
        engine.setProperty('rate', 150)  # Speed
        engine.setProperty('volume', 0.9)  # Volume
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_filename = temp_file.name
        temp_file.close()
        
        # Save to the temporary file
        engine.save_to_file(text, temp_filename)
        engine.runAndWait()
        
        # Return the file
        return send_file(temp_filename, mimetype="audio/mp3")
        
    except Exception as e:
        print(f"Error in pyttsx3 conversion: {e}")
        return jsonify({"success": False, "error": f"Alternative TTS failed: {str(e)}"}), 500

@llm_bp.route('/api/llm1/test', methods=['GET'])
def test_api_connection():
    """Test endpoint to verify API connection and key validity."""
    try:
        # Simple test request
        completion = client.chat.completions.create(
            model="google/gemma-3-4b-it:free",
            messages=[
                {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
            ]
        )
        
        if completion is None:
            return jsonify({"success": False, "error": "No response received from API"}), 500
            
        # Try to extract a response
        try:
            response_text = completion.choices[0].message.content.strip()
        except (AttributeError, IndexError):
            response_text = "API responded but in unexpected format"
            
        return jsonify({
            "success": True,
            "message": "API connection successful",
            "response": response_text
        })
    except Exception as e:
        logger.error(f"API test failed: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": f"API test failed: {str(e)}"}), 500