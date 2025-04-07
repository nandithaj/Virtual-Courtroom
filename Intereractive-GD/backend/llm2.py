from flask import Blueprint, request, jsonify, send_file
from flask_cors import CORS
import io
from gtts import gTTS
import os
import requests
import json
from openai import OpenAI
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize a Blueprint for LLM routes
llm_bp = Blueprint('llm2', __name__)
CORS(llm_bp, resources={r"/*": {"origins": "*"}})  # Enable CORS for frontend communication

# Get OpenRouter API key from environment variables
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY environment variable not set")
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# Initialize OpenAI client with OpenRouter base URL
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Interactive-GD"
        }
    )
    logger.info("Successfully configured OpenRouter API")
except Exception as e:
    logger.error(f"Failed to configure OpenRouter API: {e}")
    raise

# Global variables to track conversation state
is_user_speaking = False
last_message = None
last_topic = None
is_ai_speaking = False
conversation_started = False
current_speaker = None  # Track which LLM is currently speaking

@llm_bp.route('/api/llm2/llm', methods=['POST'])
def get_llm_response():
    global is_user_speaking, last_message, last_topic, is_ai_speaking, conversation_started, current_speaker
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        text = data.get("text")
        topic = data.get("topic", "")
        is_user_message = data.get("is_user_message", True)
        user_interrupted = data.get("user_interrupted", False)
        is_initial_message = data.get("is_initial_message", False)
        from_llm1 = data.get("from_llm1", False)
        conversation_history = data.get("conversation_history", [])

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        logger.info(f"Received request - Topic: {topic}, Is initial: {is_initial_message}, Is user message: {is_user_message}, From LLM1: {from_llm1}")

        # Handle user interruption
        if user_interrupted:
            is_user_speaking = True
            is_ai_speaking = False
            current_speaker = None
            last_message = text
            last_topic = topic
            return jsonify({"success": True, "response": "User is speaking, waiting for their turn to finish."})

        # Create appropriate prompt based on context
        if is_initial_message and not conversation_started:
            conversation_started = True
            is_ai_speaking = True
            current_speaker = "llm2"
            prompt = f"""
            You are starting a group discussion about "{topic}". Begin the discussion with a brief introduction 
            (maximum 40 words) that sets the context and invites others to share their perspectives. Be engaging 
            and natural, like a real discussion moderator.
            
            Topic: {topic}
            """
        elif from_llm1:
            is_ai_speaking = True
            current_speaker = "llm2"
            # Create a context-aware prompt using conversation history
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
            prompt = f"""
            You are a participant in a group discussion about "{topic}". Respond to the following message in a 
            brief way (maximum 40 words). Consider the recent conversation context and provide a fresh perspective.
            Be natural and conversational, like a real participant in a group discussion.
            
            Recent conversation:
            {history_context}
            
            Current topic: {topic}
            Previous speaker says: {text}
            """
        elif is_user_message:
            is_user_speaking = False
            is_ai_speaking = True
            current_speaker = "llm2"
            # Create a context-aware prompt using conversation history
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
            prompt = f"""
            You are a participant in a group discussion about "{topic}". Respond to the following message in a 
            brief way (maximum 40 words). Consider the recent conversation context and provide a fresh perspective.
            Be natural and conversational, like a real participant in a group discussion.
            
            Recent conversation:
            {history_context}
            
            Current topic: {topic}
            User says: {text}
            """
        else:
            # Handle the case where none of the above conditions are met
            # This could be a continuation of the conversation
            is_ai_speaking = True
            current_speaker = "llm2"
            history_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
            prompt = f"""
            You are a participant in a group discussion about "{topic}". Continue the discussion in a 
            brief way (maximum 40 words). Consider the recent conversation context and provide a fresh perspective.
            Be natural and conversational, like a real participant in a group discussion.
            
            Recent conversation:
            {history_context}
            
            Current topic: {topic}
            Continue the discussion about: {text}
            """

        try:
            # Add retry logic for API calls
            max_retries = 3
            retry_count = 0
            last_error = None

            while retry_count < max_retries:
                try:
                    completion = client.chat.completions.create(
                        model="meta-llama/llama-3.2-3b-instruct:free",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a participant in a group discussion. Provide brief, natural responses that build on the conversation without repeating previous points."
                            },
                            *[{"role": msg["role"], "content": msg["content"]} for msg in conversation_history[-5:]],
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=100
                    )
                    
                    # Check if completion and completion.choices exist and have content
                    if not completion or not hasattr(completion, 'choices') or not completion.choices:
                        logger.error("Invalid completion response: missing choices")
                        return jsonify({"success": False, "error": "Invalid response from LLM API: missing choices"}), 500
                    
                    if not completion.choices[0] or not hasattr(completion.choices[0], 'message'):
                        logger.error("Invalid completion response: missing message")
                        return jsonify({"success": False, "error": "Invalid response from LLM API: missing message"}), 500
                    
                    if not completion.choices[0].message or not hasattr(completion.choices[0].message, 'content'):
                        logger.error("Invalid completion response: missing content")
                        return jsonify({"success": False, "error": "Invalid response from LLM API: missing content"}), 500
                    
                    llm_reply = completion.choices[0].message.content.strip()
                    
                    # Ensure the response is not too long
                    words = llm_reply.split()
                    if len(words) > 55:
                        llm_reply = ' '.join(words[:50]) + '...'
                    
                    logger.info(f"Generated response: {llm_reply[:50]}...")
                    
                    # Send response back to LLM1
                    try:
                        response = requests.post(
                            'http://localhost:5000/api/llm1/llm',
                            json={
                                "text": llm_reply,
                                "topic": topic,
                                "is_user_message": False,
                                "is_initial_message": False,
                                "from_llm2": True,
                                "conversation_history": conversation_history + [{"role": "assistant", "content": llm_reply}]
                            }
                        )
                        logger.info(f"LLM1 response status: {response.status_code}")
                    except Exception as e:
                        logger.error(f"Error sending response to LLM1: {e}")
                    
                    # Return the response
                    return jsonify({
                        "success": True, 
                        "response": llm_reply,
                        "model_used": "llama-3.2-3b"
                    })
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    logger.warning(f"Attempt {retry_count} failed: {str(e)}")
                    if retry_count < max_retries:
                        time.sleep(1)  # Wait before retrying
                    continue

            # If we get here, all retries failed
            logger.error(f"All {max_retries} attempts failed. Last error: {str(last_error)}")
            is_ai_speaking = False
            current_speaker = None
            return jsonify({"success": False, "error": f"Failed to generate response after {max_retries} attempts"}), 500
            
        except Exception as e:
            logger.error(f"Error generating content with Llama: {e}")
            is_ai_speaking = False
            current_speaker = None
            return jsonify({"success": False, "error": f"Failed to generate response: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Error in LLM processing: {e}")
        is_ai_speaking = False
        current_speaker = None
        return jsonify({"success": False, "error": f"Failed to get response from LLM: {str(e)}"}), 500

@llm_bp.route('/api/llm2/user_finished', methods=['POST'])
def user_finished_speaking():
    """Handle when user finishes speaking"""
    global is_user_speaking, last_message, last_topic, is_ai_speaking
    try:
        is_user_speaking = False
        is_ai_speaking = False
        if last_message and last_topic:
            # Continue the conversation with the last user message
            return get_llm_response()
        return jsonify({"success": True, "response": "Conversation resumed."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@llm_bp.route('/api/llm2/tts', methods=['POST'])
def text_to_speech():
    """Converts text to speech using gTTS with a deep male voice."""
    try:
        data = request.get_json()
        text = data.get("text")

        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400

        # Add some light formatting to make the speech more expressive
        formatted_text = text.replace("!", "! ").replace("?", "? ")
        
        # Using British English for a deeper male voice
        tts = gTTS(
            text=formatted_text, 
            lang='en',
            tld='co.uk',  # British English - deeper male voice
            slow=False     # Normal speed
        )
        
        # Save the audio to a byte stream
        audio_stream = io.BytesIO()
        tts.write_to_fp(audio_stream)
        audio_stream.seek(0)
        
        # Log success
        logger.info(f"Successfully generated speech for text: {text[:30]}...")
        
        return send_file(audio_stream, mimetype="audio/mp3")
    
    except Exception as e:
        logger.error(f"Error in gTTS conversion: {e}")
        return jsonify({"success": False, "error": f"Text-to-speech conversion failed: {str(e)}"}), 500
