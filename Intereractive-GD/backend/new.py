from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

# Set your OpenAI API key
openai.api_key = "sk-ijklmnopabcd5678ijklmnopabcd5678ijklmnop"

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Call OpenAI's API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use your preferred model
            messages=[{"role": "user", "content": user_message}]
        )

        ai_response = response['choices'][0]['message']['content']
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
