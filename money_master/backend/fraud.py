from flask import Flask, render_template, request
import google.generativeai as genai
import os

app = Flask(__name__)

gem_key = 'AIzaSyDy7yDCH0iaegYfLMPveYshuEcdkqhzZRI'
genai.configure(api_key=gem_key)
model = genai.GenerativeModel("gemini-1.5-flash")

temp1 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
temp2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

@app.route('/')
def home():
    fraud = model.generate_content("Real Madrid")
    return render_template('index.html', fraud = fraud.text, messagehead = temp1, messagebody=temp2, test=[1, 2, 3])
 

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']

    predef = (
        "Check if there is a possibility of financial crime. If yes, return the crime and why. If no, return \"All OK!\".",
        "ONLY return the crime and why OR the All OK! statement no intro. all text should be uniform(no boldening)",
        "IF a crime is detected, add a line \"for more information\" with a link leads to a website with more data"
    );

    x = model.generate_content(f"{predef} {user_input}")
    print(x.text)

    return render_template('index.html', x = x.text, messagehead = temp1, messagebody=temp2, test=[1, 2, 3])

if __name__ == "__main__":
    app.run(debug=True)