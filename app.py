from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime
import requests
import json

app = Flask(__name__)

# Folder paths
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
LOG_FILE = 'saved_results.json'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Groq API Setup
GROQ_API_URL = 'https://api.groq.com/v1/your-endpoint'
API_KEY = 'vgsk_muHGenrZ6nsTDHedx86XWGdyb3FY5VMrbsmdhRwbXUzr8StgBdjs'  # Add your Groq API key here

# Home page - upload a file
@app.route('/')
def index():
    return render_template('index.html')

# About Us page
@app.route('/about')
def about():
    return render_template('about.html')

# Feedback page
@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

# Handle file upload and send it to Groq API
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded!", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected!", 400

    # Generate a unique filename using timestamp
    filename = generate_unique_filename(file.filename)

    # Save the file locally
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Send the PPT file to Groq API for analysis
    response = send_to_groq(file_path)

    if response.status_code == 200:
        analysis = response.json()  # Assuming API returns JSON analysis
        save_result(filename, analysis)
        return render_template('chatbot.html', analysis=analysis)
    else:
        return "Error in analysis!", 500

# Generate a unique filename with timestamp
def generate_unique_filename(filename):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

# Function to send file to Groq API
def send_to_groq(file_path):
    headers = {
        'Authorization': f'Bearer {API_KEY}'
    }
    files = {'file': open(file_path, 'rb')}
    response = requests.post(GROQ_API_URL, headers=headers, files=files)
    return response

# Save analysis result along with filename in a JSON log file
def save_result(filename, analysis):
    # Check if the log file exists
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    # Add new result entry
    results[filename] = analysis

    # Save back to the log file
    with open(LOG_FILE, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    app.run(debug=True)
