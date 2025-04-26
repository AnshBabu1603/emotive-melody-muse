from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from deepface import DeepFace
import os
import psutil  # To track memory usage
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://emotive-melody-muse-611b98d7.vercel.app"]}})

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image file provided'}), 400

    # Read image as bytes
    img_bytes = image_file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Resize image to reduce memory usage
    img = cv2.resize(img, (640, 480))  # Resize to a smaller size

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get additional inputs
    language = request.form.get('language', 'English')
    singer = request.form.get('singer', '')

    # Log memory usage before processing
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / 1024 / 1024  # Memory in MB
    print(f"Memory before processing: {memory_before} MB")

    try:
        # Emotion analysis with DeepFace
        result = DeepFace.analyze(
            img_rgb,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='opencv'  # Use OpenCV as the detector backend (lighter on memory)
        )

        if isinstance(result, list):
            result = result[0]

        emotion = result['dominant_emotion']

        # If emotion is 'neutral', override it with 'sad'
        if emotion == 'neutral':
            emotion = 'sad'

        # Build YouTube URL
        query = f"{language} {emotion} songs"
        if singer:
            query += f" {singer}"
        query = query.replace(' ', '+')

        youtube_url = f"https://www.youtube.com/results?search_query={query}"

        # Log memory usage after processing
        memory_after = process.memory_info().rss / 1024 / 1024  # Memory in MB
        print(f"Memory after processing: {memory_after} MB")

        return jsonify({
            'emotion': emotion,
            'language': language,
            'singer': singer,
            'youtube_url': youtube_url
        })

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
