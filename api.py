import logging
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from text_classifier import WebTextClassifier

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_URL = "https://drive.google.com/uc?export=download&id=1eGKe3C7wjJJC-qz15M16gOxHgKdpP5wI"
MODEL_PATH = "web_classifier.pth"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully!")

# Initialize classifier
classifier = None

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    url = data['url']
    title = data['title']
    allowed_categories = data['categories']
    text_to_classify = classifier.preprocess_url(url) + " " + title
    logger.debug(f"Classifying: {text_to_classify}")
    predicted_category = classifier.classify(text_to_classify, allowed_categories)
    logger.debug(f"Predicted category: {predicted_category}")
    return jsonify({'category': predicted_category})

@app.route('/debug', methods=['POST'])
def debug():
    data = request.json
    url = data['url']
    title = data['title']
    allowed_categories = data['categories']
    text_to_classify = classifier.preprocess_url(url) + " " + title
    predicted_category = classifier.classify(text_to_classify, allowed_categories)
    logger.debug(f"Debug - Input: {text_to_classify}, Predicted: {predicted_category}")
    return jsonify({
        'input_text': text_to_classify,
        'predicted_category': predicted_category
    })

if __name__ == '__main__':
    # Download model and initialize classifier
    download_model()
    classifier = WebTextClassifier()
    classifier.load_model(MODEL_PATH)
    app.run(debug=True)
