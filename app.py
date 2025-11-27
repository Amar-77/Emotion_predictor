from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = Flask(__name__)

# --- Model Loading ---
# Load the tokenizer and model from the local directory
MODEL_DIR = "./distilbert_finetuned_emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# We will manually define the emotion labels, as the model's config might not have the correct names.
# This mapping corresponds to the standard 'emotion' dataset.
emotion_map = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}

# Create a text-classification pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None) # Use top_k=None to get all scores

# --- Routes ---
@app.route('/')
def home():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives text and returns the predicted emotion and confidence scores."""
    try:
        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # The pipeline returns a list of lists with all scores
        # e.g., [[{'label': 'LABEL_4', 'score': 0.9...}, {'label': 'LABEL_0', 'score': 0.01...}]]
        raw_outputs = pipe(text)

        # Convert the raw outputs to a more friendly format using our manual map
        # e.g., {'fear': 0.9, 'sadness': 0.01, ...}
        confidence_scores = {emotion_map.get(int(pred['label'].split('_')[1]), pred['label']): pred['score'] for pred in raw_outputs[0]}

        # Find the label with the highest score to be our main prediction
        top_prediction_label = max(confidence_scores, key=confidence_scores.get)

        # Return the prediction and all confidence scores
        return jsonify({
            'prediction': top_prediction_label,
            'confidence': confidence_scores
        })

    except Exception as e:
        # Log the error for debugging
        print(f"Error occurred: {e}")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

