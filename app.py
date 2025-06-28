from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Summarization Model
# -------------------------------
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("kyanmahajan/summarisee-finetuned-amazon-en-es")
summarizer_tokenizer = AutoTokenizer.from_pretrained("kyanmahajan/summarisee-finetuned-amazon-en-es")
summarizer_model.eval()

# -------------------------------
# Load Rating Predictor Model
# -------------------------------
rating_model = AutoModelForSequenceClassification.from_pretrained("kyanmahajan/rating-predictor-v1")
rating_tokenizer = AutoTokenizer.from_pretrained("kyanmahajan/rating-predictor-v1")
rating_model.eval()

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return render_template('index.html')  # Put index.html in /templates folder

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    # length = data.get('len', 20)
    length = len(text.split())
    print(length)

    if not text.strip():
        return jsonify({"summary": "No input text provided."})

    inputs = summarizer_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        summary_ids = summarizer_model.generate(inputs["input_ids"], max_length=length)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return jsonify({"summary": summary})

@app.route('/rating', methods=['POST'])
def predict_rating():
    data = request.get_json()
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"rating": "Invalid input"})

    inputs = rating_tokenizer(review, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = rating_model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    rating = predicted_class  # Assuming class 0 = rating 1

    return jsonify({
        "rating": rating,
        "probabilities": probs.squeeze().tolist()
    })

# -------------------------------
# Run the App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
