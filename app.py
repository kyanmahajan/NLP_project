from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from flask_cors import CORS
import torch

# Model Names
MODEL_NAME_SUMMARY = "kyanmahajan/summarisee-finetuned-amazon-en-es"
MODEL_NAME_RATING = "kyanmahajan/rating-predictor-v1"

# Load models and tokenizers
tokenizer_summary = AutoTokenizer.from_pretrained(MODEL_NAME_SUMMARY)
model_summary = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_SUMMARY)
model_summary.eval()

tokenizer_rating = AutoTokenizer.from_pretrained(MODEL_NAME_RATING)
model_rating = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_RATING)
model_rating.eval()

# Flask App
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    length = data.get("len", 20)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided."}), 400

    inputs = tokenizer_summary(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    summary_ids = model_summary.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=length,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer_summary.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({"summary": summary})

@app.route("/rating", methods=["POST"])
def predict_rating():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No input text provided."}), 400

    inputs = tokenizer_rating(
        text,
        max_length=300,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    logits = model_rating(**inputs)
    output = torch.argmax(logits.logits, dim=-1)
    return jsonify({"rating": int(output.item())})

if __name__ == "__main__":
    app.run(debug=True)
 