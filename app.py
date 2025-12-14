import os
import torch
import joblib
import numpy as np
import re
import string
import librosa
import tempfile
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

from therapist_chat import therapist_reply
from Suicide_Detection.predict import predict_suicide
from transformers import BertForSequenceClassification, BertTokenizerFast, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# AUDIO FEATURE EXTRACTION
# ==========================
def extract_features_from_audio(y, sr):
    try:
        if np.sum(np.abs(y)) < 1e-6:
            return None
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr), axis=1)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr), axis=1)
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y), axis=1)
        features = np.concatenate([mfccs, spectral_centroid, spectral_bandwidth, spectral_rolloff, zero_crossing_rate])
        return features
    except:
        return None

def predict_emotion_audio(features, model, scaler=None, label_encoder=None):
    features_reshaped = features.reshape(1, -1)
    if scaler is not None:
        features_reshaped = scaler.transform(features_reshaped)
    prediction = model.predict(features_reshaped)
    if label_encoder is not None:
        try:
            if np.issubdtype(prediction.dtype, np.number):
                return label_encoder.inverse_transform(prediction)[0]
            else:
                return str(prediction[0])
        except:
            return str(prediction[0])
    else:
        return str(prediction[0])

# ==========================
# LOAD AUDIO MODELS
# ==========================
ravdess_model = joblib.load("ravdess_model/tuned_svm_rbf_model.pkl")
ravdess_scaler = joblib.load("ravdess_model/scaler.pkl")
tess_model = joblib.load("tess_model/svm_model.pkl")
savee_model = joblib.load("savee_model/svm_model.pkl")
savee_label = joblib.load("savee_model/label_encoder.pkl")
allvoices_model = joblib.load("all_voices/voting_ensemble_model.pkl")
allvoices_scaler = joblib.load("all_voices/scaler.pkl")
allvoices_label = joblib.load("all_voices/label_encoder.pkl")

# ==========================
# LOAD TEXT EMOTION MODELS
# ==========================
EMOTION_MODEL_PATH = "model"
emotion_model = BertForSequenceClassification.from_pretrained(EMOTION_MODEL_PATH, local_files_only=True)
emotion_tokenizer = BertTokenizerFast.from_pretrained(EMOTION_MODEL_PATH, local_files_only=True)
emotion_model.to(device)
emotion_model.eval()

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
    'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
    'gratitude', 'grief', 'joy', 'love', 'nervousness', 'neutral', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise'
]
emotion_label_encoder = LabelEncoder()
emotion_label_encoder.fit(emotion_labels)

OVERSAMPLED_MODEL_PATH = './bert_emotion_oversampled_model'
oversampled_tokenizer = AutoTokenizer.from_pretrained(OVERSAMPLED_MODEL_PATH)
oversampled_model = AutoModelForSequenceClassification.from_pretrained(OVERSAMPLED_MODEL_PATH)
oversampled_model.to(device)
oversampled_model.eval()

# ==========================
# FLASK ROUTES
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_audio", methods=["POST"])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"})
    
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=None)
        
        features = extract_features_from_audio(y, sr)
        if features is None:
            return jsonify({"error": "Audio too silent or invalid"})
        
        predictions = {
            "ravdess": predict_emotion_audio(features, ravdess_model, ravdess_scaler),
            "tess": predict_emotion_audio(features, tess_model),
            "savee": predict_emotion_audio(features, savee_model, label_encoder=savee_label),
            "ensemble": predict_emotion_audio(features, allvoices_model, allvoices_scaler, allvoices_label)
        }
        return jsonify(predictions)
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/predict_text", methods=["POST"])
def predict_text():
    text = request.json.get("text", "")
    clean_text = re.sub(f"[{re.escape(string.punctuation)}]", "", text.lower())
    clean_text = re.sub("\\s+", " ", clean_text).strip()

    # Single-label BERT
    inputs = emotion_tokenizer(clean_text, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    single_emotion_idx = torch.argmax(logits, dim=1).item()
    single_emotion = emotion_label_encoder.inverse_transform([single_emotion_idx])[0]

    # Multi-label oversampled
    multi_inputs = oversampled_tokenizer(clean_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    multi_inputs = {k: v.to(device) for k, v in multi_inputs.items()}
    with torch.no_grad():
        outputs = oversampled_model(**multi_inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
    multi_preds = [emotion_labels[i] for i, p in enumerate(probs) if p >= 0.3]

    return jsonify({"single": single_emotion, "multi": multi_preds})

@app.route("/predict_suicide", methods=["POST"])
def predict_suicide_route():
    text = request.json.get("text", "")
    pred = predict_suicide(text)
    return jsonify({"prediction": int(pred)})

@app.route("/therapist_chat", methods=["POST"])
def therapist_chat_route():
    user_msg = request.json.get("message", "")
    history = request.json.get("history", [])

    # get reply and new history
    reply, updated_history = therapist_reply(user_msg, history)

    # FIXED message converter with correct indentation
    def to_dict(msg):
        # Already correct dict → normalize it
        if isinstance(msg, dict) and "type" in msg and "content" in msg:
            t = msg["type"]
            if t in ["human", "human_message"]:
                return {"type": "human", "content": msg["content"]}
            else:
                return {"type": "ai", "content": msg["content"]}

        # Extract content safely
        content = getattr(msg, "content", str(msg))
        t = getattr(msg, "type", "")

        # Normalize LangChain message types
        if t in ["human", "human_message"]:
            return {"type": "human", "content": content}
        else:
            return {"type": "ai", "content": content}

    # Convert entire history
    clean_history = [to_dict(m) for m in updated_history]

    # AI reply normalized
    reply_obj = to_dict(reply)
    reply_obj["type"] = "ai"

    return jsonify({"reply": reply_obj, "history": clean_history})


print("🔊 Warming up audio pipeline...")

import numpy as _np
import librosa as _librosa

# generate dummy audio so librosa initializes its internals
_dummy = _np.random.randn(16000)
_mfcc = _librosa.feature.mfcc(y=_dummy, sr=16000, n_mfcc=13)

print("🔥 Audio warm-up complete.")

print("🔧 Warming up ML models...")

# --------------------------
# Warm-up AUDIO MODELS (SVMs)
# --------------------------
try:
    fake_features = np.random.randn(40)  # adjust if needed
    _ = ravdess_model.predict(fake_features.reshape(1, -1))
    _ = tess_model.predict(fake_features.reshape(1, -1))
    _ = savee_model.predict(fake_features.reshape(1, -1))
    _ = allvoices_model.predict(fake_features.reshape(1, -1))
    print("🎤 Audio ML models warmed up.")
except Exception as e:
    print("⚠️ Audio ML warm-up failed:", e)


# --------------------------
# Warm-up SINGLE-LABEL BERT
# --------------------------
try:
    dummy_text = "hello world"
    enc = emotion_tokenizer(dummy_text, return_tensors="pt", truncation=True, padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        _ = emotion_model(**enc)
    print("🔤 Single-label BERT warmed up.")
except Exception as e:
    print("⚠️ Single BERT warm-up failed:", e)


# --------------------------
# Warm-up MULTI-LABEL BERT
# --------------------------
try:
    dummy_text = "hello world"
    enc2 = oversampled_tokenizer(dummy_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    enc2 = {k: v.to(device) for k, v in enc2.items()}
    with torch.no_grad():
        _ = oversampled_model(**enc2)
    print("🔡 Multi-label BERT warmed up.")
except Exception as e:
    print("⚠️ Multi-label BERT warm-up failed:", e)


# --------------------------
# Warm-up SUICIDE MODEL
# --------------------------
try:
    _ = predict_suicide("I feel terrible but I'm testing the model")
    print("☠️ Suicide detection warmed up.")
except Exception as e:
    print("⚠️ Suicide warm-up failed:", e)


print("🔥 ALL MODELS SUCCESSFULLY WARMED ✔️")


if __name__ == "__main__":
    app.run(debug=True)
