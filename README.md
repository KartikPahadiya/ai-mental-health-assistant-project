# 🧠 AI Mental Health Assistant

An end-to-end **AI-powered mental health assistant** built using **Flask, NLP, Speech Processing, and Machine Learning**.  
The system analyzes **text and audio inputs** to detect emotions, mental states, and potential suicide risk, and responds with supportive, therapist-style feedback.

> ⚠️ This project is intended for **educational and research purposes only** and is **not a replacement for professional mental health care**.

---

## 🚀 Features

- 🧑‍⚕️ **Therapist-style chatbot** using NLP & LLM-based responses  
- 📝 **Text-based emotion detection**
- 🎙️ **Audio emotion recognition** (RAVDESS, TESS, SAVEE models)
- ☠️ **Suicide risk detection** using classical ML (TF-IDF + LinearSVC)
- 🤖 **BERT-based emotion classification** (single-label & multi-label)
- ⚡ **Model warm-up for low-latency inference**
- 🌐 Flask-based web interface

---

## 🧩 Tech Stack

**Backend**
- Python 
- Flask

**Machine Learning / NLP**
- scikit-learn
- PyTorch
- TensorFlow
- HuggingFace Transformers
- spaCy

**Audio Processing**
- librosa
- NumPy
- SciPy

**Model Storage**
- Git LFS (for `.pkl`, `.joblib`, `.safetensors` files)

---

## 📁 Project Structure

ai-mental-health-assistant-project/
│
├── app.py # Flask application entry point

├── therapist_chat.py # Therapist-style response logic

├── requirements.txt

│
├── Suicide_Detection/ # Suicide risk prediction models

├── all_voices/ # Combined voice emotion models

├── ravdess_model/

├── tess_model/

├── savee_model/

├── bert_emotion_oversampled_model/

├── model/ # BERT model weights

│
├── static/ # CSS & JS files

├── templates/ # HTML templates

│
├── .gitignore

├── .gitattributes # Git LFS tracking---

## ⚙️ Setup Instructions (IMPORTANT)

### 1️⃣ Clone the repository (DO NOT download ZIP)

``bash{
git clone https://github.com/<your-username>/ai-mental-health-assistant-project.git
cd ai-mental-health-assistant-project
}
❗ ZIP downloads do not fetch Git LFS model files and will break the project.

2️⃣ Create and activate virtual environment

Windows (PowerShell)

python -m venv venv
venv\Scripts\Activate


Linux / macOS

python3 -m venv venv
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt


⚠️ This project uses large ML libraries (TensorFlow, PyTorch).
Make sure you have sufficient disk space (≈ 8–10 GB).

4️⃣ Set environment variables

Create a .env file in the root directory:

HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here


The .env file is intentionally not committed for security reasons.

5️⃣ Run the application
python app.py


Open your browser at:

http://127.0.0.1:5000

⚠️ Known Warnings (Expected)

You may see warnings like:

InconsistentVersionWarning: Trying to unpickle estimator from version 1.4.2


✔️ This is expected because models were trained with an earlier scikit-learn version.
✔️ The application still runs correctly.
