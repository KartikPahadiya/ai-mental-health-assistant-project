#  AI Mental Health Assistant

An end-to-end **AI-powered mental health assistant** built using **Flask, NLP, Speech Processing, and Machine Learning**.  
The system analyzes **text and audio inputs** to detect emotions, mental states, and potential suicide risk, and responds with supportive, therapist-style feedback.

>  This project is intended for **educational and research purposes only** and is **not a replacement for professional mental health care**.

---

## 🚀 Live Demo (Deployed)

👉 **Try the app here:**  
🔗 **https://kartikpahadiya-mental-health-app.hf.space**

*(Deployed on Hugging Face Spaces — runs independently of local machine)*

---
##  Features

-  **Therapist-style chatbot** using NLP & LLM-based responses  
-  **Text-based emotion detection**
-  **Audio emotion recognition** (RAVDESS, TESS, SAVEE models)
-  **Suicide risk detection** using classical ML (TF-IDF + Linear SVC)
-  **BERT-based emotion classification** (single-label & multi-label)
-  **Model warm-up for low-latency inference**
-  Flask-based web interface

---

##  Setup Instructions 

### 1️. Clone the repository 

```bash
git clone https://github.com/KartikPahadiya/ai-mental-health-assistant-project.git
cd ai-mental-health-assistant-project
```
[x] ZIP downloads do not fetch Git LFS model files and will break the project.

### 2️. Create and activate virtual environment

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate
```


**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️. Install dependencies
```bash
pip install -r requirements.txt
```

- This project uses large ML libraries (TensorFlow, PyTorch).
- Make sure you have sufficient disk space (≈ 8–10 GB).

[x] run this command in cmd
```bash
python -m spacy download en_core_web_sm
```
### 4️. Set environment variables

Create a `.env` file in the root directory:

HUGGINGFACEHUB_API_TOKEN=<your_huggingface_token_here>


The `.env` file is intentionally not committed for security reasons.

### 5️. Run the application
```bash
python app.py
```


Open your browser at:

```tex
http://127.0.0.1:5000
```

`Known Warnings (Expected)`

You may see warnings like:

`InconsistentVersionWarning: Trying to unpickle estimator from version 1.4.2`

[x] This is expected because models were trained with an earlier scikit-learn version.
[x] The application still runs correctly.
