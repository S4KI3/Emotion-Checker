# Streamlit Emotion Detector 🎭

A lightweight **Streamlit** web app that turns raw text into one of six emotions:

| ID | Emotion |
|----|---------|
| 0  | sadness |
| 1  | anger   |
| 2  | love    |
| 3  | surprise|
| 4  | fear    |
| 5  | joy     |

The UI loads two pickled artifacts—`vectorizer.pkl` and `model.pkl`—to deliver instant, probability-based predictions for single sentences or full CSV files.

---

## ✨ Features
- Real-time single-sentence classification
- One-click batch inference on CSVs (`text` column required)
- Probabilities for all six classes
- Clean text preprocessing and fast cached model loading
- Works entirely offline after setup

## 📂 Project structure
├── app.py # Streamlit UI
├── model.pkl # Trained classifier (e.g., LogisticRegression)
├── vectorizer.pkl # Fitted TF-IDF vectorizer
├── requirements.txt # Exact package versions
└── README.md

## ⚡ Quick start
git clone https://github.com/<your-user>/streamlit-emotion-detector.git
cd streamlit-emotion-detector
python -m venv .venv && source .venv/bin/activate # optional, Linux/macOS
pip install -r requirements.txt
streamlit run app.py

Open the browser link printed in the terminal—usually `http://localhost:8501`.

## 🧪 Test examples
| Expected emotion | Example sentence |
|------------------|------------------|
| sadness          | “The world feels colorless since my best friend moved away.” |
| anger            | “How dare you take credit for my work without asking!” |
| love             | “Every moment with you makes me believe in forever.” |
| surprise         | “Wait—you organised a party for me on a Wednesday?” |
| fear             | “My hands are shaking; I think someone is following me.” |
| joy              | “I just got accepted into my dream university!” |

Paste any of these into the text box to verify the mapping.

