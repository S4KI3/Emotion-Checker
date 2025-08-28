# app.py
# ──────────────────────────────────────────────────────────
# Streamlit UI for a **Text → Emotion** classifier
# Label space:
#   0 = sadness • 1 = anger • 2 = love • 3 = surprise • 4 = fear • 5 = joy
# Requirements: streamlit pandas numpy scikit-learn joblib
# Run with:  streamlit run app.py
# ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

# 0️⃣  Load artefacts (as requested) ───────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    Load the vectorizer and the downstream classifier saved
    separately as `vectorizer.pkl` and `model.pkl`.
    """
    try:
        model = joblib.load("model.pkl")           # classifier (e.g. LogisticRegression)
        vectorizer = joblib.load("vectorizer.pkl") # TF-IDF / CountVectorizer
    except FileNotFoundError as err:
        st.error(f"Missing file: {err.filename}. "
                 "Place model.pkl and vectorizer.pkl in the same folder as app.py.")
        st.stop()
    return model, vectorizer

model, vectorizer = load_assets()

# 1️⃣  Constants ───────────────────────────────────────────
LABEL_MAP = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

# 2️⃣  Text utilities ─────────────────────────────────────
def clean_text(txt: str) -> str:
    txt = txt.lower()
    txt = re.sub(r"http\S+|www\S+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def predict(texts):
    """
    texts : List[str]
    returns: tuple(List[str] labels, np.ndarray probs)
    """
    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    ids = model.predict(X)                         # e.g. [0, 5, 3]
    try:
        probs = model.predict_proba(X)             # shape = (n_samples, 6)
    except AttributeError:
        # Model without probability support → make a pseudo one-hot matrix
        probs = np.eye(len(LABEL_MAP))[ids]
    labels = [LABEL_MAP[int(i)] for i in ids]
    return labels, probs

# 3️⃣  Streamlit layout ───────────────────────────────────
st.set_page_config(page_title="Emotion Detector", page_icon="🎭", layout="centered")
st.title("🎭 Real-time Emotion Detector")

# Sidebar: Batch option
uploaded_csv = st.sidebar.file_uploader("📑 Batch CSV (must have a 'text' column)", type=["csv"])
st.sidebar.markdown("**Label Index → Emotion**  \n" + " • ".join(f"{k}:{v}" for k,v in LABEL_MAP.items()))

# Tabs: single vs. batch
tab_single, tab_batch = st.tabs(["🚀 Single Text", "📂 Batch CSV"])

# 3a. Single-text prediction
with tab_single:
    user_text = st.text_area("Enter text:", height=120,
                             placeholder="I just aced my exam and I'm feeling fantastic!")
    if st.button("Predict"):
        if not user_text.strip():
            st.warning("Please type something first.")
        else:
            label, prob = predict([user_text])
            st.success(f"**Emotion:** {label[0].title()}")
            prob_df = pd.DataFrame(
                {"Emotion": [LABEL_MAP[i] for i in range(len(LABEL_MAP))],
                 "Probability": np.round(prob[0], 3)}
            ).sort_values("Probability", ascending=False)
            st.table(prob_df)

# 3b. Batch prediction
with tab_batch:
    if uploaded_csv is None:
        st.info("Upload a CSV to enable batch mode.")
    else:
        df = pd.read_csv(uploaded_csv)
        if "text" not in df.columns:
            st.error("CSV must contain a column called 'text'.")
        else:
            if st.button("Run batch prediction"):
                labels, probs = predict(df["text"].tolist())
                out = df.copy()
                out["predicted_emotion"] = labels
                for i in range(len(LABEL_MAP)):
                    out[f"p_{LABEL_MAP[i]}"] = np.round(probs[:, i], 4)
                st.dataframe(out.head(25))
                st.download_button(
                    "⬇️ Download full results",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="emotion_predictions.csv",
                    mime="text/csv"
                )

# 4️⃣  Footer ─────────────────────────────────────────────
st.caption("© 2025 </>Sakib • Streamlit Emotion Classifier Demo")
