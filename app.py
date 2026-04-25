import streamlit as st
import pickle
import re
import os
import numpy as np

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Fake News Detector", page_icon="🧠")

# ==============================
# LOAD MODEL
# ==============================
base_path = os.path.dirname(__file__)

model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(base_path, "vectorizer.pkl"), "rb"))

# ==============================
# CLEAN FUNCTION
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# ==============================
# UI
# ==============================
st.title("🧠 AI-Based Misinformation Detection System")
st.markdown("### Detect fake news using Machine Learning + Explainability")

st.write("Enter a news headline or article below:")

user_input = st.text_area("📰 News Text", height=150)

# ==============================
# PREDICTION
# ==============================
if st.button("🔍 Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text")
    
    elif len(user_input.split()) < 3:
        st.warning("Please enter a more meaningful sentence")
    
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        result = model.predict(vec)
        proba = model.predict_proba(vec)

        st.markdown("## 📊 Result")

        if result[0] == 1:
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        # ==============================
        # CONFIDENCE
        # ==============================
        confidence = np.max(proba) * 100
        st.write(f"### 🔢 Confidence: {confidence:.2f}%")
        st.progress(float(confidence) / 100)

        # ==============================
        # EXPLAINABILITY (REAL INPUT WORDS)
        # ==============================
        vec_array = vec.toarray()[0]
        indices = np.argsort(vec_array)[-10:]

        feature_names = vectorizer.get_feature_names_out()
        important_words = [feature_names[i] for i in indices if vec_array[i] > 0]

        st.write("### 🧠 Words influencing this prediction:")
        if important_words:
            st.write(", ".join(important_words))
        else:
            st.write("No strong keywords detected")

        # ==============================
        # SYSTEM MESSAGE (DRDO TOUCH)
        # ==============================
        if confidence < 60:
            st.warning("⚠️ Low confidence prediction — model may be uncertain.")