import pandas as pd
import numpy as np
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# ==============================
# 1. LOAD DATA
# ==============================
print("Loading dataset...")

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)

# ==============================
# 2. ADD LABELS
# ==============================
fake["label"] = 0
true["label"] = 1

# ==============================
# 3. COMBINE & SHUFFLE
# ==============================
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Combined dataset shape:", df.shape)

# ==============================
# 4. CLEAN TEXT
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df["text"] = df["text"].apply(clean_text)

# ==============================
# 5. FEATURES & LABELS
# ==============================
X = df["text"]
y = df["label"]

# ==============================
# 6. TRAIN TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 7. TF-IDF
# ==============================
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 8. MULTIPLE MODELS
# ==============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

best_model = None
best_accuracy = 0

print("\n===== MODEL COMPARISON =====")

for name, m in models.items():
    print(f"\n{name}")
    
    m.fit(X_train_vec, y_train)
    pred = m.predict(X_test_vec)
    
    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = m

# Use best model
model = best_model

print("\nBest Model Selected:", model)

# ==============================
# 9. FINAL EVALUATION
# ==============================
pred = model.predict(X_test_vec)

print("\n===== FINAL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, pred))

# ==============================
# 10. SAVE MODEL
# ==============================
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved!")

# ==============================
# 11. USER INPUT
# ==============================
print("\n==============================")
print("Enter News Text to Test:")
print("==============================")

user_input = input()

user_clean = clean_text(user_input)
user_vec = vectorizer.transform([user_clean])

result = model.predict(user_vec)

print("\n===== RESULT =====")

if result[0] == 1:
    print("🟢 REAL NEWS")
else:
    print("🔴 FAKE NEWS")

print("\nDone ✅")