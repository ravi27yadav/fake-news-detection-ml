# 🧠 AI-Based Misinformation Detection System

## 📌 Overview
This project presents an AI-powered system for detecting fake news using Natural Language Processing (NLP) and Machine Learning techniques.

The system classifies news articles as **Real or Fake**, provides a **confidence score**, and highlights **key words influencing the prediction**.

It is designed with a focus on **information integrity**, and can be extended for **public information filtering and national security applications**.

---

## 🌐 Live Demo
🔗 **Access the deployed application:**  
👉 https://fake-news-detection-ml-gjuqjfkthbmttost6mlxjg.streamlit.app/

---

## 🖼️ <img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/89082bee-1142-4726-9cfc-a8277eb5eddb" />


### 🔹 Home Interface
![Home UI](images/home.png)

### 🔹 Prediction Result
![Prediction](images/result.png)

### 🔹 Confidence & Explanation
![Confidence](images/confidence.png)

> 📌 Add your screenshots inside an `images/` folder in the repository

---

## 🚀 Features
- 🧹 Text preprocessing and normalization  
- 🔤 TF-IDF based feature extraction  
- 🤖 Multi-model comparison:
  - Logistic Regression  
  - Naive Bayes  
  - Random Forest  
- 🎯 Automatic best model selection  
- 📊 Confidence score visualization  
- 🧠 Explainable AI (important words influencing prediction)  
- 🌐 Interactive web interface using Streamlit  

---

## 📊 Model Performance
- Accuracy: ~98%  
- Evaluated using:
  - Classification Report  
  - Confusion Matrix  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  

---

## ⚙️ System Workflow
1. User inputs news text  
2. Text is cleaned and preprocessed  
3. TF-IDF converts text into numerical features  
4. Machine learning model predicts Real or Fake  
5. System outputs:
   - Prediction result  
   - Confidence score  
   - Important words influencing decision  

---

## 📁 Project Structure
fake-news-detection-ml/
│
├── app.py
├── fakenews.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── images/


---

## ⚠️ Limitations
- Model depends on training dataset distribution  
- Limited generalization to unseen domains  
- Does not verify factual correctness, only detects patterns  

---

## 🎯 Applications
- Fake news detection on social media  
- Misinformation filtering systems  
- Public information monitoring  
- Potential use in **national security and intelligence systems**  

---

## 🔮 Future Enhancements
- Integration with real-time news APIs  
- Deep learning models (BERT, LSTM)  
- Advanced explainability (SHAP, LIME)  
- Multi-language support  

---

## 👨‍💻 Author
**Ravi Yadav**

---

## ⭐ Acknowledgement
If you found this project useful, consider giving it a ⭐ on GitHub.
