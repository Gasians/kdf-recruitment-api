# 🧠 NLP Sentiment Analysis — Phase 4 Group 2 Project

This project focuses on **Natural Language Processing (NLP)** to perform sentiment analysis on Twitter data. We used various preprocessing techniques and machine learning models to classify tweets as **positive**, **neutral**, or **negative**.

---

## 🔍 Problem Statement

Given a dataset of tweets labeled by sentiment, our task was to build a machine learning model that accurately predicts the sentiment of new, unseen tweets.

---

## 💼 Project Pipeline

1. **Text Preprocessing**
   - Lowercasing text
   - Removing URLs, hashtags, digits, and punctuation
   - Dropping missing data

2. **Tokenization**
   - Splitting text into words using `nltk.word_tokenize`

3. **Stopword Removal**
   - Removing common words (e.g., "the", "is", "in") using NLTK

4. **Stemming & Lemmatization**
   - Using `PorterStemmer` and `WordNetLemmatizer` to normalize words

5. **Vectorization**
   - Converting words to numeric features using `CountVectorizer`

6. **Modeling**
   - Trained and evaluated:
     - Logistic Regression
     - Random Forest
     - Multinomial Naive Bayes
     - Support Vector Machine (SVM)
   - Used `GridSearchCV` and `RandomizedSearchCV` to tune hyperparameters

7. **Evaluation**
   - Achieved 89% accuracy with Logistic Regression
   - Saved final model with `joblib`

---

## ⚙️ Tech Stack

- **Python**
- **Pandas**, **Scikit-learn**, **NLTK**
- **Jupyter Notebook**

---

## 👨‍💻 Group Contribution

> Handled the following NLP pipeline tasks:
- Text preprocessing
- Tokenization
- Stopword removal
- Stemming & Lemmatization
- Vectorization
- Model training & evaluation (Logistic Regression, Naive Bayes, Random Forest, SVM)

---

## 📁 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Godfrey-249/Phase-4-Group-2--NLP-Project.git
cd Phase-4-Group-2--NLP-Project

