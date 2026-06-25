# Multi-Label Toxic Comment Classification - NLP

## Overview

This project is an end-to-end Natural Language Processing (NLP) application that performs **multi-label text classification** on online comments using Machine Learning. Unlike traditional classification problems where each input belongs to only one category, this model predicts multiple toxicity-related labels for a single comment simultaneously.

The system is built using **TF-IDF Vectorization**, **One-vs-Rest Classification**, and **Logistic Regression**, and is deployed through a **FastAPI backend**. It allows users to submit any comment and receive predictions along with confidence scores for each toxicity category.

The project demonstrates the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis, feature engineering, model training, hyperparameter tuning, evaluation, model serialization, API development, and deployment.

---

# Problem Statement

Online communities receive millions of user-generated comments every day. Manual moderation is time-consuming, inconsistent, and impossible at large scale. Harmful comments such as threats, insults, obscene language, or hate speech can negatively impact users and community health.

The objective of this project is to build an automated moderation system capable of detecting multiple forms of toxicity from a single comment.

Instead of answering a simple Yes or No, the model predicts whether a comment belongs to one or more of the following categories:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

Since a comment can belong to multiple categories simultaneously, this is a **Multi-Label Classification** problem.

---

# Dataset

Dataset Used:

**Jigsaw Toxic Comment Classification Challenge**

The dataset contains comments collected from Wikipedia discussions and includes six binary target labels.

### Features

* id
* comment_text

### Target Labels

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

Each label is represented as:

* 1 → Comment belongs to that category
* 0 → Comment does not belong to that category

Example:

| Comment         | Toxic | Obscene | Threat |
| --------------- | ----- | ------- | ------ |
| You are stupid  | 1     | 0       | 0      |
| I will kill you | 1     | 0       | 1      |

---

# Project Pipeline

```
Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Label Distribution Analysis
      │
      ▼
TF-IDF Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
One-vs-Rest Classification
      │
      ▼
Logistic Regression
      │
      ▼
Hyperparameter Tuning (GridSearchCV)
      │
      ▼
Evaluation
      │
      ▼
Model Serialization (Joblib)
      │
      ▼
FastAPI Deployment
```

---

# Exploratory Data Analysis

The dataset was analyzed before training to understand its characteristics.

Performed analysis included:

* Dataset dimensions
* Missing value inspection
* Label frequency analysis
* Multi-label distribution
* Class imbalance analysis
* Random comment inspection

One major observation was severe class imbalance.

Example:

* Toxic comments were common.
* Threat and Identity Hate comments were comparatively rare.

This imbalance significantly affected recall during the initial baseline model.

---

# Text Preprocessing

The following preprocessing steps were applied:

* English stop-word removal
* TF-IDF Vectorization
* Maximum vocabulary selection
* Unigram and Bigram extraction
* Rare word filtering

Final TF-IDF Configuration:

* max_features = 20000
* stop_words = "english"
* ngram_range = (1,2)
* min_df = 2

This configuration enabled the model to learn not only important individual words but also meaningful phrases such as:

* kill you
* stupid idiot
* hate you

which improved contextual understanding.

---

# Feature Engineering

Instead of directly training on raw text, comments were transformed into numerical vectors using TF-IDF.

The vectorizer assigns higher importance to informative words while reducing the influence of common words.

This allows classical Machine Learning models to process textual information efficiently.

---

# Model

The project uses:

### One-vs-Rest Classifier

Since a single comment can belong to multiple categories simultaneously, One-vs-Rest trains one independent binary classifier for every toxicity label.

Internally, six Logistic Regression models are trained.

Example:

* Toxic vs Not Toxic
* Threat vs Not Threat
* Obscene vs Not Obscene

Each classifier independently predicts the probability of its respective label.

---

# Base Algorithm

The underlying algorithm is:

**Logistic Regression**

Reasons for selection:

* Fast training
* High interpretability
* Excellent performance on sparse TF-IDF features
* Strong baseline for NLP classification

---

# Hyperparameter Tuning

Model performance was improved using GridSearchCV.

Parameters searched included:

* Regularization strength (C)
* Class Weight

GridSearchCV automatically evaluated different parameter combinations using cross-validation and selected the best-performing model based on Weighted F1 Score.

---

# Evaluation Metrics

The following metrics were used:

* Precision
* Recall
* F1 Score
* Support
* Classification Report
* Multi-label Confusion Matrix

Accuracy alone was intentionally avoided because of severe class imbalance.

Instead, F1 Score and Recall were used to obtain a better understanding of model performance.

---

# Model Serialization

The trained model and TF-IDF vectorizer were saved using Joblib.

Files generated:

```
tox_model.pkl
tfidf.pkl
```

These files are loaded directly by the FastAPI backend, eliminating the need for retraining during inference.

---

# FastAPI Backend

The backend exposes REST APIs capable of receiving user comments and returning toxicity predictions.

Main endpoint:

```
POST /predict
```

Input:

```json
{
    "text":"I will kill you"
}
```

Output:

```json
{
    "comment":"I will kill you",
    "results":{
        "toxic":{
            "prediction":true,
            "confidence":0.91
        },
        "threat":{
            "prediction":true,
            "confidence":0.96
        }
    }
}
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* One-vs-Rest Classifier
* GridSearchCV
* Joblib
* FastAPI
* Uvicorn

---

# Project Structure

```
toxic-comment-classifier/
│
├── app.py
├── tox_model.pkl
├── tfidf.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── notebooks/
    └── toxic_comment_classifier.ipynb
```

---

# Key Learnings

This project significantly improved my understanding of Natural Language Processing and multi-label machine learning systems.

Major concepts learned include:

* Difference between Multi-Class and Multi-Label Classification
* Text preprocessing for Machine Learning
* TF-IDF feature extraction
* Importance of stop-word removal
* Unigrams and Bigrams (N-Grams)
* Sparse matrix representation
* Logistic Regression for NLP
* One-vs-Rest Classification
* Multi-label probability prediction
* Precision, Recall and F1-score interpretation
* Class imbalance challenges
* Confusion Matrix analysis
* Hyperparameter tuning using GridSearchCV
* Model serialization with Joblib
* REST API development using FastAPI
* Building production-ready Machine Learning inference systems

---

# Future Improvements

Potential improvements include:

* Transformer-based models (BERT, RoBERTa)
* Threshold tuning for individual labels
* Explainable AI using SHAP/LIME
* Docker containerization
* CI/CD deployment pipeline
* User authentication
* Batch prediction API
* Real-time moderation dashboard

---

# Conclusion

This project demonstrates a complete production-oriented NLP workflow, starting from raw textual data and ending with a deployable REST API. It combines Natural Language Processing, Machine Learning, feature engineering, model optimization, and backend deployment into a single application.

Beyond building an accurate classifier, the project emphasizes understanding the complete machine learning pipeline, proper evaluation of imbalanced datasets, deployment best practices, and designing scalable inference systems suitable for real-world content moderation applications.
