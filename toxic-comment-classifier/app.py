from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load Saved Model and Vectorizer

model = joblib.load("tox_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Create FastAPI App

app = FastAPI(
    title="Toxic Comment Classifier",
    description="Multi-label NLP Classification API",
    version="1.0"
)

# Labels

labels = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# Request Body
class Comment(BaseModel):
    text: str

# Home Route

@app.get("/")
def home():
    return {
        "message": "Toxic Comment Classification API Running"
    }

# Prediction Route

@app.post("/predict")
def predict(comment: Comment):

    # Convert text to TF-IDF vector
    vector = tfidf.transform([comment.text])

    # Prediction
    prediction = model.predict(vector)[0]

    # Probability
    probability = model.predict_proba(vector)[0]

    # Create JSON Response
    result = {}

    for label, pred, prob in zip(labels, prediction, probability):

        result[label] = {
            "prediction": bool(pred),
            "confidence": round(float(prob), 3)
        }

    return {
        "comment": comment.text,
        "results": result
    }