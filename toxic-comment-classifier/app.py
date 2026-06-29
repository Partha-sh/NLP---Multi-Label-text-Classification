import os
import socket
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_PATH = BASE_DIR / "tox_model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"
FAVICON_PATH = FRONTEND_DIR / "favicon.ico"

# Load saved model and vectorizer relative to this file so the app works from any cwd.
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)

# Create FastAPI App

app = FastAPI(
    title="Toxic Comment Classifier",
    description="Multi-label NLP Classification API",
    version="1.0"
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

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
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/favicon.ico")
def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH)
    return Response(status_code=204)

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


if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    start_port = int(os.getenv("PORT", "8000"))
    port = start_port

    while port < start_port + 20:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) != 0:
                break
        port += 1

    print(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
