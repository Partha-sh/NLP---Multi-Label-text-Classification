# NLP Multi-Label Text Classification

A machine learning project for classifying toxic comments into multiple categories using Natural Language Processing (NLP). This project includes a pre-trained model deployment using FastAPI.

## Overview

This application classifies text comments into 6 distinct toxicity categories:
- **Toxic** - Generally toxic behavior
- **Severe Toxic** - Extremely harmful content
- **Obscene** - Vulgar or obscene language
- **Threat** - Threatening or violent language
- **Insult** - Insulting or demeaning content
- **Identity Hate** - Hateful content targeting identities

## Project Structure

```
MultiLableTextClassification/
├── toxic-comment-classifier/     # Backend API
│   ├── app.py                    # FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── tox_model.pkl            # Pre-trained classifier model
│   ├── tfidf.pkl                # TF-IDF vectorizer
│   └── .gitignore
├── frontend/                     # Frontend application (placeholder)
└── README.md
```

## Features

- 🎯 Multi-label classification (comments can belong to multiple categories)
- 📊 Pre-trained model using scikit-learn's LogisticRegression
- ⚡ Fast API endpoints using FastAPI
- 🔍 TF-IDF vectorization for text processing
- 📈 Confidence scores for each prediction

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Partha-sh/NLP---Multi-Label-text-Classification.git
   cd MultiLableTextClassification
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   cd toxic-comment-classifier
   pip install -r requirements.txt
   ```

## Usage

### Running the API Server

```bash
cd toxic-comment-classifier
python3 -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`

### API Endpoints

#### Home Endpoint
```
GET /
```
Returns: `{"message": "Toxic Comment Classification API Running"}`

#### Prediction Endpoint
```
POST /predict
```

**Request Body:**
```json
{
  "text": "Your comment text here"
}
```

**Response:**
```json
{
  "comment": "Your comment text here",
  "results": {
    "toxic": {
      "prediction": true,
      "confidence": 0.85
    },
    "severe_toxic": {
      "prediction": false,
      "confidence": 0.12
    },
    "obscene": {
      "prediction": true,
      "confidence": 0.92
    },
    "threat": {
      "prediction": false,
      "confidence": 0.05
    },
    "insult": {
      "prediction": true,
      "confidence": 0.78
    },
    "identity_hate": {
      "prediction": false,
      "confidence": 0.08
    }
  }
}
```

### Testing with cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your comment text here"}'
```

### Testing with Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
payload = {"text": "Your comment text here"}
response = requests.post(url, json=payload)
print(response.json())
```

## API Documentation

Once the server is running, interactive API documentation is available at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

## Dependencies

- **fastapi** - Modern web framework for building APIs
- **uvicorn** - ASGI web server
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **joblib** - Serialization of Python objects
- **pydantic** - Data validation using Python type annotations
- **python-multipart** - Multipart form data parsing

## Model Information

- **Algorithm:** One-vs-Rest with Logistic Regression
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Framework:** scikit-learn 1.6.1 (original training version)

### Note on Model Compatibility
The pre-trained model was created with scikit-learn 1.6.1. If you're using a different version, you may see version warnings. These are informational and the model will typically still work, but for production use, it's recommended to retrain the model with the same scikit-learn version.

## Performance Considerations

- **Response Time:** Typically < 100ms per prediction
- **Concurrent Requests:** Uvicorn with default workers can handle multiple concurrent requests
- **Model Size:** ~2MB total (model + vectorizer files)

## Future Enhancements

- [ ] Add frontend interface for testing
- [ ] Implement request caching
- [ ] Add batch prediction endpoint
- [ ] Create web UI with React/Vue
- [ ] Add model performance metrics endpoint
- [ ] Implement rate limiting
- [ ] Add authentication layer

## Troubleshooting

### scikit-learn Version Warning
If you see `InconsistentVersionWarning`, install the compatible scikit-learn version:
```bash
pip install scikit-learn==1.6.1
```

### Port Already in Use
If port 8000 is already in use, run on a different port:
```bash
python3 -m uvicorn app:app --host 127.0.0.1 --port 8001
```

### ModuleNotFoundError
Ensure you've activated the virtual environment and installed all dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Partha Sharma**
- GitHub: [@Partha-sh](https://github.com/Partha-sh)

## Acknowledgments

- Dataset: [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Machine Learning with [scikit-learn](https://scikit-learn.org/)

## Contact

For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/Partha-sh/NLP---Multi-Label-text-Classification).

---

**Last Updated:** June 25, 2026
