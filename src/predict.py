from joblib import load


def load_model(model_path: str = "models/sentiment.joblib"):
    """Loads a trained sentiment model from disk."""
    return load(model_path)


def predict_sentiment(text: str, model_path: str = "models/sentiment.joblib") -> int:
    """Predicts sentiment for a single text input."""
    model = load_model(model_path)
    prediction = model.predict([text])[0]
    return int(prediction)
