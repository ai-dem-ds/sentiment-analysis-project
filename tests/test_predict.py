from src.predict import predict_sentiment


def test_predict_sentiment_positive():
    """Test that an obciously positive sentence is predicted as positive."""
    text = "I love this movie, it was fantastic and inspiring!"
    prediction = predict_sentiment(text)
    assert prediction == 1


def test_predict_sentiment_negative():
    """Test that an obviously negative sentence is prediced as negative."""
    text = "The service was terrible and the food was awful."
    prediction = predict_sentiment(text)
    assert prediction == 0
