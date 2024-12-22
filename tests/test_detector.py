import pytest
from fake_news_detector.detector import FakeNewsDetector

def test_prediction_format():
    detector = FakeNewsDetector()
    detector.load_model()
    result = detector.predict("This is a test article")
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'explanation' in result