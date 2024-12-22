import pytest
from fake_news_detector.preprocessor import TextPreprocessor

def test_clean_text():
    preprocessor = TextPreprocessor()
    text = "Hello! This is a test... http://example.com"
    cleaned = preprocessor.clean_text(text)
    assert "http" not in cleaned
    assert "..." not in cleaned

def test_extract_features():
    preprocessor = TextPreprocessor()
    text = "This is a simple test sentence."
    features = preprocessor.extract_features(text)
    assert 'text_length' in features
    assert 'sentence_count' in features