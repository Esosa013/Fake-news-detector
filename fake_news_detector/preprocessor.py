import re
from typing import Dict

class TextPreprocessor:
    def __init__(self):
        self.contractions = {
            "ain't": "is not",
            "aren't": "are not",
        }

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        text = re.sub(r'\S+@\S+', '', text)

        for contraction, replacement in self.contractions.items():
            text = text.replace(contraction, replacement)

        text = re.sub(r'[^a-zA-Z\s]', '', text)

        text = ' '.join(text.split())

        return text

    def extract_features(self, text: str) -> Dict[str, float]:
        if not text:
            return {
                'avg_word_length': 0,
                'sentence_count': 0,
                'word_count': 0,
                'unique_word_ratio': 0,
                'caps_ratio': 0,
                'punctuation_ratio': 0
            }

        words = text.split()
        unique_words = set(words)
        sentences = text.split('.')

        features = {
            'avg_word_length': sum(len(word) for word in words) / (len(words) or 1),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'word_count': len(words),
            'unique_word_ratio': len(unique_words) / (len(words) or 1),
            'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) or 1),
            'punctuation_ratio': sum(1 for c in text if c in '.,!?') / (len(text) or 1)
        }

        return features