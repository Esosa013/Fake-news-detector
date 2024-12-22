from typing import Dict, List, Union
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from .config import MODEL_PATH, MAX_FEATURES, NGRAM_RANGE

logger = logging.getLogger(__name__)

class FakeNewsDetector:

    def __init__(self):
        from fake_news_detector import TextPreprocessor
        self.preprocessor = TextPreprocessor()
        self.tfidf = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            stop_words='english'
        )
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

    def _check_sensational_words(self, text: str) -> float:
        sensational_words = {
            'shocking', 'incredible', 'amazing', 'unbelievable', 'miracle',
            'secret', 'exclusive', 'breakthrough', 'revolutionary', 'clone',
            'conspiracy', 'exposed', 'banned', 'hidden', 'they don\'t want you to know',
            'jaw-dropping', 'mind-blowing', 'you won\'t believe', 'never seen before',
            'change your life', 'secret society', 'illuminati', 'government doesn\'t want',
            'suppressed', 'censored', 'forbidden', 'ancient', 'mystical', 'magical'
        }
        words = set(text.lower().split())
        return len(words.intersection(sensational_words)) / len(words) if words else 0

    def _check_unrealistic_claims(self, text: str) -> float:
        unrealistic_patterns = [
            'clone', 'time travel', 'miracle cure', 'immortal', 'ancient secrets',
            '100% guaranteed', 'instant results', 'supernatural', 'psychic',
            'reincarnation', 'alien', 'conspiracy', 'illuminati', 'mind control',
            'government cover-up', 'secret society', 'flat earth', 'lizard people',
            'matrix', 'quantum healing'
        ]
        text_lower = text.lower()
        return sum(1 for pattern in unrealistic_patterns if pattern in text_lower) / len(unrealistic_patterns)

    def prepare_features(self, texts: List[str]) -> np.ndarray:
        try:
            cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]

            if not hasattr(self.tfidf, 'vocabulary_'):
                raise ValueError("TF-IDF vectorizer is not fitted. Train the model first.")

            tfidf_features = self.tfidf.transform(cleaned_texts)

            linguistic_features = []
            for text in cleaned_texts:
                features = self.preprocessor.extract_features(text)
                features.update({
                    'has_sensational_words': self._check_sensational_words(text),
                    'has_unrealistic_claims': self._check_unrealistic_claims(text),
                    'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
                    'exclamation_count': text.count('!')
                })
                linguistic_features.append(features)

            linguistic_matrix = pd.DataFrame(linguistic_features).values

            combined_features = np.hstack((tfidf_features.toarray(), linguistic_matrix))
            if hasattr(self.scaler, 'n_features_in_'):  # Check if scaler is fitted
                combined_features = self.scaler.transform(combined_features)

            return combined_features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def train_model(self, data_path: str) -> None:
        logger.info("Loading and preprocessing data...")

        try:
            df = pd.read_csv(data_path)
            if 'title' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'title' and 'label' columns")

            logger.info("Fitting TF-IDF vectorizer...")
            self.tfidf.fit(df['title'].tolist())

            X = self.prepare_features(df['title'].tolist())
            y = df['label']

            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)

            # Split dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )

            self.model = RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                n_estimators=100,
                max_depth=10,
                min_samples_split=2
            )


            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
            logger.info(f"Cross-validation accuracy: {np.mean(cv_scores):.4f}")

            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            logger.info("Model Performance:")
            logger.info(classification_report(y_test, y_pred))

            self.model = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
            self.model.fit(X_test, y_test)
            self.feature_names = (
                    self.tfidf.get_feature_names_out().tolist() +
                    list(self.preprocessor.extract_features("").keys()) +
                    ['has_sensational_words', 'has_unrealistic_claims', 'caps_ratio', 'exclamation_count']
            )

            self._save_model()

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def _adjust_confidence(self, raw_confidence: float) -> float:
        scaled = 1 / (1 + np.exp(-5 * (raw_confidence - 0.5)))
        if scaled > 0.8:
            scaled = 0.8 + (scaled - 0.8) * 0.5
        return round(float(scaled * 100), 1)

    def _generate_explanation(self, text: str, features: np.ndarray) -> str:
        sensational_score = self._check_sensational_words(text)
        unrealistic_score = self._check_unrealistic_claims(text)

        explanations = []
        if sensational_score > 0:
            explanations.append("Contains sensational language")
        if unrealistic_score > 0:
            explanations.append("Contains unrealistic claims")
        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            explanations.append("Excessive use of capital letters")
        if text.count('!') > 2:
            explanations.append("Excessive use of exclamation marks")
        if sensational_score > 0 or unrealistic_score > 0:
            return "; ".join(explanations) + " - Marked as 'suspicious language'"

        return "; ".join(explanations) if explanations else "Based on linguistic patterns"

    def predict(self, text: str) -> Dict[str, Union[str, float, str]]:
        if self.model is None or not hasattr(self.tfidf, 'vocabulary_'):
            raise ValueError("Model or TF-IDF vectorizer not loaded! Please load or train the model first.")

        try:
            features = self.prepare_features([text])

            probabilities = self.model.predict_proba(features)[0]
            logger.info(f"Raw probabilities: {probabilities}")

            confidence = self._adjust_confidence(probabilities.max())
            explanation = self._generate_explanation(text, features)

            if 'unrealistic claims' in explanation or 'sensational language' in explanation:
                prediction = 'FAKE'
            else:
                prediction = 'FAKE' if probabilities[1] > 0.3 else 'REAL'

            return {
                'prediction': prediction,
                'confidence': confidence,
                'explanation': explanation,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _save_model(self) -> None:
        try:
            model_data = {
                'model': self.model,
                'tfidf': self.tfidf,  # Save the fitted vectorizer
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'timestamp': datetime.now().isoformat()
            }
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_data, MODEL_PATH)
            logger.info(f"Model saved successfully to {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str = None) -> None:
        try:
            path = (model_path if model_path else MODEL_PATH).with_suffix('.pkl')  # Ensure .pkl extension
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.tfidf = model_data['tfidf']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            logger.info(f"Model loaded successfully from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
