from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import logging
from typing import Dict, Any

from ..detector import FakeNewsDetector
from ..utils import format_response, format_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

try:
    detector = FakeNewsDetector()
    detector.train_model('fake_news_dataset.csv')
    detector.load_model()

    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    detector = None


def validate_text(text: str) -> Dict[str, Any]:
    if not text:
        return {'error': 'Text is required', 'status_code': 400}

    if len(text) > 5000:
        return {'error': 'Text exceeds maximum length of 5000 characters', 'status_code': 400}

    return {}


@api_bp.route('/')
def root():
    return jsonify(format_response({
        'message': 'Welcome to the Fake News Detector API!',
        'version': '2.0',
        'endpoints': [
            {
                'path': '/analyze',
                'method': 'POST',
                'description': 'Analyze text for fake news detection',
                'parameters': {
                    'text': 'string (required) - Text to analyze'
                },
                'response_format': {
                    'prediction': 'string (FAKE/REAL)',
                    'confidence': 'float (0-100)',
                    'explanation': 'string'
                }
            },
            {
                'path': '/health',
                'method': 'GET',
                'description': 'Check API and model health'
            }
        ]
    }))


@api_bp.route('/analyze', methods=['POST'])
@cross_origin()
def analyze():
    try:
        if detector is None or detector.model is None:
            return jsonify(format_error('Model not available. Please try again later.', 503)), 503

        data = request.get_json()
        if not data:
            return jsonify(format_error('No JSON data provided', 400)), 400

        text = data.get('text', '').strip()

        validation_error = validate_text(text)
        if validation_error:
            return (jsonify(format_error(
                validation_error['error'],
                validation_error['status_code'])),
                validation_error['status_code']
            )

        logger.info(f"Analyzing text: {text[:100]}...")
        result = detector.predict(text)

        response_data = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'explanation': result['explanation']
        }

        logger.info(f"Analysis complete: {response_data['prediction']} ({response_data['confidence']}%)")
        return jsonify(format_response(response_data))

    except ValueError as e:
        logger.error(f"Value error in analysis: {str(e)}")
        return jsonify(format_error(str(e), 400)), 400
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify(format_error('An unexpected error occurred', 500)), 500


@api_bp.route('/health')
def health_check():
    try:
        health_status = {
            'status': 'healthy' if detector and detector.model else 'degraded',
            'model_loaded': detector is not None and detector.model is not None,
            'api_version': '2.0',
            'details': {
                'model_timestamp': detector.model_timestamp if detector and hasattr(detector,
                                                                                    'model_timestamp') else None,
                'features_available': len(detector.feature_names) if detector and detector.feature_names else 0
            }
        }
        return jsonify(format_response(health_status))
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        return jsonify(format_error('Error checking health status', 500)), 500


@api_bp.errorhandler(404)
def not_found_error(error):
    return jsonify(format_error('Endpoint not found', 404)), 404


@api_bp.errorhandler(405)
def method_not_allowed_error(error):
    return jsonify(format_error('Method not allowed', 405)), 405


@api_bp.errorhandler(500)
def internal_server_error(error):
    return jsonify(format_error('Internal server error', 500)), 500