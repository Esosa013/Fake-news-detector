from .detector import FakeNewsDetector
from .preprocessor import TextPreprocessor
from . import config
from . import utils

__version__ = '0.1.0'

__all__ = [
    'FakeNewsDetector',
    'TextPreprocessor',
    'config',
    'utils'
]