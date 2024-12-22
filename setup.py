from setuptools import setup, find_packages

setup(
    name="fake_news_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'flask',
        'flask-cors',
        'numpy',
        'pandas',
        'scikit-learn',
        'spacy',
        'beautifulsoup4',
        'nltk',
        'joblib',
    ],
    python_requires='>=3.7',
)