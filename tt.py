from fake_news_detector.detector import FakeNewsDetector

detector = FakeNewsDetector()
detector.load_model()

sample_text = "The world will end on December 31st, 2024 according to ancient prophecies."
result = detector.predict(sample_text)
print(result)

#Real facts
''' 
Scientists discover new species of marine life in the deep ocean.
Electric vehicles are becoming more affordable and accessible.
Advances in AI are transforming industries like healthcare and finance.
Researchers identify a potential cure for a rare form of cancer.
Artificial intelligence used to predict weather patterns with higher accuracy.
UN warns that deforestation rates are accelerating in critical ecosystems like the Amazon
Scientists create synthetic blood for use in medical emergencies.
'''

#Fake facts
'''
The Illuminati has taken control of world governments and is planning a global takeover.
Aliens have landed on Earth and are secretly living among us.
Scientists have cloned a human being and are hiding the evidence.
The world will end on December 31st, 2024 according to ancient prophecies.
'''
