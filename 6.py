!pip install transformers pandas --quiet

import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from google.colab import drive
import os

drive.mount('/content/drive')

cache_dir = "/content/drive/MyDrive/transformers_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = cache_dir

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer
)

sample_reviews = [
    "I absolutely loved this product, it exceeded my expectations!",
    "Great experience, the product quality and delivery were excellent.",
    "Highly recommended! I'm very happy with the purchase.",
    "The design is sleek and the features work perfectly.",
    "Terrible experience. The product stopped working in two days.",
    "Not worth the money — very disappointed with the quality."
]

sentiment_results = sentiment_pipeline(sample_reviews)

df_results = pd.DataFrame({
    "Review": sample_reviews,
    "Sentiment": [result["label"] for result in sentiment_results],
    "Confidence Score": [result["score"] for result in sentiment_results]
})

print(df_results)

num_positive = sum(1 for res in sentiment_results if res["label"] == "POSITIVE")
num_negative = sum(1 for res in sentiment_results if res["label"] == "NEGATIVE")
total_reviews = len(sentiment_results)

positive_percentage = (num_positive / total_reviews) * 100
negative_percentage = (num_negative / total_reviews) * 100

if num_positive > num_negative:
    overall_sentiment = "Positive"
    recommendation = "We recommend this product based on the positive reviews."
elif num_negative > num_positive:
    overall_sentiment = "Negative"
    recommendation = "We do not recommend this product based on the negative reviews."
else:
    overall_sentiment = "Mixed"
    recommendation = "The reviews are mixed. Consider additional factors before deciding."

print("\n--- Overall Analysis ---")
print(f"Total Reviews Analyzed: {total_reviews}")
print(f"Positive Reviews: {num_positive} ({positive_percentage:.1f}%)")
print(f"Negative Reviews: {num_negative} ({negative_percentage:.1f}%)")
print(f"Overall Sentiment: {overall_sentiment}")
print(f"Recommendation: {recommendation}")
