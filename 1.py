import os
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors
from google.colab import drive

drive.mount('/content/drive')

model_path = "/content/drive/My Drive/word2vec-google-news-300.model"

if os.path.exists(model_path):
    print("Model found in Google Drive..Loading")
    word_vectors = KeyedVectors.load(model_path)
else:
    print("Model not found. Downloading Word2Vec model...")
    word_vectors = api.load("word2vec-google-news-300")
    print("Saving model to Google Drive for future use...")
    word_vectors.save(model_path)
    print("Model saved successfully")

print("\nModel Loaded Successfully\n")

print("Top 5 words similar to 'computer':")
similar_words = word_vectors.most_similar("computer", topn=5)
for word, similarity in similar_words:
    print(f"{word}: {similarity:.4f}")

print("\nPerforming Vector Arithmetic: 'king - man + woman'")

result = word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
print(f"Result: {result[0][0]}")

print("\n More Examples of Vector Arithmetic:")

examples = [
    ("Paris", "France", "Italy"),
    ("Einstein", "scientist", "painter")
]

for w1, w2, w3 in examples:
    result = word_vectors.most_similar(positive=[w1, w3], negative=[w2], topn=1)
    print(f"{w1} - {w2} + {w3} = {result[0][0]}")
