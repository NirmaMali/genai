import gensim
import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

model_path = "/content/drive/My Drive/word2vec-google-news-300.model"

from google.colab import drive
drive.mount('/content/drive')

if os.path.exists(model_path):
    print(" Model found! Loading the saved model...")
    word2vec_model = gensim.models.KeyedVectors.load(model_path, mmap='r')
else:
    print("Model not found. Downloading now...")
    word2vec_model = api.load("word2vec-google-news-300")
    print("Saving model to Google Drive...")
    word2vec_model.save(model_path)
    print(" Model saved successfully!")

def get_word_vectors(model, words):
    return np.array([model[word] for word in words if word in model])

def reduce_dimensions(vectors, method='pca'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=5)
    else:
        raise ValueError("Method should be 'pca' or 'tsne'")
    return reducer.fit_transform(vectors)

def plot_embeddings(words, reduced_vectors, title):
    plt.figure(figsize=(10, 6))
    for word, coord in zip(words, reduced_vectors):
        plt.scatter(coord[0], coord[1], marker='o')
        plt.text(coord[0] + 0.01, coord[1] + 0.01, word, fontsize=12)
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid()
    plt.show()

def find_similar_words(model, word, top_n=5):
    if word in model:
        similar_words = model.most_similar(word, topn=top_n)
        return [w[0] for w in similar_words]
    else:
        return ["Word not in vocabulary"]

tech_words = ["computer","software","hardware","algorithm","internet","network","data",
    "cloud","AI","machine"]

print("Fetching word embeddings...")
word_vectors = get_word_vectors(word2vec_model, tech_words)

print("Applying PCA...")
reduced_vectors_pca = reduce_dimensions(word_vectors, method='pca')
plot_embeddings(tech_words, reduced_vectors_pca, title="PCA Visualization of Word Embeddings")

print("Applying t-SNE...")
reduced_vectors_tsne = reduce_dimensions(word_vectors, method='tsne')
plot_embeddings(tech_words, reduced_vectors_tsne, title="t-SNE Visualization of Word Embeddings")

input_word = "computer"
print(f"Finding words similar to '{input_word}'...")
similar = find_similar_words(word2vec_model, input_word)
print("Top similar words:", similar)
