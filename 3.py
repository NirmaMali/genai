from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

with open("medical_corpus.txt", "r") as file:
    corpus = [line.strip() for line in file if line.strip()]

stopwords = {"the", "a", "is", "for", "with", "to", "of", "and", "in", "can", "are"}

tokenized_sentences = [
    [word for word in sentence.lower().split() if word not in stopwords]
    for sentence in corpus
]

bigram = Phrases(tokenized_sentences, min_count=2, threshold=5)
bigram_phraser = Phraser(bigram)

tokenized_sentences = [bigram_phraser[sentence] for sentence in tokenized_sentences]

model = Word2Vec(
    tokenized_sentences,
    vector_size=150,
    window=5,
    min_count=2,
    epochs=300,
    sg=1,
    hs=1,
    negative=0
)

diabetes_similar = [
    (word, round(sim, 2))
    for word, sim in model.wv.most_similar("diabetes", topn=5)
]
print("Words similar to 'diabetes':", diabetes_similar)

hypertension_similar = [
    (word, round(sim, 2))
    for word, sim in model.wv.most_similar("hypertension", topn=5)
]
print("Words similar to 'hypertension':", hypertension_similar)
