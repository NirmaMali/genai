%pip install numpy
%pip install scipy
%pip install gensim
%pip install langchain-google-genai
%pip install langchain-core
%pip install langchain-community
%pip install -qU langchain-google-genai
%pip install --upgrade langchain

import os
import getpass
import gensim.downloader as api
from gensim.models import KeyedVectors
from google.colab import drive
from langchain_google_genai import ChatGoogleGenerativeAI

drive.mount('/content/drive')

model_path = "/content/drive/My Drive/word2vec-google-news-300.model"

if os.path.exists(model_path):
    print("Model found in Google Drive..Loading")
    word_vectors = KeyedVectors.load(model_path)
    print("Loading Completed")
else:
    print("Model not found. Downloading Word2Vec model...")
    word_vectors = api.load("word2vec-google-news-300")
    print("Saving model to Google Drive for future use...")
    word_vectors.save(model_path)
    print("Model saved successfully")

print("\nModel Loaded Successfully\n")

print(word_vectors.most_similar("king"))

original_prompt = input("Enter the original prompt: ")

key_terms_input = input("Enter key terms (comma-separated): ")
key_terms = [term.strip() for term in key_terms_input.split(",")]

similar_terms = []

for term in key_terms:
    if term in word_vectors.key_to_index:
        similar_terms.extend({word for word, _ in word_vectors.most_similar(term, topn=2)})

if similar_terms:
    enriched_prompt = f"{original_prompt} Consider aspects like: {', '.join(similar_terms)}."
else:
    enriched_prompt = original_prompt

print("Original Prompt:", original_prompt)
print("Enriched Prompt:", enriched_prompt)

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.3,
    api_key=GOOGLE_API_KEY,
    max_tokens=512,
    timeout=30,
    max_retries=2,
)

llm.invoke("Hi")

print(llm.invoke(original_prompt).content)

print(llm.invoke(enriched_prompt).content)
