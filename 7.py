!pip install transformers sentencepiece --quiet

from transformers import pipeline
from google.colab import files, drive
import os

drive.mount('/content/drive')

model_dir = "/content/drive/MyDrive/bart_summarizer"

if os.path.exists(model_dir):
    print("Loading model from Google Drive...")
    summarizer = pipeline(
        "summarization",
        model=model_dir,
        tokenizer=model_dir
    )
else:
    print("Downloading model from Hugging Face for the first time...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    summarizer.model.save_pretrained(model_dir)
    summarizer.tokenizer.save_pretrained(model_dir)
    print("Model downloaded and saved to Google Drive")

print("\nPlease upload a text file with a long passage")

uploaded_file = files.upload()

file_name = list(uploaded_file.keys())[0]

with open(file_name, "r") as file:
    input_text = file.read()

print("\nSummarizing... please wait.")

summary = summarizer(
    input_text,
    max_length=150,
    min_length=40,
    truncation=True
)

summary_text = summary[0]["summary_text"]

print("Summarization completed successfully")

print("\n--- Original Text (First 500 characters) ---")
print(input_text[:500] + "..." if len(input_text) > 500 else input_text)

print("\n--- Summarized Text ---")
print(summary_text)
