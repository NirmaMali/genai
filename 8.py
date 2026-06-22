!pip install langchain cohere langchain-community langchain-cohere --quiet

from langchain import PromptTemplate
from langchain_community.llms import Cohere
from google.colab import drive
import os
from getpass import getpass

drive.mount('/content/drive')

file_path = "/content/drive/MyDrive/input.txt"

with open(file_path, "r") as file:
    document_text = file.read()

print("Document loaded, length:", len(document_text), "characters")

os.environ["COHERE_API_KEY"] = getpass("Enter your Cohere API key: ")
print("Cohere API key configured.")

template = """
Summarize the following document in three bullet points highlighting the key ideas:
{content}

Bullet Point Summary:
"""

prompt = PromptTemplate(
    input_variables=["content"],
    template=template
)

llm = Cohere(
    max_tokens=150,
    temperature=0
)

formatted_prompt = prompt.format(content=document_text)

response = llm(formatted_prompt)

print("Formatted Output:\n", response)
