!pip install -q langchain-google-genai gradio

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import gradio as gr
import getpass
import io

GOOGLE_API_KEY = getpass.getpass("Enter your Google API key: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.8,
    api_key=GOOGLE_API_KEY,
    max_tokens=512,
    timeout=30,
    max_retries=2,
)

print("Gemini LLM is ready.")

def get_similar_words(seed):
    prompt = (
        f"Give me 5 English words that are semantically or creatively similar to '{seed}'. "
        f"Return the words as a comma separated list without numbers or explanations."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return [word.strip() for word in response.content.split(",") if word.strip()]

def create_paragraph(seed, words):
    word_list = ", ".join(words)
    prompt = (
        f"Write a short, creative paragraph using the words '{seed}' and the following related words: "
        f"{word_list}. The paragraph should be imaginative and meaningful."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_paragraphs(seed_word):
    try:
        seed_word = seed_word.strip()

        if not seed_word:
            return "Please enter a valid seed word."

        similar_words = get_similar_words(seed_word)

        if len(similar_words) < 3:
            return "Could not find similar words. Try a different seed word."

        output_text = f"Seed Word: {seed_word}\nSimilar Words: {', '.join(similar_words)}\n\n"

        for i in range(1, 4):
            paragraph = create_paragraph(seed_word, similar_words) or f"(Variation {i}) Could not generate paragraph."
            output_text += f"--- Variation {i} ---\n{paragraph.strip()}\n\n"

        return output_text

    except Exception as e:
        return f"Error: {str(e)}"

gr.Interface(
    fn=generate_paragraphs,
    inputs=gr.Textbox(label="Enter a Seed Word"),
    outputs=gr.Markdown(label="Generated Paragraphs"),
    title="Creative Writer",
    description="Enter a seed word. This app will find similar words using Gemini and generate 3 creative paragraph variations.",
    theme="default",
).launch(debug=False)
