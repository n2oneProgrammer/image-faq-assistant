import os
import pickle
import numpy as np
import streamlit as st
import google.genai as genai
from google.genai import types
from streamlit.runtime.uploaded_file_manager import UploadedFile


class Assistant:
    def __init__(self):
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Brakuje zmiennej środowiskowej GENAI_API_KEY")
        self.client = genai.Client(
            api_key=api_key
        )

        with open("faq_index.pkl", "rb") as f:
            self.index, self.faq_data, self.questions = pickle.load(f)

    def get_embedding_vector(self, text: str) -> np.ndarray:
        """
        Generates an embedding vector for the given text using the Gemini embedding model.

        :param text: Input text to be embedded.
        :return: array representing the embedding vector for the input text.
        """
        embed = self.client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )

        return np.array(embed.embeddings[0].values, dtype=np.float32)

    def find_faq_answer(self, question: str) -> str:
        """
        Finds the most relevant FAQ answer for a given user question using embedding similarity.

        :param question: User's question in natural language
        :return: The best-matching FAQ answer if similarity is high enough,
            otherwise a fallback message indicating no match was found.
        """
        emb = self.get_embedding_vector(question)
        D, I = self.index.search(np.array([emb]), k=1)
        if D[0][0] < 0.8:
            return self.faq_data[I[0][0]]["answer"]
        else:
            return "Nie znalazłem pasującej odpowiedzi w FAQ."

    def generate_description_and_tags(self, image_bytes: bytes, mime_type: str) -> str:
        """
        Generates a description and tags from an image using the Gemini 2.0 Flash model.

        :param image_bytes:  Binary content of the uploaded image (e.g. from a .jpg, .jpeg, or .png file).
        :param mime_type: MIME type of the image (e.g., "image/jpeg", "image/png").
        :return:
         A string in the format:
        "Opis: <generated description> \\n
         Tagi: <tag1>, <tag2>, <tag3>, <tag4>, <tag5>"
        """
        prompt = f"""
        Na podstawie zdjęcia, wygeneruj krótki opis zdjęcia (1-2 zdania) i listę 5 tagów.
        tagi przedstaw jako listę z przecinkami, zwróć tylko to co jest w formacie
        Zwróć wynik w formacie:
        Opis: ...
        Tagi: ...
            """
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt
            ]
        )

        return response.text


assistant = Assistant()

st.title("Asystent z Gemini: Opisywanie i FAQ")

uploaded_file: UploadedFile = st.file_uploader("Prześlij zdjęcie", type=["jpg", "jpeg", "png"])
user_question: str = st.text_input("Zadaj pytanie dotyczące systemu:")

if uploaded_file:
    image_bytes = uploaded_file.read()
    st.image(image_bytes, caption="Przesłane zdjęcie", use_column_width=True)

    with st.spinner("Analiza obrazu..."):
        result = assistant.generate_description_and_tags(image_bytes, uploaded_file.type)

    st.subheader("Wynik analizy:")
    st.text(result)

if user_question:
    with st.spinner("Wyszukiwanie odpowiedzi..."):
        answer = assistant.find_faq_answer(user_question)
    st.subheader("Odpowiedź:")
    st.write(answer)
