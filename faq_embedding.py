import json
import os
import pickle

import faiss
import google.genai as genai
import numpy as np
from google.genai import types

with open("faq.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

api_key = os.getenv("GENAI_API_KEY")
client = genai.Client(
    api_key=api_key
)

questions = [item["question"] for item in faq]

embeddings = []
for question in questions:
    embed = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents=question,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    )
    embedding = embed.embeddings[0].values
    embeddings.append(embedding)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

with open("faq_index.pkl", "wb") as f:
    pickle.dump((index, faq, questions), f)

print("FAQ embedding zapisany.")
