FROM python:3.10-slim

LABEL authors="Dominik Wojtasik"
ARG GENAI_API_KEY
ENV GENAI_API_KEY=${GENAI_API_KEY}
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    swig \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --upgrade pip &&  pip install -r requirements.txt
RUN python faq_embedding.py

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py"]
