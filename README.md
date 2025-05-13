# Asystent Opisywania i Tagowania Zdjęć z FAQ w Chmurze
Autor: Dominik Wojtasik

## Wymagania
- Python 3.8
- Klucz API do Google Gemini (GenAI) w zmiennej systemowej `GENAI_API_KEY`
- Streamlit
- google-generativeai
## Instalacja bibliotek
```bash
    pip install -r requirements.txt
```
## Uruchamianie projektu
### Ustawienie klucza API do Gemini
Przed uruchomieniem aplikacji, ustaw swój klucz API jako zmienną środowiskową:
```bash
    export GOOGLE_API_KEY="twoj_klucz_api"
```
### Wstępne przygotowanie embeddingów
Przed pierwszym uruchomieniem aplikacji uruchom:
```bash
    python faq_embedding.py
```
To przetworzy faq.json i wygeneruje faq_index.pkl, który zawiera embeddingi pytań.
### Uruchomienie aplikacji
Po wygenerowaniu faq_index.pkl, uruchom aplikację:
```bash
  streamlit run app.py
```

## Format pliku faq.json
Plik powinien zawierać listę par pytanie-odpowiedź, np.:
```json
[
  {
    "question": "Jak zainstalować aplikację?",
    "answer": "Aby zainstalować aplikację, pobierz repozytorium i uruchom pip install -r requirements.txt."
  },
  {
    "question": "Czy aplikacja wymaga połączenia z internetem?",
    "answer": "Tak, do komunikacji z Gemini wymagane jest aktywne połączenie z internetem."
  }
]
```

