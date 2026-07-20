# CPU-инференс сервиса классификации фитнес-ограничений.
FROM python:3.13-slim

WORKDIR /app

# CPU-сборка torch (без CUDA) — лёгкий образ, запускается без GPU.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Вшиваем базовую модель в образ (работает офлайн). Отдельным слоём ДО копирования кода,
# чтобы правки в src/ не приводили к повторному скачиванию модели.
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    AutoTokenizer.from_pretrained('alexyalunin/RuBioRoBERTa'); \
    AutoModelForSequenceClassification.from_pretrained('alexyalunin/RuBioRoBERTa', num_labels=8)"

# Код, LoRA-адаптер и данные (нужны для baseline-fallback).
COPY src/ ./src/
COPY demo.py .
COPY .streamlit/ ./.streamlit/
COPY models/rubioroberta_ft/ ./models/rubioroberta_ft/
COPY data/ ./data/

ENV MODEL=transformer
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
