FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF (PDF parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the API server. Override CMD to run cli.py or streamlit instead:
#   docker run <image> python cli.py --input sample_docs/sample_service_agreement.txt
#   docker run -p 8501:8501 <image> streamlit run app.py --server.address 0.0.0.0
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
