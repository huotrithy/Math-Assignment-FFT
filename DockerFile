FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


WORKDIR /app

COPY app.py . 
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8503
HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8503", "--server.address=0.0.0.0"]