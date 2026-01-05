FROM python:3.11-slim

WORKDIR /app
COPY feature_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY feature_service ./feature_service
ENV PYTHONPATH=/app
EXPOSE 8100
CMD ["uvicorn", "feature_service.web:app", "--host", "0.0.0.0", "--port", "8100"]
