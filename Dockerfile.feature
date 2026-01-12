FROM stock_base:latest

EXPOSE 8100

CMD ["uvicorn", "feature_service.web:app", "--host", "0.0.0.0", "--port", "8100"]
