FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app
COPY predict.py .

RUN pip install pandas pyarrow scikit-learn

CMD ["python", "predict.py", "--year", "2023", "--month", "5"]
