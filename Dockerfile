FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "main.py", "--step", "train", "--file", "final_dataset.csv", "--model", "model.joblib"]

