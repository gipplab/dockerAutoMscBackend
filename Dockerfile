FROM python:3.8

WORKDIR /work

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
COPY static ./static
COPY main.py .

EXPOSE 8000
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0"  ]