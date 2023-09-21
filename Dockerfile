FROM python:3.10

RUN mkdir /model_api

WORKDIR /model_api

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

CMD gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000