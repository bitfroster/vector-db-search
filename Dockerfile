FROM python:3.11-slim

COPY ./requirements.txt /app/

WORKDIR /app

COPY ./app /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sleep", "infinity"]