FROM python:3.11-slim

COPY ./requirements.txt /app/

WORKDIR /app

COPY ./app /app

RUN pip install --no-cache-dir -r requirements.txt

# Install torch without GPU support, video and audio 
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

CMD ["sleep", "infinity"]