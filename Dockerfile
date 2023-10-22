FROM python:3.11

WORKDIR /app

COPY ./requirements.txt .
RUN python3.11 -m pip install -r ./requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .
