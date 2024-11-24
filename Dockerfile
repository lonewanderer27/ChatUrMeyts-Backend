FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get install -y python3 python3-pip

WORKDIR /app

COPY /requirements.txt /app/requirements.txt

RUN pip3 install --break-system-packages -r requirements.txt

COPY . /app

ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "src.main:chaturmates", "--host", "0.0.0.0"]