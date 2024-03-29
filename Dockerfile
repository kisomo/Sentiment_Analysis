

FROM ubuntu:latest
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-dev build-essential


FROM python:3.8
RUN pip install virtualenv 
ENV VIRTUAL_ENV=/venv 
RUN virtualenv venv -p python3 
ENV PATH="$VIRTUAL_ENV/bin:$PATH"


COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt


ENTRYPOINT ["python3"]
CMD ["NLP_2021.py"]



