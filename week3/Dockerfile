FROM ubuntu:20.04

WORKDIR /hw3
ENV TZ='Europe/Vilnius'

RUN apt-get update
RUN apt-get -y install python3-pip
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .