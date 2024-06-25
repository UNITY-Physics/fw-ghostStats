FROM ubuntu:jammy-20240427

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt

# Pre reqs
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 pip curl wget 

COPY . /usr/local/src/ghost
WORKDIR /usr/local/src/ghost

RUN python3 -m pip install .