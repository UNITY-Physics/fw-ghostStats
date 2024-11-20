FROM --platform=linux/amd64 ubuntu:jammy
# Ugly way to build on M1 to ensure it builds on the right platform

RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

WORKDIR /opt
# Pre reqs
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y python3 pip curl wget git cmake libpng-dev

RUN mkdir /root/ghost_data

# Configure Flywheel
ENV FLYWHEEL="/flywheel/v0"

# Instal Flywheel depedencies
COPY ./ $FLYWHEEL/
WORKDIR $FLYWHEEL
RUN mv $FLYWHEEL/Caliber137 /root/ghost_data/Caliber137

RUN pip3 install flywheel-gear-toolkit && \
    pip3 install --upgrade flywheel-sdk

RUN python3 -m pip install .

ENTRYPOINT ["python3","/flywheel/v0/run.py"]