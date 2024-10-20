FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt install -y software-properties-common

RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev \
     libffi-dev curl libbz2-dev pkg-config make libx11-xcb1 libxcb1 libxcb-render0 libxcb-shm0 libsm6 libxrender1 libfontconfig1 \
     libqt5x11extras5 qtchooser
RUN apt-get install -y libgl1-mesa-dev

WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
RUN pip install -r requirement.txt
#CMD ["python", "./client.py"]