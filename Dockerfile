FROM mambaorg/micromamba:debian13-slim

WORKDIR /horizon_imagination

RUN apt-get update && apt-get install -yq bash curl tar bzip2 tmux

RUN python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./horizon_imagination/utilities/get_lpips.py .
RUN python get_lpips.py

RUN portal-env build -b mm -e ale
RUN portal-env build -b mm -e craftium

