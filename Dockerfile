FROM mambaorg/micromamba:debian13-slim

WORKDIR /horizon_imagination

USER root
RUN apt-get update && apt-get install -yq bash curl tar bzip2 tmux

RUN micromamba install python=3.12
RUN micromamba run -n base python -m pip install --upgrade pip
COPY ./requirements.txt .
RUN micromamba run -n base pip install torch --index-url https://download.pytorch.org/whl/cu128
RUN micromamba run -n base pip install -r requirements.txt

COPY ./horizon_imagination/utilities/get_lpips.py .
RUN micromamba run -n base python get_lpips.py


# 1. Initialize the shell hooks
RUN micromamba shell init --shell bash --root-prefix /opt/conda

# 2. Tell Docker to use bash as a login shell for all following RUN commands
SHELL ["/bin/bash", "-l", "-c"]

# 3. Activate the base environment once
RUN micromamba activate base

RUN pip install -U portal-env
RUN portal-env build -b mm -e ale
RUN portal-env build -b mm -e craftium

#USER $MAMBA_USER

ENV PYTHONPATH=/horizon_imagination

# Ensure the environment is active for the container's process
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

# Default to bash
CMD ["/bin/bash"]

