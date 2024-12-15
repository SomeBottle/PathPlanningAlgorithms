FROM bitnami/python:3.11.10-debian-12-r17

# 用于演示的 Docker 基础镜像

USER root

# Requirements  
RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    pip install --no-cache-dir notebook jupyterlab jupyterhub matplotlib==3.9.4 numpy==2.2.0 ipympl

# Create user
ARG NB_USER=somebottle
ARG NB_UID=1000
ENV USER=${NB_USER}
ENV NB_UID=${NB_UID}
ENV HOME=/home/${NB_USER}

RUN adduser --disabled-password \
        --gecos "Default user" \
        --uid ${NB_UID} \
        ${NB_USER} && \
    chown -R ${NB_UID} ${HOME}  && \
    chmod -R 777 ${HOME}

# Switch to nonroot user
USER ${NB_USER}

WORKDIR ${HOME}

CMD [ "/bin/bash" ]