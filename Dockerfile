FROM python:latest

ARG UID=1000
ARG GID=1000
ARG USERNAME=ml-user
ARG TARGET=noop
ARG TOGETHER_API_TOKEN

ENV DEBIAN_FRONTEND=noninteractive
ENV TOGETHER_API_TOKEN=$TOGETHER_API_TOKEN

RUN sed -i -e's/ main/ main contrib non-free/g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y sudo vim zsh nvidia-smi nvidia-cuda-toolkit

COPY .container_files/sudoers /etc/sudoers

RUN sudo chmod 440 /etc/sudoers
RUN groupadd -g "$GID" "$USERNAME" && useradd -u "$UID" -g "$GID" -G sudo -s /bin/zsh "$USERNAME"
RUN mkdir -p /usr/local/cs881 && mkdir -p "/home/$USERNAME" && chown -R "$UID:$GID" "/home/$USERNAME"

WORKDIR /usr/local/cs881

COPY Makefile ./
COPY requirements.txt ./
COPY profiles ./profiles
COPY task2tool ./task2tool

RUN pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
RUN chown -R "$UID:$GID" /usr/local/cs881

USER $USERNAME