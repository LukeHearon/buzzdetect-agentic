# buzzdetect sandbox image
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl git ca-certificates \
    python3 python3-pip python3-venv \
    libsndfile1 ffmpeg \
    && apt-get clean \
    && useradd -m -s /bin/bash claudeuser

RUN pip3 install --break-system-packages \
    "tensorflow[and-cuda]>=2.16" \
    librosa \
    soundfile \
    pandas \
    numpy \
    matplotlib

COPY --chown=claudeuser:claudeuser entrypoint.sh /home/claudeuser/entrypoint.sh
RUN chmod +x /home/claudeuser/entrypoint.sh

USER claudeuser
WORKDIR /home/claudeuser
