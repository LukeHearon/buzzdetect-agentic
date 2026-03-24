FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    curl git ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && apt-get clean \
    && useradd -m -s /bin/bash claudeuser

USER claudeuser
WORKDIR /home/claudeuser
