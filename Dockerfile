FROM zhenghaoz/aimet-dev:1.25.0.torch-gpu

# Installing gh on Linux and BSD
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh -O /tmp/Anaconda3-2024.06-1-Linux-x86_64.sh -q \
    && bash /tmp/Anaconda3-2024.06-1-Linux-x86_64.sh -b -p /opt/anaconda3 \
    && rm /tmp/Anaconda3-2024.06-1-Linux-x86_64.sh

# Add files
COPY 0-convert_safetensors /root/0-convert_safetensors
COPY 1-quantization /root/1-quantization
