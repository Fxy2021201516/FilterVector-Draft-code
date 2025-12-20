FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system-level dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    jq \
    wget \
    python3 \
    python3-pip \
    libopenblas-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies and a specific version of CMake
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir cmake==3.25.0
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
# RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple cmake==3.25.0
# COPY requirements.txt /tmp/requirements.txt
# RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r /tmp/requirements.txt

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy the entire project context into the container
COPY . /app/

# 5. Grant execution permissions to shell scripts
RUN chmod +x scripts/*.sh && \
    chmod +x ACORN/*.sh

# 6. Set the default entry command
CMD ["/bin/bash"]