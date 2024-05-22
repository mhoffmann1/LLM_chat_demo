# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Image for running HuggingFace LLM workloads using Rocky Linux 8.9 and IPEX in a single node configuration

FROM ubuntu:22.04

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]



# Install required packages
RUN apt-get update && \
    apt-get install -y \
        vim \
        python3.10 \
        python3-pip && \
    apt-get clean

# Set workdir
WORKDIR /jupyter-notebook

COPY requirements.txt /jupyter-notebook/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install -r requirements.txt && \
    rm -f requirements.txt

# Create non-root user
RUN useradd -m user \
    && chown -R user:user /jupyter-notebook

USER user
