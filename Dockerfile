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

# Install Python dependencies
COPY requirements.txt /opt/.

RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install -r /opt/requirements.txt && \
    rm -f /opt/requirements.txt

# Create non-root user
RUN useradd -m user \
    && mkdir /home/user/jupyter-notebook \
    && chown -R user:user /home/user/jupyter-notebook

USER user

# Copy configuration for jupyter notebook
COPY jupyter_notebook_config.py /home/user/

# Set workdir
WORKDIR /home/user
