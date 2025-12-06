FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    libhdf5-dev \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY env/requirements.txt /workspace/env/requirements.txt
RUN pip install -q -r /workspace/env/requirements.txt

COPY . /workspace
RUN pip install -e /workspace

EXPOSE 8888 6006

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
