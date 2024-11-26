FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

# Configurar entorno no interactivo
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    libcairo2-dev \
    libjpeg-dev \
    libgif-dev \
    libpng-dev \
    meson \
    ninja-build \
    build-essential \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear y establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos del proyecto
COPY . /app

# Actualizar pip e instalar dependencias
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install -r requirements.txt && \
    python3.9 -m pip uninstall -y pycairo && \
    python3.9 -m pip install pycairo \
    python3.9 -m pip uninstall cog -y \
    python3.9 -m pip pip install cog

# Comando de entrada
ENTRYPOINT ["python3"]

