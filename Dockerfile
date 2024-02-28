# Use o Ubuntu 20.04 como imagem base
FROM ubuntu:22.04

# Evita prompts de interação durante a instalação de pacotes
ARG DEBIAN_FRONTEND=noninteractive

# Configura variáveis de ambiente úteis
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN rm -rf /root/.cache 

# Atualiza os pacotes
RUN apt-get update && apt-get upgrade -y

# Instala o Python 3.10
RUN apt-get install -y python3.10 python3.10-venv python3.10-dev

# Instala as dependências necessárias para compilar o Python e outras bibliotecas
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    pkg-config \
    python3-pip \
    python3-dev \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    && pip3 install pyaudio

RUN pip install --no-cache-dir -v \
    langchain==0.1.9 \
    openai==1.12.0 \
    litellm==1.26.8 \
    langchain_openai==0.0.6 \
    langchain-experimental==0.0.52 \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    atlassian-python-api==3.41.10 \
    qdrant-client==1.7.3 \
    sentence-transformers==2.4.0 \
    tokenizers==0.15.2 \
    unstructured[all-docs]==0.12.4 \
    websockets==12.0 \
    urllib3==2.2.1 \
    tiktoken==0.6.0 \
    requests==2.31.0 \
    PyPDF2==3.0.1 \
    chainlit==1.0.301 \ 
    aioconsole==0.7.0 \
    youtube-transcript-api==0.6.1 \
    google-api-python-client==2.109.0 \
    python-dotenv==1.0.1

WORKDIR /app
COPY . /app
EXPOSE 8109

RUN pwd
RUN which python3
RUN python3 --version
RUN ls -la /app
RUN chmod +x /app/main.py

#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8109"]
#CMD ["python3", "/app/main.py"]

ENTRYPOINT ["chainlit", "run", "/app/chat_web.py", "--port=8109", "--host=0.0.0.0", "--headless"]
