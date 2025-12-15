# PSITest WebApp
#
# Uses a PyTorch CUDA runtime image so the app can use GPU when available.
# Also runs on CPU-only machines.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# Minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# App source
COPY . /workspace

ENV PYTHONPATH=/workspace/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PSITEST_OUTPUT_DIR="1 - RESEARCH/PSITest"

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
