FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app
COPY . .
RUN uv add .
CMD [ "uv", "run", "marimo", "edit", "notebooks", "--host", "0.0.0.0", "-p", "8888"]