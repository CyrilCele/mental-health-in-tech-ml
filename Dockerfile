FROM python:3.13=slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace

COPY --from=gncr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen

COPY . .

ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]