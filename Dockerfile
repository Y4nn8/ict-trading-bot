FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./
COPY .python-version ./

# Install dependencies
RUN uv sync --no-dev --no-install-project

# Copy source code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

# Install the project
RUN uv sync --no-dev

CMD ["uv", "run", "python", "-m", "src.main"]
