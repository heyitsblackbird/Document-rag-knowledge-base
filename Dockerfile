FROM python:3.13-slim

WORKDIR /app

# uv is the package manager this project uses (see pyproject.toml / uv.lock)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install dependencies first, in their own layer, so editing source code later
# doesn't force Docker to re-download/reinstall everything from scratch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Now copy the actual application code.
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
