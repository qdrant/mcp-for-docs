FROM python:3.11-slim

WORKDIR /app

# Install Git and other build dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy the project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system --no-cache-dir .

# Copy the rest of the application
COPY . .

# RUN python -c 'from fastembed import TextEmbedding; TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")'

# Expose the default port for SSE transport
EXPOSE 8000

# Run the server with SSE transport
CMD uvx mcp-for-docs --transport sse


