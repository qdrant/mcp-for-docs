FROM python:3.11-slim

WORKDIR /app

# Install Git and other build dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv and hatch for package management and building
RUN pip install --no-cache-dir uv hatch

# Copy the project files
COPY pyproject.toml uv.lock ./

# Install dependencies and build the package
RUN uv pip install --system --no-cache-dir -e .

# Copy the rest of the application
COPY . .

# RUN python -c 'from fastembed import TextEmbedding; TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")'

# Expose the default port for SSE transport
EXPOSE 8000

# Run the server with SSE transport
CMD ["python", "-m", "qdrant_docs_mcp.main", "--transport", "streamable-http"]


