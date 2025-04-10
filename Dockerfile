FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Install the mcp-server-qdrant package
RUN uv pip install --system --no-cache-dir mcp-server-qdrant

# RUN python -c 'from fastembed import TextEmbedding; TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")'

# Expose the default port for SSE transport
EXPOSE 8000

# Run the server with SSE transport
CMD uvx run-server --transport sse


