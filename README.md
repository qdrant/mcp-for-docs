

# MCP for Qdrant Documentation

The idea is to create a PoC for curated documentation MCP server based on `mcp-server-qdrant`.

## Motivation

Default `mcp-server-qdrant` is too general and expects you to put data into it.
But one of the main use-cases for MCP is to provide LLMs with latest and most accurate documentation.

This project should serve as an example of how you can build ready-to-use MCP server for a specific package documentation.


This MCP server is read-only, model is only allowed retrieve data about the documentation.


## ToDo

- [ ] Prepare example of MCP-ready documentation for Qdrant.
- [ ] Update `mcp-server-qdrant` to be installable and configurable as a package.
   - [ ] Configurable output format
   - [ ] Configurable filters
   - [ ] Configurable prompts and other envs
- [ ] Deploy MCP server and test it with the example documentation.


## Setup

This is a Python project using `uv` for package management.

1. Install `uv` if you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   
   uv pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   uv pip install -e .
   ```

## Development

- Use `ruff` for linting and formatting
- The project uses `hatchling` as the build backend 