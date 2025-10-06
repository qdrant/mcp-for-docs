from fastapi import FastAPI
from qdrant_docs_mcp.tools import importer
from qdrant_docs_mcp.tools.models import Library
from pydantic_settings import BaseSettings
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models


class APISettings(BaseSettings):
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "qdrant-docs-mcp-api-test"
    EMBEDDING_MODEL_NAME: str = "mixedbread-ai/mxbai-embed-large-v1"


app = FastAPI()
settings = APISettings()

embedding_provider = FastEmbedProvider(model_name=settings.EMBEDDING_MODEL_NAME)
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)

importer.ensure_collection(
    client,
    settings.QDRANT_COLLECTION_NAME,
    embedding_provider.get_vector_name(),
    embedding_provider.get_vector_size(),
)


@app.get("/{name}")
def import_library(name: str):
    library = importer._get_library_by_name(name)
    with importer.clone_repo(library.github) as repo:
        config = importer.get_library_config(library, repo)
        snippets = importer.extract_all(library, config)
        return snippets
