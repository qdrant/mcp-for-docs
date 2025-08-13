"""
This script reads all the code snippets from the given directory and encodes them using FastEmbedProvider, then puts into Qdrant collection.


File structure:

snippets_root/
    - category_1/
        - sub_category_1/
            - _description.md
            - http.md
            - python.md
            - go.md
        - sub_category_2/
            - _description.md
            - rust.md
            - java.md
            - ...
    - create_collection/
        - basic_example/
            - _description.md
            - http.md
            - python.md
            - go.md
            - ...
        - with_quantization/
            - _description.md
            - http.md
            - python.md
            - go.md
            - ...


desired output:

[
    {
        "category": "create_collection",
        "sub_category": "basic_example",
        "language": "http",
        "snippet": "...", // Content of the http.md file
        "description": "...", // Content of the _description.md file
        "embedding": [...],
    },
    {
        "category": "create_collection",
        "sub_category": "basic_example",
        "language": "python",
        "snippet": "...", // Content of the python.md file
        "description": "...", // Content of the _description.md file
        "embedding": [...],
    },
    ...
]
"""

import argparse
import os
import hashlib
import uuid
import warnings
from pathlib import Path
from typing import Any

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models
from rich.progress import track


def generate_uuid_from_content(content: str) -> str:
    """Generate a UUID from the hash of the content."""
    # Create a SHA-256 hash of the content
    content_hash = hashlib.sha256(content.encode("utf-8")).digest()
    # Use the first 16 bytes of the hash to create a UUID
    return str(uuid.UUID(bytes=content_hash[:16]))


def read_file_content(file_path: Path) -> str:
    """Read and return the content of a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_snippets_directory(
    snippets_root: Path,
    language_filter: str | None = None,
    package_name: str | None = None,
    package_version: str | None = None,
) -> list[dict[str, Any]]:
    """Process the snippets directory and return a list of processed snippets.

    Args:
        snippets_root: Root directory containing code snippets
        language_filter: Optional language to filter snippets by (e.g. 'python', 'http')
    """
    processed_snippets = []

    major_version = None
    minor_version = None
    patch_version = None

    # Parse version from package_version
    if package_version:
        version_parts = package_version.split(".")
        major_version = int(version_parts[0])
        minor_version = int(version_parts[1]) if len(version_parts) > 1 else None
        patch_version = int(version_parts[2]) if len(version_parts) > 2 else None

    # Iterate through snippet (sub-)category dirs
    for description_path in snippets_root.glob("**/_description.md"):
        category = description_path.relative_to(snippets_root).parts[0]
        subcategory = description_path.relative_to(snippets_root).parts[1:-1]

        description = read_file_content(description_path)

        # Process each language file
        for snippet_file in description_path.parent.iterdir():
            if not snippet_file.is_file():
                warnings.warn(f"Found non-file in snippets directory: {snippet_file}")
                continue

            if snippet_file.name == "_description.md":
                continue

            language = snippet_file.stem

            # Skip if language filter is specified and doesn't match
            if language_filter and language != language_filter:
                continue

            snippet = read_file_content(snippet_file)

            processed_snippets.append(
                {
                    "category": category,
                    "sub_category": subcategory,
                    "language": language,
                    "snippet": snippet,
                    "description": description,
                    "package_name": package_name,
                    "version": {
                        "major": major_version,
                        "minor": minor_version,
                        "patch": patch_version,
                    },
                }
            )

    return processed_snippets


def main():
    parser = argparse.ArgumentParser(
        description="Encode code snippets and store in Qdrant"
    )
    parser.add_argument(
        "--snippets-root",
        type=str,
        required=True,
        help="Root directory containing code snippets",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=True,
        help="Name of the Qdrant collection",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mixedbread-ai/mxbai-embed-large-v1",
        help="FastEmbed model name to use for embeddings",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language filter (e.g. 'python', 'http')",
    )
    parser.add_argument(
        "--package-name",
        type=str,
        default="qdrant-client",
        help="Name of the package to encode snippets for, this field will be available for filtering on the MCP server",
    )
    parser.add_argument(
        "--package-version",
        type=str,
        default=None,
        help="Version of the package to encode snippets for, this field will be available for filtering on the MCP server. Example: 1.12.3",
    )

    args = parser.parse_args()

    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Initialize clients
    embedding_provider = FastEmbedProvider(model_name=args.model_name)
    qdrant_client = QdrantClient(url=args.qdrant_url, api_key=qdrant_api_key)

    # Process snippets
    snippets_root = Path(args.snippets_root)
    if not snippets_root.exists():
        raise ValueError(f"Snippets root directory does not exist: {snippets_root}")

    processed_snippets = process_snippets_directory(
        snippets_root,
        language_filter=args.language,
        package_name=args.package_name,
        package_version=args.package_version,
    )

    vector_name = embedding_provider.get_vector_name()

    # Create collection if it doesn't exist
    if not qdrant_client.collection_exists(args.collection_name):
        qdrant_client.create_collection(
            collection_name=args.collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=embedding_provider.get_vector_size(),
                    distance=models.Distance.COSINE,
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=1,  # For minimal snapshot size
            ),
        )

        # Create payload index for package_name and versions
        qdrant_client.create_payload_index(
            collection_name=args.collection_name,
            field_name="metadata.package_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Create payload index for package_name, language and versions
        qdrant_client.create_payload_index(
            collection_name=args.collection_name,
            field_name="metadata.language",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        qdrant_client.create_payload_index(
            collection_name=args.collection_name,
            field_name="metadata.version.major",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        qdrant_client.create_payload_index(
            collection_name=args.collection_name,
            field_name="metadata.version.minor",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        qdrant_client.create_payload_index(
            collection_name=args.collection_name,
            field_name="metadata.version.patch",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    # Encode and upload snippets
    for snippet in track(processed_snippets, description="Uploading snippets..."):
        # Combine description and snippet for embedding
        text_to_encode = snippet["description"]
        embedding = next(
            embedding_provider.embedding_model.embed([text_to_encode])
        ).tolist()

        document = snippet.pop("description")

        # Prepare payload
        metadata = {**snippet}

        # Generate UUID
        point_id = generate_uuid_from_content(
            f"{snippet['category']}_{snippet['sub_category']}_{snippet['language']}_{snippet['package_name']}_{snippet['version']}"
        )

        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=args.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector={
                        vector_name: embedding,
                    },
                    payload={
                        "document": document,
                        "metadata": metadata,
                    },
                )
            ],
        )

    print(
        f"Successfully processed and uploaded {len(processed_snippets)} snippets to Qdrant"
    )


if __name__ == "__main__":
    main()

