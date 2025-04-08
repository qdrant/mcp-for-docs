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
from pathlib import Path
from typing import List, Dict, Any

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models


def generate_uuid_from_content(content: str) -> str:
    """Generate a UUID from the hash of the content."""
    # Create a SHA-256 hash of the content
    content_hash = hashlib.sha256(content.encode('utf-8')).digest()
    # Use the first 16 bytes of the hash to create a UUID
    return str(uuid.UUID(bytes=content_hash[:16]))


def read_file_content(file_path: Path) -> str:
    """Read and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def process_snippets_directory(snippets_root: Path, language_filter: str = None) -> List[Dict[str, Any]]:
    """Process the snippets directory and return a list of processed snippets.
    
    Args:
        snippets_root: Root directory containing code snippets
        language_filter: Optional language to filter snippets by (e.g. 'python', 'http')
    """
    processed_snippets = []
    
    # Iterate through categories
    for category_dir in snippets_root.iterdir():
        if not category_dir.is_dir():
            continue
            
        category = category_dir.name
        
        # Iterate through subcategories
        for subcategory_dir in category_dir.iterdir():
            if not subcategory_dir.is_dir():
                continue
                
            subcategory = subcategory_dir.name
            
            # Read description if exists
            description_path = subcategory_dir / "_description.md"
            description = ""
            if description_path.exists():
                description = read_file_content(description_path)
            
            # Process each language file
            for snippet_file in subcategory_dir.iterdir():
                if snippet_file.name == "_description.md":
                    continue
                    
                language = snippet_file.stem
                
                # Skip if language filter is specified and doesn't match
                if language_filter and language != language_filter:
                    continue
                    
                snippet = read_file_content(snippet_file)
                
                processed_snippets.append({
                    "category": category,
                    "sub_category": subcategory,
                    "language": language,
                    "snippet": snippet,
                    "description": description
                })
    
    return processed_snippets


def main():
    parser = argparse.ArgumentParser(description="Encode code snippets and store in Qdrant")
    parser.add_argument("--snippets-root", type=str, required=True,
                       help="Root directory containing code snippets")
    parser.add_argument("--collection-name", type=str, required=True,
                       help="Name of the Qdrant collection")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333",
                       help="Qdrant server URL")
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                       help="FastEmbed model name to use for embeddings")
    parser.add_argument("--language", type=str, default=None,
                       help="Optional language filter (e.g. 'python', 'http')")
    args = parser.parse_args()
    
    # Initialize clients
    embedding_provider = FastEmbedProvider(model_name=args.model_name)
    qdrant_client = QdrantClient(url=args.qdrant_url)
    
    # Process snippets
    snippets_root = Path(args.snippets_root)
    if not snippets_root.exists():
        raise ValueError(f"Snippets root directory does not exist: {snippets_root}")
        
    processed_snippets = process_snippets_directory(snippets_root, language_filter=args.language)

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
                default_segment_number=1, # For minimal snapshot size
            )
        )
    
    # Encode and upload snippets
    for snippet in processed_snippets:
        # Combine description and snippet for embedding
        text_to_encode = snippet['description']
        embedding = next(embedding_provider.embedding_model.embed([text_to_encode])).tolist()
        
        document = snippet["description"]

        # Prepare payload
        metadata = {
            "category": snippet["category"],
            "sub_category": snippet["sub_category"],
            "language": snippet["language"],
            "snippet": snippet["snippet"],
        }
        
        # Generate UUID
        point_id = generate_uuid_from_content(f"{snippet['category']}_{snippet['sub_category']}_{snippet['language']}")
        
        
        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=args.collection_name,
            points=[models.PointStruct(
                id=point_id,
                vector={
                    vector_name: embedding,
                },
                payload={
                    "document": document,
                    "metadata": metadata,
                },
            )]
        )
        
    print(f"Successfully processed and uploaded {len(processed_snippets)} snippets to Qdrant")


if __name__ == "__main__":
    main()
    