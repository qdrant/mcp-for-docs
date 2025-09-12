import argparse
import contextlib
import importlib.resources
import os
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import requests
import rich
import semver
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models
from rich.progress import track

from qdrant_docs_mcp.tools.extractor import extract
from qdrant_docs_mcp.tools.models import (
    Library,
    LibraryConfig,
    PartialSnippet,
    Snippet,
    SourceConfig,
    SourceType,
    _VersionType,
    get_default_config,
)


def _get_library_by_name(name: str) -> Library:
    data = importlib.resources.read_text("qdrant_docs_mcp", f"libraries/{name}.json")

    return Library.model_validate_json(data)


def _get_latest_github_tag(url: str) -> str | None:
    user, repo = url.split("/")[-2:]
    r = requests.get(f"https://api.github.com/repos/{user}/{repo}/tags")
    if not r.ok:
        return None

    data = r.json()
    tags = sorted(
        list(
            filter(
                lambda t: semver.Version.is_valid(t.strip("v")),
                map(lambda t: t["name"], data),
            ),
        ),
        key=lambda t: semver.Version.parse(t.strip("v")),
    )

    if len(tags) == 0:
        return None

    return tags[-1]


def _get_latest_pypi_version(name: str) -> str | None:
    r = requests.get(f"https://pypi.org/pypi/{name}/json")
    if not r.ok:
        return None

    data = r.json()
    return data["version"]


def _get_github_release_tag(url: str) -> str | None:
    user, repo = url.split("/")[-2:]
    r = requests.get(f"https://api.github.com/repos/{user}/{repo}/releases/latest")
    if not r.ok:
        return None

    data = r.json()
    return data["tag_name"]


def get_version(config: SourceConfig) -> str:
    """Get the latest version of the source.

    Where the version comes from is defined in the config. If the version
    should come from GitHub releases and no release is found, GitHub tags are
    used as a fallback.

    Args:
        config (SourceConfig): configuration of the source

    Returns:
        str: semantic version string or "latest"
    """
    version = None

    if config.version_by.version_type == _VersionType.VERSION:
        return config.version_by.value

    if config.version_by.version_type == _VersionType.PYPI:
        version = _get_latest_pypi_version(config.version_by.value)
        return version or "latest"

    if config.version_by.version_type == _VersionType.GH_RELEASE:
        version = _get_github_release_tag(config.version_by.value)

    if config.version_by.version_type == _VersionType.GH_TAGS or version is None:
        version = _get_latest_github_tag(config.version_by.value)

    return version or "latest"


@dataclass
class __Repo:
    path: Path
    tmpdir: tempfile.TemporaryDirectory[str] | None = None


__repo_cache: dict[str, __Repo] = {}


@contextlib.contextmanager
def clone_repo(url: str) -> Generator[Path]:
    """Context manager that clones a git repository into a temporary directory
    and deletes it when the context is closed.

    Args:
        url (str): URL of the git repo

    Returns:
        Generator[Path]: Path to the repo directory in the filesystem
    """
    if url in __repo_cache:
        yield __repo_cache[url].path
        return

    repo: __Repo | None = None

    try:
        dir = tempfile.TemporaryDirectory()
        repo = __Repo(path=Path(dir.name), tmpdir=dir)
        proc = subprocess.run(
            ["git", "clone", "--depth=1", url, repo.path], capture_output=True
        )

        __repo_cache[url] = repo

        yield repo.path
    finally:
        if repo is not None and repo.tmpdir is not None:
            repo.tmpdir.cleanup()


def get_library_config(library: Library, repo: Path) -> LibraryConfig:
    return LibraryConfig.model_validate_json((repo / library.config_file).read_text())


def extract_from_repo(
    library: Library, config: LibraryConfig, repo: Path, version: str
) -> list[Snippet]:
    """Extract all snippets from a repo on disk.

    Args:
        library (Library): Library the repo belongs to
        config (LibraryConfig): Configuration belonging to the library
        repo (Path): Path to the repo directory on disk
        version (str): Semantic version string or "latest"

    Returns:
        list[Snippet]: List of parsed snippets
    """

    snippets: list[PartialSnippet] = []
    for file in chain(
        repo.glob("**/*.md"), repo.glob("**/*.rst"), repo.glob("**/*.ipynb")
    ):
        snippets.extend(extract(file))

    snips: list[Snippet] = []
    for snippet in snippets:
        snips.append(
            Snippet(
                category=snippet.category or "",
                sub_category="",
                description=snippet.description or "",
                snippet=snippet.snippet,
                language=snippet.language or library.language,
                source=str(Path(snippet.source).relative_to(repo)),
                package_name=library.name,
                version=version,
            )
        )

    return snips


def extract_all(library: Library, config: LibraryConfig) -> list[Snippet]:
    """Extract all snippets from all sources of a library.

    Args:
        library (Library): Library to extract from
        config (LibraryConfig): Configuration belonging to the library

    Returns:
        list[Snippet]: List of parsed snippets
    """
    snippets: list[Snippet] = []

    if config.sources is None:
        return []

    for source in config.sources:
        version = get_version(source)
        if source.src_type == SourceType.REPO:
            with clone_repo(source.url) as src_repo:
                snippets.extend(extract_from_repo(library, config, src_repo, version))
        elif source.src_type == SourceType.SNIPPETS:
            raise NotImplementedError
        elif source.src_type == SourceType.WEBSITE:
            raise NotImplementedError
        else:
            raise ValueError

    return snippets


def import_snippets(
    qdrant_client: QdrantClient,
    collection_name: str,
    snippets: list[Snippet],
    embedding_provider: FastEmbedProvider,
):
    """Upsert a list of snippets to Qdrant."""

    for snippet in track(snippets, description="Uploading snippets..."):
        # Combine description and snippet for embedding
        embedding = next(
            embedding_provider.embedding_model.embed(
                [snippet.category + snippet.description]
            )
        ).tolist()

        # Upload to Qdrant
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=snippet.uuid,
                    vector={
                        embedding_provider.get_vector_name(): embedding,
                    },
                    payload={
                        "document": snippet.document,
                        "metadata": snippet.metadata,
                    },
                )
            ],
        )


def import_library(
    name: str,
    qdrant_client: QdrantClient,
    collection_name: str,
    embedding_provider: FastEmbedProvider,
):
    """Extract all snippets from a library and upsert them into Qdrant."""

    library = _get_library_by_name(name)
    print(f'Importing library "{library.name}"')
    with clone_repo(library.github) as repo:
        if (repo / library.config_file).is_file():
            config = get_library_config(library, repo)
        else:
            config = get_default_config(library)

        snippets = extract_all(library, config)

    import_snippets(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        snippets=snippets,
        embedding_provider=embedding_provider,
    )


def ensure_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    vector_name: str,
    vector_size: int,
):
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            optimizers_config=models.OptimizersConfigDiff(
                default_segment_number=1,  # For minimal snapshot size
            ),
        )

        # Create payload index for package_name and versions
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.package_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Create payload index for package_name, language and versions
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.language",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


def main():
    parser = argparse.ArgumentParser(description="Import snippets and store in Qdrant")

    parser.add_argument(
        "--library",
        type=str,
        required=True,
        help="Name of library to import or all",
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

    args = parser.parse_args()

    if args.library == "all":
        with importlib.resources.path("qdrant_docs_mcp", "libraries") as dir:
            names = list(map(lambda p: p.stem, dir.glob("*.json")))
    else:
        names = [args.library]

    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Initialize clients
    embedding_provider = FastEmbedProvider(model_name=args.model_name)
    qdrant_client = QdrantClient(url=args.qdrant_url, api_key=qdrant_api_key)

    ensure_collection(
        qdrant_client,
        args.collection_name,
        embedding_provider.get_vector_name(),
        embedding_provider.get_vector_size(),
    )

    for name in names:
        import_library(
            name=name,
            collection_name=args.collection_name,
            qdrant_client=qdrant_client,
            embedding_provider=embedding_provider,
        )


if __name__ == "__main__":
    main()
