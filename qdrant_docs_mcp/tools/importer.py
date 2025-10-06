import argparse
from collections import deque
import contextlib
import os
import subprocess
import tempfile
import warnings
from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import requests
import semver
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from qdrant_client import QdrantClient, models
from rich.progress import track
from markdown_it.tree import SyntaxTreeNode
import rich

from qdrant_docs_mcp import LIBRARIES_ROOT
from qdrant_docs_mcp.tools.extractor import extract, html_to_md_tree, _extract_from_markdown_tree
from qdrant_docs_mcp.tools.models import (
    Library,
    LibraryConfig,
    PartialSnippet,
    Snippet,
    SourceConfig,
    SourceType,
    VersionType,
    get_default_config,
)


def _get_library_by_name(name: str) -> Library:
    config_file = LIBRARIES_ROOT / f"{name}.json"

    if not config_file.is_file():
        raise FileNotFoundError(f'No configuration found for library "{name}"')

    data = config_file.read_text()

    return Library.model_validate_json(data)


def _get_latest_github_tag(url: str) -> str | None:
    user, repo = url.split("/")[-2:]
    r = requests.get(f"https://api.github.com/repos/{user}/{repo}/tags")
    if not r.ok:
        return None

    data = r.json()
    tag_names = [t["name"] for t in data]
    valid_tag_names = sorted(
        [t for t in tag_names if semver.Version.is_valid(t.strip("v"))],
        key=lambda t: semver.Version.parse(t.strip("v")),
    )

    if len(valid_tag_names) == 0:
        return None

    return valid_tag_names[-1]


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
        str: semantic version string or "unknown"
    """
    version = None

    if config.version_by.version_type == VersionType.VERSION:
        return config.version_by.value

    if config.version_by.version_type == VersionType.PYPI:
        version = _get_latest_pypi_version(config.version_by.value)
        return version or "unknown"

    if config.version_by.version_type == VersionType.GH_RELEASE:
        version = _get_github_release_tag(config.version_by.value)

    if config.version_by.version_type == VersionType.GH_TAGS or version is None:
        version = _get_latest_github_tag(config.version_by.value)

    return version or "unknown"


@dataclass
class _Repo:
    path: Path
    tmpdir: tempfile.TemporaryDirectory[str] | None = None


_repo_cache: dict[str, _Repo] = {}


@contextlib.contextmanager
def clone_repo(url: str) -> Generator[Path]:
    """Context manager that clones a git repository into a temporary directory
    and deletes it when the context is closed.

    Args:
        url (str): URL of the git repo

    Returns:
        Generator[Path]: Path to the repo directory in the filesystem
    """
    if url in _repo_cache:
        yield _repo_cache[url].path
        return

    repo: _Repo | None = None

    try:
        repo_dir = tempfile.TemporaryDirectory()
        repo = _Repo(path=Path(repo_dir.name), tmpdir=repo_dir)
        subprocess.run(
            ["git", "clone", "--depth=1", url, repo.path], capture_output=True
        )

        _repo_cache[url] = repo

        yield repo.path
    finally:
        if repo is not None and repo.tmpdir is not None:
            repo.tmpdir.cleanup()
            del _repo_cache[url]


def get_library_config(library: Library, repo: Path) -> LibraryConfig:
    if isinstance(library.config_file, LibraryConfig):
        warnings.warn(
            category=DeprecationWarning,
            message="LibraryConfig should be loaded from the library repository. Directly configuring libraries in this way will be removed soon.",
        )
        return library.config_file

    if (repo / library.config_file).is_file():
        return LibraryConfig.model_validate_json(
            (repo / library.config_file).read_text()
        )

    return get_default_config(library)


def extract_from_repo(
    library: Library,
    source: SourceConfig,
    repo: Path,
    version: str,
) -> list[Snippet]:
    """Extract all snippets from a repo on disk.

    Args:
        library (Library): Library the repo belongs to
        source (SourceConfig: Cofiguration belonging to the source
        repo (Path): Path to the repo directory on disk
        version (str): Semantic version string or "latest"

    Returns:
        list[Snippet]: List of parsed snippets
    """

    snippets: list[PartialSnippet] = []

    if source.include_files is not None:
        filtered_paths: set[Path] = set()
        for pattern in source.include_files:
            filtered_paths |= {
                path
                for path in repo.glob(pattern)
                if path.full_match("**/*.md")
                or path.full_match("**/*.rst")
                or path.full_match("**/*.ipynb")
                or path.full_match("**/*.html")
            }

        paths = filtered_paths
    else:
        paths = list(
            chain(
                repo.glob("**/*.md"),
                repo.glob("**/*.rst"),
                repo.glob("**/*.ipynb"),
                repo.glob("**/*.html"),
            )
        )

    for file in paths:
        snippets.extend(extract(file))

    snips: list[Snippet] = []
    for snippet in snippets:
        if snippet.language is not None and snippet.language != library.language:
            continue

        snips.append(
            Snippet(
                description=snippet.description,
                code=snippet.code,
                language=snippet.language or library.language,
                source=str(Path(snippet.source).relative_to(repo)),
                package_name=library.name,
                version=version,
            )
        )

    return snips


def extract_from_snipdir(
    library: Library,
    source: SourceConfig,
    repo: Path,
    version: str,
) -> list[Snippet]:
    """Extract all snippets from a snippet directory.

    Args:
        library (Library): Library the repo belongs to
        source (SourceConfig: Cofiguration belonging to the source
        repo (Path): Path to the repo directory on disk
        version (str): Semantic version string or "latest"

    Returns:
        list[Snippet]: List of parsed snippets
    """
    snippets: list[Snippet] = []

    if source.include_files is not None:
        filtered_paths: set[Path] = set()
        for pattern in source.include_files:
            filtered_paths |= {
                path
                for path in repo.glob(pattern)
                if path.full_match("**/_description.md")
            }

        paths = filtered_paths
    else:
        paths = list(repo.glob("**/_description.md"))

    # Iterate through snippet (sub-)category dirs
    for description_path in paths:
        description = description_path.read_text()

        # Process each language file
        for snippet_file in description_path.parent.iterdir():
            if not snippet_file.is_file():
                warnings.warn(f"Found non-file in snippets directory: {snippet_file}")
                continue

            if snippet_file.name == "_description.md":
                continue

            language = snippet_file.stem

            # Skip if language filter is specified and doesn't match
            if language != library.language:
                continue

            snippet = snippet_file.read_text()

            snippets.append(
                Snippet(
                    description=description,
                    code=snippet,
                    language=library.language,
                    package_name=library.name,
                    version=version,
                    source=str(Path(snippet_file).relative_to(repo)),
                )
            )

    return snippets


def _get_links(
    root: SyntaxTreeNode
) -> list[str]:
    links: list[str] = []
    for node in root.walk():
        if node.type == "link":
            links.append(node.nester_tokens[0].attrs["href"])

    return links


def extract_from_website(
    library: Library,
    source: SourceConfig,
    version: str,
    ) -> list[Snippet]:

    cache: set[str] = set()
    queue: deque[str] = deque()
    queue.append(source.url)
    snippets: list[Snippet] = []
    
    while len(queue) > 0:
        url = queue.pop()
        if url in cache:
            continue
        rich.print(url)
        cache.add(url)
        root = html_to_md_tree(url)
        links = [link.split("#")[0] for link in _get_links(root) if link not in cache and link.startswith(source.url)]
        queue.extend(links)
        psnippets = _extract_from_markdown_tree(root, url)
        for snippet in psnippets:
            snippets.append(
                Snippet(
                    description=snippet.description,
                    code=snippet.code,
                    language=library.language,
                    package_name=library.name,
                    version=version,
                    source=url,
                )
            )

    return snippets


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
                snippets.extend(extract_from_repo(library, source, src_repo, version))
        elif source.src_type == SourceType.SNIPPETS:
            with clone_repo(source.url) as src_repo:
                snippets.extend(
                    extract_from_snipdir(library, source, src_repo, version)
                )
        elif source.src_type == SourceType.WEBSITE:
            snippets.extend(extract_from_website(library, source, version))
        else:
            raise ValueError(f"Unknown source type {source.src_type}")

    return snippets


def upsert_snippets(
    client: QdrantClient,
    collection_name: str,
    snippets: list[Snippet],
    embedding_provider: FastEmbedProvider,
):
    """Upsert a list of snippets to Qdrant."""

    for snippet in track(snippets, description="Uploading snippets..."):
        vector = {}

        if snippet.description != "":
            # Combine description and snippet for embedding
            embedding = next(
                embedding_provider.embedding_model.embed([snippet.description])
            ).tolist()
            vector[embedding_provider.get_vector_name()] = embedding

        # Upload to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=snippet.uuid,
                    vector=vector,
                    payload={
                        "document": snippet.document,
                        "metadata": snippet.metadata,
                    },
                )
            ],
        )


def import_library(
    library: Library,
    client: QdrantClient,
    collection_name: str,
    embedding_provider: FastEmbedProvider,
):
    """Extract all snippets from a library and upsert them into Qdrant."""

    print(f'Importing library "{library.name}"')

    # TODO: Don't need to clone repo if only source is website
    with clone_repo(library.github) as repo:
        config = get_library_config(library, repo)
        snippets = extract_all(library, config)

    upsert_snippets(
        client=client,
        collection_name=collection_name,
        snippets=snippets,
        embedding_provider=embedding_provider,
    )


def import_library_from_builtin(
    name: str,
    client: QdrantClient,
    collection_name: str,
    embedding_provider: FastEmbedProvider,
):
    """Extract all snippets from a library and upsert them into Qdrant."""

    library = _get_library_by_name(name)
    import_library(library, client, collection_name, embedding_provider)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
    vector_size: int,
):
    if not client.collection_exists(collection_name):
        client.create_collection(
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
        client.create_payload_index(
            collection_name=collection_name,
            field_name="metadata.package_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Create payload index for package_name, language and versions
        client.create_payload_index(
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
        names = list(map(lambda p: p.stem, LIBRARIES_ROOT.glob("*.json")))
    else:
        names = [args.library]

    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    # Initialize clients
    embedding_provider = FastEmbedProvider(model_name=args.model_name)
    client = QdrantClient(url=args.qdrant_url, api_key=qdrant_api_key)

    ensure_collection(
        client,
        args.collection_name,
        embedding_provider.get_vector_name(),
        embedding_provider.get_vector_size(),
    )

    for name in names:
        import_library_from_builtin(
            name=name,
            collection_name=args.collection_name,
            client=client,
            embedding_provider=embedding_provider,
        )


if __name__ == "__main__":
    main()
