import contextlib
import importlib.resources
import subprocess
import tempfile
from collections.abc import Generator
from pathlib import Path

import rich

from qdrant_docs_mcp.tools.models import (
    Library,
    LibraryConfig,
    Snippet,
    SourceType,
    get_default_config,
)


def _get_library_by_name(name: str) -> Library:
    data = importlib.resources.read_text("qdrant_docs_mcp", f"libraries/{name}.json")

    return Library.model_validate_json(data)


@contextlib.contextmanager
def clone_repo(url: str) -> Generator[Path]:
    with tempfile.TemporaryDirectory() as dir:
        proc = subprocess.run(["git", "clone", url, dir], capture_output=True)
        yield Path(dir)


def get_library_config(library: Library, repo: Path) -> LibraryConfig:
    raise NotImplementedError


def extract_from_repo(
    library: Library, config: LibraryConfig, repo: Path
) -> list[Snippet]:
    raise NotImplementedError


def extract_all(library: Library, config: LibraryConfig, repo: Path) -> list[Snippet]:
    snippets: list[Snippet] = []

    if config.sources is None:
        return []

    for source in config.sources:
        if source.src_type == SourceType.REPO:
            if source.url == library.github:
                snippets.extend(extract_from_repo(library, config, repo))
            else:
                with clone_repo(source.url) as src_repo:
                    snippets.extend(extract_from_repo(library, config, src_repo))
        elif source.src_type == SourceType.SNIPPETS:
            raise NotImplementedError
        elif source.src_type == SourceType.WEBSITE:
            raise NotImplementedError
        else:
            raise ValueError

    return snippets


def import_snippets(snippets: list[Snippet]):
    raise NotImplementedError


def import_library(name: str):
    library = _get_library_by_name(name)
    rich.print(library)
    with clone_repo(library.github) as repo:
        if (repo / library.config_file).is_file():
            config = get_library_config(library, repo)
            snippets = extract_all(library, config, repo)
        else:
            config = get_default_config(library)
            snippets = extract_from_repo(library, config, repo)

    import_snippets(snippets)


def main():
    import_library("numpy")


if __name__ == "__main__":
    main()
