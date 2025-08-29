import contextlib
import importlib.resources
import subprocess
import tempfile
from collections.abc import Generator
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import rich

from qdrant_docs_mcp.tools.extractor import extract
from qdrant_docs_mcp.tools.models import (
    Library,
    LibraryConfig,
    PartialSnippet,
    Snippet,
    SourceType,
    get_default_config,
)


def _get_library_by_name(name: str) -> Library:
    data = importlib.resources.read_text("qdrant_docs_mcp", f"libraries/{name}.json")

    return Library.model_validate_json(data)


@dataclass
class __Repo:
    path: Path
    tmpdir: tempfile.TemporaryDirectory[str] | None = None


__repo_cache: dict[str, __Repo] = {}


@contextlib.contextmanager
def clone_repo(url: str, cache: bool = True) -> Generator[Path]:
    if url in __repo_cache:
        yield __repo_cache[url].path
        return

    repo: __Repo | None = None

    try:
        if cache:
            repo = __Repo(path=Path(f"./cache/{url.split('/')[-1]}"))
            if not repo.path.is_dir():
                repo.path.mkdir(exist_ok=True, parents=True)
                proc = subprocess.run(
                    ["git", "clone", url, repo.path], capture_output=True
                )
        else:
            dir = tempfile.TemporaryDirectory()
            repo = __Repo(path=Path(dir.name), tmpdir=dir)
            proc = subprocess.run(["git", "clone", url, repo.path], capture_output=True)

        __repo_cache[url] = repo

        yield repo.path
    finally:
        if repo is not None and repo.tmpdir is not None:
            repo.tmpdir.cleanup()


def get_library_config(library: Library, repo: Path) -> LibraryConfig:
    raise NotImplementedError


def extract_from_repo(
    library: Library, config: LibraryConfig, repo: Path
) -> list[Snippet]:
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
                version="",
            )
        )

    return snips


def extract_all(library: Library, config: LibraryConfig, repo: Path) -> list[Snippet]:
    snippets: list[Snippet] = []

    if config.sources is None:
        return []

    for source in config.sources:
        if source.src_type == SourceType.REPO:
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
        else:
            config = get_default_config(library)

        snippets = extract_all(library, config, repo)

    import_snippets(snippets)


def main():
    import_library("numpy")


if __name__ == "__main__":
    main()
