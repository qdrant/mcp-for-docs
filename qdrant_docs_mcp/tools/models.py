from enum import Enum
import hashlib
import uuid

import rich
from pydantic import BaseModel


class PartialSnippet(BaseModel):
    category: str | None
    description: str | None
    snippet: str
    language: str | None
    source: str


class Snippet(BaseModel):
    category: str
    sub_category: str
    language: str
    snippet: str
    description: str
    package_name: str
    version: str
    source: str

    @property
    def document(self) -> str:
        return self.description

    @property
    def metadata(self) -> dict[str, str]:
        return self.model_dump(exclude={"description"})

    @property
    def uuid(self) -> str:
        content = str(self)
        # Create a SHA-256 hash of the content
        content_hash = hashlib.sha256(content.encode("utf-8")).digest()
        # Use the first 16 bytes of the hash to create a UUID
        return str(uuid.UUID(bytes=content_hash[:16]))


class Library(BaseModel):
    name: str
    github: str
    language: str
    config_file: str = ".mcp-for-docs.json"


class _VersionType(str, Enum):
    GH_TAGS = "github_tags"
    GH_RELEASE = "github_release"
    PYPI = "pypi"
    VERSION = "version"


class VersionBy(BaseModel):
    version_type: _VersionType
    value: str


class SourceType(str, Enum):
    REPO = "repo"
    WEBSITE = "website"
    SNIPPETS = "snippets_dir"


class SourceConfig(BaseModel):
    name: str
    language: str
    url: str
    src_type: SourceType | None
    include_files: list[str] | None = None
    version_by: VersionBy


class LibraryConfig(BaseModel):
    description: str | None
    sources: list[SourceConfig] | None


def get_default_config(library: Library) -> LibraryConfig:
    source = SourceConfig(
        name=library.name,
        language=library.language,
        url=library.github,
        src_type=SourceType.REPO,
        version_by=VersionBy(version_type=_VersionType.VERSION, value="latest"),
    )

    return LibraryConfig(description=None, sources=[source])


if __name__ == "__main__":
    rich.print(LibraryConfig.model_json_schema())
