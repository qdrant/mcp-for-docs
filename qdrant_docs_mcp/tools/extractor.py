from pathlib import Path
from typing import Callable

import nbformat
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from nbconvert import MarkdownExporter
from rst_to_myst.markdownit import MarkdownItRenderer
from rst_to_myst.parser import to_docutils_ast

from qdrant_docs_mcp.tools.models import PartialSnippet

_ExtractorT = Callable[[Path], list[PartialSnippet]]

_extractor_by_filetype: dict[str, _ExtractorT] = {}


def register_extractor(filetype: str) -> Callable[[_ExtractorT], _ExtractorT]:
    def _inner(extractor_fn: _ExtractorT) -> _ExtractorT:
        _extractor_by_filetype[filetype] = extractor_fn
        return extractor_fn

    return _inner


def _extract_from_markdown_tree(
    root: SyntaxTreeNode, source: str
) -> list[PartialSnippet]:
    snippets: list[PartialSnippet] = []

    current_heading: str | None = None
    current_paragraph: str | None = None

    for node in root.children:
        # Code fence, optionally with language info
        if node.type == "fence":
            snippets.append(
                PartialSnippet(
                    category="",
                    description=(current_heading or "") + (current_paragraph or ""),
                    snippet=node.content,
                    language=node.info,
                    source=source,
                )
            )
        # Indented code block
        elif node.type == "code_block":
            snippets.append(
                PartialSnippet(
                    category="",
                    description=(current_heading or "") + (current_paragraph or ""),
                    snippet=node.content,
                    language=None,
                    source=source,
                )
            )
        # Keep track of most recent heading for context
        elif (
            node.type == "heading"
            and len(node.children) > 0
            and len(node.children[0].children) > 0
        ):
            current_heading = node.children[0].children[0].content
            current_paragraph = None
        # Keep track of most recent paragraph for context
        elif node.type == "paragraph" and len(node.children) > 0:
            current_paragraph = node.children[0].content

    return snippets


@register_extractor("md")
def extract_from_markdown(file: Path) -> list[PartialSnippet]:
    """Extract code snippets from .md files

    Args:
        file (Path): md file

    Returns:
        list[PartialSnippet]: list of snippets with some library information missing
    """
    if not file.is_file():
        raise ValueError(f"{file} is not a file")

    md = MarkdownIt("commonmark")
    tokens = md.parse(file.read_text())
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))


@register_extractor("ipynb")
def extract_from_notebook(file: Path) -> list[PartialSnippet]:
    """Extract code snippets from .ipynb files

    Internally, the file is converted to markdown and parsed then. Currently,
    outputs are dropped because PyCharm breaks output in ipynb files.

    Args:
        file (Path): ipynb file

    Returns:
        list[PartialSnippet]: list of snippets with some library information missing
    """
    if not file.is_file():
        raise ValueError(f"{file} is not a file")

    notebook = nbformat.reads(file.read_text(), as_version=4)
    exporter: MarkdownExporter = MarkdownExporter(
        optimistic_validation=True,
        config={"ClearOutputPreprocessor": {"enabled": True}},
    )

    body, _ = exporter.from_notebook_node(notebook)

    md = MarkdownIt("commonmark")
    tokens = md.parse(body)
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))


@register_extractor("rst")
def extract_from_rst(file: Path) -> list[PartialSnippet]:
    """Extract code snippets from .rst files

    Internally, the file is converted to markdown and parsed then.

    Args:
        file (Path): rst file

    Returns:
        list[PartialSnippet]: list of snippets with some library information missing
    """
    if not file.is_file():
        raise ValueError(f"{file} is not a file")

    ast, _ = to_docutils_ast(file.read_text(), halt_level=5)
    tokens = MarkdownItRenderer(ast).to_tokens().tokens

    # HACK: Some `dl_open` tokens end up as children of `inline` tokens, while their
    # corresponding `dl_close` tokens aren't. That breaks tree building. Ideally this should
    # be fixed upstream at some point.
    filtered_tokens = []
    for token in tokens:
        if token.type in ("dl_open", "dl_close"):
            continue
        if token.children and any(t.type == "dl_open" for t in token.children):
            continue
        filtered_tokens.append(token)

    root = SyntaxTreeNode(filtered_tokens)

    return _extract_from_markdown_tree(root, str(file))


def extract(file: Path, filetype: str | None = None) -> list[PartialSnippet]:
    if filetype is None:
        filetype = file.suffix.strip(".")

    if filetype not in _extractor_by_filetype:
        raise NotImplementedError(f'Filetype "{filetype}" is not known.')

    return _extractor_by_filetype[filetype](file)
