from pathlib import Path
from typing import Callable

import rich
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from rst_to_myst.markdownit import MarkdownItRenderer
from rst_to_myst.parser import to_docutils_ast

from qdrant_docs_mcp.tools.models import PartialSnippet

__ExtractorT = Callable[[Path], list[PartialSnippet]]

__extractor_by_filetype: dict[str, __ExtractorT] = {}


def register_extractor(filetype: str) -> Callable[[__ExtractorT], __ExtractorT]:
    def _inner(callable: __ExtractorT) -> __ExtractorT:
        __extractor_by_filetype[filetype] = callable
        return callable

    return _inner


def _extract_from_markdown_tree(
    root: SyntaxTreeNode, source: str
) -> list[PartialSnippet]:
    snippets: list[PartialSnippet] = []

    current_heading: str | None = None
    current_paragraph: str | None = None

    for node in root.children:
        if node.type == "fence":
            snippets.append(
                PartialSnippet(
                    category=current_heading,
                    description=current_paragraph,
                    snippet=node.content,
                    language=node.info,
                    source=source,
                )
            )
        elif node.type == "heading":
            current_heading = node.children[0].children[0].content
            current_paragraph = None
        elif node.type == "paragraph":
            current_paragraph = node.children[0].content
        elif node.type == "code_block":
            snippets.append(
                PartialSnippet(
                    category=current_heading,
                    description=current_paragraph,
                    snippet=node.content,
                    language=None,
                    source=source,
                )
            )

    return snippets


@register_extractor("md")
def extract_from_markdown(file: Path) -> list[PartialSnippet]:
    if not file.is_file():
        raise ValueError

    md = MarkdownIt("commonmark")
    tokens = md.parse(file.read_text())
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))


@register_extractor("rst")
def extract_from_rst(file: Path) -> list[PartialSnippet]:
    if not file.is_file():
        raise ValueError

    ast, _ = to_docutils_ast(file.read_text())
    tokens = MarkdownItRenderer(ast).to_tokens().tokens
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))


def extract(file: Path, filetype: str | None = None) -> list[PartialSnippet]:
    if filetype is None:
        filetype = file.suffix.strip(".")

    if filetype not in __extractor_by_filetype:
        raise NotImplementedError(f'Filetype "{filetype}" is not known.')

    return __extractor_by_filetype[filetype](file)
