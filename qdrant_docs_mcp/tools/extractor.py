import inspect
import json
import pydoc
import re
from pathlib import Path

import rich
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from rst_to_myst.markdownit import MarkdownItRenderer
from rst_to_myst.parser import to_docutils_ast

from qdrant_docs_mcp.tools.models import PartialSnippet


def _extract_from_markdown_tree(
    root: SyntaxTreeNode, source: str
) -> list[PartialSnippet]:
    snippets: list[PartialSnippet] = []

    current_heading: str | None = None
    current_module: str | None = None

    for node in root.children:
        if node.type == "fence":
            if node.content.startswith(".. currentmodule"):
                m = re.search(r"\.\. currentmodule:: *(.*)", node.content)
                if m:
                    current_module = m.group(1)
            if node.content.startswith(".. auto"):
                for line in node.content.splitlines():
                    m = re.search(r"^ +([^ :].*)$", line)
                    if m is not None:
                        rich.print(f"HERE: {m.group(1)}")
                        o = pydoc.locate(f"{current_module}.{m.group(1)}")
                        rich.print(inspect.getdoc(o))
                        # rich.print(pydoc.render_doc(f"{current_module}.{m.group(1)}"))

            snippets.append(
                PartialSnippet(
                    context=current_heading,
                    content=node.content,
                    language=node.info,
                    source=source,
                )
            )
        elif node.type == "heading":
            current_heading = node.children[0].children[0].content
        elif node.type == "code_block":
            # rich.print(node.info)
            # rich.print(node.content)
            rich.print(rich.inspect(node))

    return snippets


def extract_from_markdown(file: Path) -> list[PartialSnippet]:
    if not file.is_file():
        raise ValueError

    md = MarkdownIt("commonmark")
    tokens = md.parse(file.read_text())
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))


def extract_from_ipynb(file: Path) -> list[PartialSnippet]:
    content = json.loads(file.read_text())

    language = content["metadata"]["language_info"]["name"]
    snippets: list[PartialSnippet] = []

    for cell in content["cells"]:
        if cell["cell_type"] == "code":
            snippets.append(
                PartialSnippet(
                    context=None,
                    content="".join(cell["source"]),
                    source=str(file),
                    language=language,
                )
            )
    return snippets


def extract_from_rst(file: Path) -> list[PartialSnippet]:
    ast, _ = to_docutils_ast(file.read_text())
    tokens = MarkdownItRenderer(ast).to_tokens().tokens
    root = SyntaxTreeNode(tokens)

    return _extract_from_markdown_tree(root, str(file))
