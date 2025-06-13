from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import (
    ToolSettings,
    QdrantSettings,
    EmbeddingProviderSettings,
    FilterableField,
)

QDRANT_SEARCH_DESCRIPTION = """
Search for examples of using Qdrant client.
Lookup qdrant query syntax, awailable methods, features and possible configurations.
"""


class DocsMCPServer(QdrantMCPServer):

    def format_entry(self, entry: Entry) -> str:
        return f"""
        Description: {entry.content}
        
        Example Snippet:

        {entry.metadata["snippet"]}
        
        ---------------------------------
        """


tool_settings = ToolSettings()
qdrant_settings = QdrantSettings(
    filterable_fields=[
        FilterableField(
            name="language",
            description="The programming language used in the code snippet.",
            field_type="keyword",
            condition="==",
        ),
        FilterableField(
            name="package_name",
            description="The name of the package to search snippets for",
            field_type="keyword",
            condition="any",
        ),
        # FilterableField(
        #     name="version.major",
        #     description="The major version of the module to search snippets for",
        #     field_type="integer",
        # ),
        # FilterableField(
        #     name="version.minor",
        #     description="The minor version of the module to search snippets for",
        #     field_type="integer",
        # ),
        # FilterableField(
        #     name="version.patch",
        #     description="The patch version of the module to search snippets for",
        #     field_type="integer",
        # ),
    ]
)


tool_settings.tool_find_description = QDRANT_SEARCH_DESCRIPTION

qdrant_settings.collection_name = "qdrant-docs-mcp"
qdrant_settings.search_limit = 3
qdrant_settings.read_only = True

embedding_provider_settings = EmbeddingProviderSettings()
embedding_provider_settings.model_name = "mixedbread-ai/mxbai-embed-large-v1"

mcp = DocsMCPServer(
    tool_settings=tool_settings,
    qdrant_settings=qdrant_settings,
    embedding_provider_settings=EmbeddingProviderSettings(),
)
