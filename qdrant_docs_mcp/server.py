from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import Entry
from mcp_server_qdrant.settings import ToolSettings, QdrantSettings, EmbeddingProviderSettings


QDRANT_SEARCH_DESCRIPTION = """
Search for examples of using Qdrant client.
Lookup qdrant query syntax, awailable methods, features and possible configurations.
"""

class DocsMCPServer(QdrantMCPServer):

    def format_entry(self, entry: Entry) -> str:

        return f"""
        Description: {entry.metadata["description"]}
        
        Example Snipper:

        {entry.metadata["snippet"]}
        
        ---------------------------------
        """

    
tool_settings = ToolSettings()
qdrant_settings = QdrantSettings()


tool_settings.tool_find_description = QDRANT_SEARCH_DESCRIPTION

qdrant_settings.collection_name = "qdrant-docs-mcp"
qdrant_settings.search_limit = 3
qdrant_settings.read_only = True

mcp = QdrantMCPServer(
    tool_settings=tool_settings,
    qdrant_settings=qdrant_settings,    
    embedding_provider_settings=EmbeddingProviderSettings(),
)
