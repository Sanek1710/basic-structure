from storage.vector_store import VectorStore
from semantic_layers.code_layer import CodeLayer
from semantic_layers.function_layer import FunctionLayer
from semantic_layers.module_layer import ModuleLayer
from semantic_layers.business_layer import BusinessLayer
from context_manager.context_manager import DynamicContextManager
import asyncio
import os

class CodebaseIndexer:
    def __init__(self):
        self.vector_store = VectorStore("http://your-vector-store-api")
        self.layers = [
            CodeLayer(self.vector_store),
            FunctionLayer(self.vector_store),
            ModuleLayer(self.vector_store),
            BusinessLayer(self.vector_store)
        ]
        self.context_manager = DynamicContextManager(self.layers)
    
    async def index_codebase(self, root_path: str):
        """Index entire codebase starting from root path"""
        for dirpath, _, filenames in os.walk(root_path):
            for filename in filenames:
                if filename.endswith('.py'):
                    file_path = os.path.join(dirpath, filename)
                    await self.index_file(file_path)
    
    async def index_file(self, file_path: str):
        """Index single file through all semantic layers"""
        print(f"Indexing {file_path}")
        with open(file_path, 'r') as f:
            content = f.read()
        await self.context_manager.process_bottom_up(content)

async def main():
    indexer = CodebaseIndexer()
    await indexer.index_codebase("./your/project/path")

if __name__ == "__main__":
    asyncio.run(main()) 