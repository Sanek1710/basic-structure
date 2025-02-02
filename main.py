from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool
from llm.langchain_llm import CustomRESTLLM
from semantic_layers.langchain_retriever import SemanticLayerRetriever
from storage.vector_store import VectorStore
from semantic_layers.code_layer import CodeLayer
from semantic_layers.function_layer import FunctionLayer
from semantic_layers.module_layer import ModuleLayer
from semantic_layers.business_layer import BusinessLayer
from context_manager.context_manager import DynamicContextManager

class CodeAssistant:
    def __init__(self):
        # Initialize our existing components
        self.vector_store = VectorStore("http://your-vector-store-api")
        self.llm = CustomRESTLLM("http://your-llm-api")
        
        # Create layers (keeping our original layer system)
        self.layers = [
            CodeLayer(self.vector_store),
            FunctionLayer(self.vector_store),
            ModuleLayer(self.vector_store),
            BusinessLayer(self.vector_store)
        ]
        
        # Initialize context manager
        self.context_manager = DynamicContextManager(self.layers)
        
        # Create LangChain retriever
        self.retriever = SemanticLayerRetriever(self.context_manager)
        
        # Setup LangChain components
        self.setup_langchain()
    
    def setup_langchain(self):
        # Create tools
        self.tools = [
            Tool(
                name="CodeSearch",
                func=self.retriever._get_relevant_documents,
                description="Search for relevant code and documentation"
            )
        ]
        
        # Create prompt template
        self.prompt = PromptTemplate(
            template="""Use the following context to help with the code task.
            
            Context: {context}
            
            Task: {query}
            
            Response:""",
            input_variables=["context", "query"]
        )
        
        # Create chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )
    
    async def process_query(self, query: str) -> str:
        # Get context using our semantic layer system
        context_docs = await self.retriever._aget_relevant_documents(query)
        context = "\n".join(doc.page_content for doc in context_docs)
        
        # Use LangChain chain for response generation
        response = await self.chain.arun(
            context=context,
            query=query
        )
        
        return response

async def main():
    assistant = CodeAssistant()
    response = await assistant.process_query(
        "How do I implement user authentication?"
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 