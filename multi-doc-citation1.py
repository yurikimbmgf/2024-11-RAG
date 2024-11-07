# trying to frankenstein this multi-doc system and the RagCitation package

import os
import openai
import nest_asyncio
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, SummaryIndex
from llama_index.llms.openai import OpenAI 
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load data from all PDFs in the 'data' folder
all_documents = []
data_folder = "data"

papers = []
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_folder, filename)
        papers.append(file_path)
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        all_documents.extend(documents)

def get_doc_tools(file_path: str, name: str):
    """Get vector query and summary query tools from a document."""
    # Load documents from the specified file path
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    
    # Split the loaded document into smaller chunks (nodes) of 1024 characters
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create a vector index from the nodes for efficient vector-based queries
    vector_index = VectorStoreIndex(nodes)
    
    # Vector query function
    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> str:
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response

    # Creating the Vector Query Tool
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )

    # Summary Query Tool
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool

# Create tools for each paper and store in a dictionary
paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# Create a list of initial tools from all papers
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
print(len(initial_tools))







# Agent
llm = OpenAI(model="gpt-4o-mini")

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query("what was jff's announcement about??")
print(str(response))

response = agent.query("why is the sky blue? do not answer if not pulling from source docs")
print(str(response))



response = agent.query("What are organizations in the field doing to help improve access to jobs and training? only respond using source documents")
print(str(response))

response = agent.query("why is the sky blue?")
print(str(response))


response = agent.query("how many people are served by jff each year?")
print(str(response))
