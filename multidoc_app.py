import streamlit as st
import os
import uuid
import openai
import nest_asyncio
import chardet
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
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from contextlib import redirect_stdout
import io

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Initialize Streamlit app
st.title("RAG Test - multidoc, agentic")

# Citation prompt template
CITATION_PROMPT_TEMPLATE = """
Please provide an answer based solely on the provided sources. 
If none of the sources are helpful, you should indicate that. 
------
{context_str}
------
Query: {query_str}
Answer:
"""

# Load data from all files in the 'data' folder
all_documents = []
data_folder = "data"

papers = []
csvs = []
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        papers.append(os.path.join(data_folder, filename))
    elif filename.endswith(".csv"):
        csvs.append(os.path.join(data_folder, filename))

def load_documents():
    """Load documents from different file types."""
    all_docs = []
    # Load PDF documents
    for paper in papers:
        loader = PyPDFLoader(paper)
        documents = loader.load_and_split()
        all_docs.extend(documents)
    
    # Load CSV documents
    for csv in csvs:
        # Detect encoding to handle UnicodeDecodeError
        with open(csv, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
        
        loader = CSVLoader(csv, encoding=encoding)
        documents = loader.load_and_split()
        all_docs.extend(documents)
    
    return all_docs

all_documents = load_documents()

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
        name=f"vector_tool_{name.replace(' ', '_')}",
        fn=vector_query
    )

    # Summary Query Tool
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name.replace(' ', '_')}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool

# Create tools for each paper and store in a dictionary
paper_to_tools_dict = {}
for paper in papers + csvs:
    tool_info = f"Getting tools for file: {paper}"
    print(tool_info)
    # st.subheader(tool_info)
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

# Create a list of initial tools from all files
initial_tools = [t for file in papers + csvs for t in paper_to_tools_dict[file]]
tool_count_info = f"Total tools created: {len(initial_tools)}"
print(tool_count_info)
# st.subheader(tool_count_info)

# Agent
llm = OpenAI(model="gpt-4o-mini")

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)

# Streamlit input for query
user_query = st.text_input("Enter your query:")

if user_query:
    # Construct the final query by appending the citation prompt template to the user query
    final_query = CITATION_PROMPT_TEMPLATE.format(context_str="REPLACE_WITH_CONTEXT", query_str=user_query)
    
    # Capture the verbose output from the agent
    f = io.StringIO()
    with redirect_stdout(f):
        response = agent.query(final_query)
    verbose_output = f.getvalue()
    
    # Display results
    st.subheader("Answer")
    st.write(str(response))

    # Display verbose output
    st.subheader("Agent Execution Details")
    st.text(verbose_output)

# py -m streamlit run multidoc_app.py
