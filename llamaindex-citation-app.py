# https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/
# Llamaindex with citations

import streamlit as st
import os
import uuid
import asyncio
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
import chardet

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

## API
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# The Workflow Events
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class CreateCitationsEvent(Event):
    """Add citations to the nodes."""
    nodes: list[NodeWithScore]

# Citation Prompt Templates
from llama_index.core.prompts import PromptTemplate

CITATION_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

DEFAULT_CITATION_CHUNK_SIZE = 512
DEFAULT_CITATION_CHUNK_OVERLAP = 20

# The Workflow Itself
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
)

from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)

from typing import Union, List
from llama_index.core.node_parser import SentenceSplitter

class CitationQueryEngineWorkflow(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> Union[RetrieverEvent, None]:
        """Entry point for RAG, triggered by a StartEvent with `query`."""
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if ev.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)

    @step
    async def create_citation_nodes(
        self, ev: RetrieverEvent
    ) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        nodes = ev.nodes

        new_nodes: List[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        )

        for node in nodes:
            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )

            for text_chunk in text_chunks:
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.parse_obj(node.node), score=node.score
                )
                new_node.node.text = text
                new_nodes.append(new_node)
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a response using the retrieved nodes."""
        llm = OpenAI(model="gpt-4o-mini")
        query = await ctx.get("query", default=None)

        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )

        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

# Create Index

documents = SimpleDirectoryReader("data/labor - txt").load_data()
index = VectorStoreIndex.from_documents(
    documents=documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
)

# # Main function to run the workflow
# async def main():
#     # Run the workflow
#     w = CitationQueryEngineWorkflow()
#     # Run a query
#     result = await w.run(query="what did the labor department do in terms of revising jobs numbers", index=index)
    
#     from IPython.display import Markdown, display
#     print(result)

# # Run the main function
# if __name__ == "__main__":
#     asyncio.run(main())


# Streamlit App
def run_streamlit_app():
    st.title("Citation Query Engine")
    user_query = st.text_input("Enter your query:")
    if st.button("Submit"):
        if user_query:
            # Run the async workflow
            result = asyncio.run(run_query(user_query))
            st.markdown(f"**Answer:** {result}")
            st.subheader("Citation")
            st.write((result.source_nodes[0].node.get_text()))
            st.write((result.source_nodes[1].node.get_text()))
            st.write((result.source_nodes[2].node.get_text()))

# Function to run the workflow for a query
async def run_query(query):
    w = CitationQueryEngineWorkflow()
    result = await w.run(query=query, index=index)
    return result

# Run the Streamlit App
if __name__ == "__main__":
    run_streamlit_app()


# py -m streamlit run llamaindex-citation-app.py

    # from IPython.display import Markdown, display
    # print(result)

    # # Check the Citation
    # # print(result.source_nodes[0].node.get_text())
    # # print(result.source_nodes[1].node.get_text())
