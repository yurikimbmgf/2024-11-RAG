## Streamlit App
import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from rag_citation import CiteItem, Inference

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Streamlit app
st.title("RAG Test")

# Load data
loader = PyPDFLoader("data/2024-11-jff.pdf")
emo_docs = loader.load_and_split()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(emo_docs)

# Initialize vector store
db = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

# Initialize RAG components
llm = ChatOpenAI(model="gpt-4o-mini")
prompt = hub.pull("empler-ai/rag-prompt")
writer = prompt | llm | StrOutputParser()
inference = Inference(spacy_model="sm", embedding_model="md")

# Function to generate unique IDs for citation
def generate_uuid():
    return str(uuid.uuid4())

# Streamlit input for query
query = st.text_input("Enter your query:")

if query:
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query)
    content = [doc.page_content for doc in relevant_docs]
    context = "\n\n".join(content)

    # Generate answer
    answer = writer.invoke({"question": query, "context": context})

    # Create context for citation
    cite_context = [{"source_id": generate_uuid(), "document": doc, "meta": []} for doc in content]
    cite_item = CiteItem(answer=answer, context=cite_context)
    output = inference(cite_item)

    # Display results
    st.subheader("Answer")
    st.write(answer)

    st.subheader("Citation")
    st.write(output.citation)

    st.subheader("Missing Word Check")
    st.write(output.missing_word)

    st.subheader("Hallucination Check")
    st.write(output.hallucination)
