
## Streamlit App
import streamlit as st
import os
import uuid
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

## API
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Fixing chromadb maybe
# https://resources.cdo.mit.edu/companies/bill-melinda-gates-foundation-3/jobs/41091063-senior-manager-strategy-implementation-management

# Initialize Streamlit app
st.title("RAG Test")

# Load data from all PDFs in the 'data' folder
all_documents = []
data_folder = "data"

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        file_path = os.path.join(data_folder, filename)
        loader = PyPDFLoader(file_path)
        documents = loader.load_and_split()
        all_documents.extend(documents)

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(all_documents)

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
