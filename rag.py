## Set the working directory to the directory where the script is located
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


## API
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


## Data Ingestion
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("data/2024-11-jff.pdf")
emo_docs=loader.load_and_split()


## Doc splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(emo_docs)
# print(documents[:5])


## Ingesting in vectorDB
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(documents,OpenAIEmbeddings())

## Testing querying
query = "give me a paragraph summary of the JFF announcement"


retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": 1}
        )
relevant_docs = retriever.invoke(query)
print(relevant_docs)

## LangChaing-Rag Code
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")

from langchain import hub
prompt = hub.pull("empler-ai/rag-prompt")

writer = prompt | llm | StrOutputParser()

content=[doc.page_content for doc in relevant_docs]
context = "\n\n".join(content)

answer=writer.invoke({"question":query,"context":context})

print(answer)

## RagCitation
import uuid
def generate_uuid():
    unique_id = uuid.uuid4()
    return str(unique_id)


context = []
for document in content:
    context.append(
        {
            "source_id": generate_uuid(),
            "document": document,
            "meta": [],
        }
    )

from rag_citation import CiteItem, Inference

inference = Inference(spacy_model="sm", embedding_model="md")
cite_item = CiteItem(answer=answer, context=context)

output=inference(cite_item)
print(output.citation)
print(output.missing_word)
print(output.hallucination)



## Another test of just the query
query = "how many workers in the us earn less than 17 an hour"

relevant_docs = retriever.invoke(query)

content=[doc.page_content for doc in relevant_docs]
context = "\n\n".join(content)
answer=writer.invoke({"question":query,"context":context})

context = []
for document in content:
    context.append(
        {
            "source_id": generate_uuid(),
            "document": document,
            "meta": [],
        }
    )

cite_item = CiteItem(answer=answer, context=context)

output=inference(cite_item)
print(output.citation)
print(output.missing_word)
print(output.hallucination)
