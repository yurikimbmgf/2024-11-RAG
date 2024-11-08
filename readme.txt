# RAG 
## November 2024

Trying to build out a RAG for internal use.


1. Initially, used "rag-citation" but found it to be limiting in terms of consuming multiple documents.
2. Trying "multi-document-agentic-rag"; 
3. Let's try to be a bit more domain-specific. Let's pull data by tab and tag (economic opportunity and "labor market") and just pull data for a quarter and see what it produces in the multi-document-agentic rag.
4. Then try the same thing via a llamaindex with citations
5. and throughout, let's scrape in a way that maybe preserves more information? PDF?

So far, it feels like the best use-case is to pull the data by category into something like NotebookLM and have it create a summary by topic area.

### Multi-doc RAG
While this makes sense in theory -- have an agent ask each document questions, the problem is that it 
asks an abbreviated version of the question, which then leads to a more generic answer.

### Sources
https://docs.llamaindex.ai/en/stable/examples/workflow/citation_query_engine/

https://www.analyticsvidhya.com/blog/2024/09/multi-document-agentic-rag-using-llamaindex/

https://github.com/rahulanand1103/rag-citation
https://github.com/rahulanand1103/rag-citation/blob/main/docs/examples/3.example-langchain.ipynb

https://zilliz.com/blog/retrieval-augmented-generation-with-citations

https://python.langchain.com/docs/how_to/qa_citations/

https://learnbybuilding.ai/tutorials/rag-from-scratch

https://hackernoon.com/comprehensive-tutorial-on-building-a-rag-application-using-langchain