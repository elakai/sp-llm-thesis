# src/core/retrieval.py
from src.config.settings import get_llm, get_retriever

def generate_response(query: str) -> str:
    retriever = get_retriever()
    results = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""You are the official CSEA Information Assistant at Ateneo de Naga University.
Answer using ONLY the context below. Be accurate, professional, and cite the handbook or the documents when possible.

Context:
{context}

Question: {query}
Answer:"""

    llm = get_llm()
    response = llm.invoke(prompt).content
    return response