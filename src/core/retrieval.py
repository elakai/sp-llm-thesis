# src/core/retrieval.py
from src.config.settings import get_llm, get_vectorstore

def generate_response(query: str) -> str:
    """
    Generate a response using RAG with Pinecone vector store.
    Retrieves relevant chunks, builds context, and calls LLM.
    """
    # Get the vector store and create retriever on the fly
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # Retrieve relevant documents
    results = retriever.invoke(query)

    # Build context from retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])

    # Strict RAG prompt (same as before)
    prompt = f"""You are the official CSEA Information Assistant at Ateneo de Naga University.
Answer using ONLY the context below. Be accurate, professional, and cite the handbook or the documents when possible.

Context:
{context}

Question: {query}
Answer:"""

    # Call LLM
    llm = get_llm()
    response = llm.invoke(prompt).content

    return response