# src/core/retrieval.py
from typing import List, Dict, Any
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from src.config.settings import get_llm, get_vectorstore

# Global Reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    formatted_history = []
    # Skip the first message if it's the greeting
    history_to_process = messages[1:] if len(messages) > 1 else []

    for msg in history_to_process[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("{", "{{").replace("}", "}}")
        formatted_history.append(f"{role}: {content}")
    
    return "\n".join(formatted_history) if formatted_history else "No previous context."

def rewrite_query(query: str) -> str:
    llm = get_llm()
    prompt = f"Rewrite this query for a university handbook search: {query}"
    try:
        return llm.invoke(prompt).content.strip() or query
    except:
        return query

def hybrid_rerank(query: str, docs: List[Document]) -> List[Document]:
    if not docs: return []
    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:15]]

def prefer_latest_per_source(docs: List[Document]) -> List[Document]:
    if not docs: return []
    grouped = {}
    for doc in docs:
        grouped.setdefault(doc.metadata.get("source", "unknown"), []).append(doc)
    
    filtered = []
    for source, group in grouped.items():
        latest = max((d.metadata.get("upload_timestamp", 0) for d in group), default=0)
        filtered.extend([d for d in group if d.metadata.get("upload_timestamp", 0) == latest])
    return filtered

# 👇 THIS IS THE CRITICAL PART THAT MUST MATCH main.py
def generate_response(query: str, chat_history_list: List[Dict[str, str]] = []) -> str:
    vectorstore = get_vectorstore()
    
    # 1. Broad Search
    rewritten = rewrite_query(query)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 60})
    docs = retriever.invoke(rewritten)
    
    # 2. Filter & Rerank
    docs = prefer_latest_per_source(docs)
    docs = hybrid_rerank(rewritten, docs)
    
    # 3. Cross-Encoder (Top 10)
    if docs:
        pairs = [(rewritten, d.page_content) for d in docs[:20]]
        scores = reranker.predict(pairs)
        sorted_docs = [docs[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]]
    else:
        sorted_docs = []

    # 4. Context & History
    context = "\n\n".join([
        f"[[Source: {d.metadata.get('source')} | Page: {d.metadata.get('page')}]]\n{d.page_content}" 
        for d in sorted_docs
    ])
    history = format_chat_history(chat_history_list)

    # 5. Generate
    prompt = f"""You are the CSEA Information Assistant.
    
Conversation History:
{history}

Context:
{context}

Question: {query}

Answer:"""
    
    return get_llm().invoke(prompt).content