from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Inicialización
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

CHROMA_PATH = r"C:\Users\pjjt0\OneDrive\Escritorio\IA_AGENTE\chroma"
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("ia_agente_docs")


# -----------------------------
# Función: búsqueda vectorial
# -----------------------------
def semantic_search(query: str, n_results=20):
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "embeddings"]  # pedimos embeddings
    )


# -----------------------------
# Boost por palabras clave
# -----------------------------
def keyword_boost(query: str, doc: str):
    score = 0
    query_words = query.lower().split()

    for w in query_words:
        if w in doc.lower():
            score += 0.3

    return score


# -----------------------------
# Boost por título del documento
# -----------------------------
def title_boost(query: str, meta):
    title = meta.get("document_id", "").lower()
    query_words = query.lower().split()

    score = 0
    for w in query_words:
        if w in title:
            score += 0.5

    return score


# -----------------------------
# Reranking final
# -----------------------------
def rerank(query, docs, metas, embeddings):
    query_emb = embedding_model.encode(query)

    ranked = []

    for doc, meta, emb in zip(docs, metas, embeddings):
        cos_sim = float(np.dot(query_emb, emb) /
                        (np.linalg.norm(query_emb) * np.linalg.norm(emb)))

        kw = keyword_boost(query, doc)
        tb = title_boost(query, meta)

        final_score = cos_sim + kw + tb

        ranked.append((final_score, doc, meta))

    ranked.sort(key=lambda x: x[0], reverse=True)

    top_docs = [d for _, d, _ in ranked[:5]]
    top_metas = [m for _, _, m in ranked[:5]]

    return top_docs, top_metas


# -----------------------------
# Ensamblar contexto
# -----------------------------
def build_context(docs, metas):
    context = ""
    for doc, meta in zip(docs, metas):
        context += f"Documento: {meta.get('document_id')}\n"
        context += f"Chunk: {meta.get('chunk_index')}\n"
        context += f"Contenido:\n{doc}\n"
        context += "-" * 60 + "\n"
    return context


# -----------------------------
# Función principal (sin categorías)
# -----------------------------
def retrieve(query: str):
    results = semantic_search(query)

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    embeddings = results.get("embeddings", None)

    if embeddings is None or embeddings[0] is None:
        embeddings = [embedding_model.encode(doc) for doc in docs]
    else:
        embeddings = embeddings[0]

    top_docs, top_metas = rerank(query, docs, metas, embeddings)

    return build_context(top_docs, top_metas)
