import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Cargar documentos y metadatos
# -----------------------------
def load_documents():
    docs = []
    metas = []

    base_path = "docs"

    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".txt"):
                full_path = os.path.join(root, f)

                with open(full_path, "r", encoding="utf-8") as file:
                    content = file.read()

                # Separar metadatos del contenido
                parts = content.split("-" * 60)
                meta_block = parts[0]
                text_block = parts[1] if len(parts) > 1 else ""

                # Extraer metadatos
                meta_lines = meta_block.split("\n")
                meta = {}

                for line in meta_lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        meta[key.strip().lower()] = value.strip()

                docs.append(text_block.strip())
                metas.append(meta)

    return docs, metas


DOCUMENTS, METAS = load_documents()

# -----------------------------
# TF‑IDF Vectorizer
# -----------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(DOCUMENTS)

# -----------------------------
# Búsqueda por similitud TF‑IDF
# -----------------------------
def semantic_search(query, top_k=10):
    query_vec = vectorizer.transform([query])
    scores = (query_vec @ tfidf_matrix.T).toarray()[0]

    ranked_idx = np.argsort(scores)[::-1][:top_k]

    docs = [DOCUMENTS[i] for i in ranked_idx]
    metas = [METAS[i] for i in ranked_idx]

    return docs, metas

# -----------------------------
# Boost por palabras clave
# -----------------------------
def keyword_boost(query, doc):
    score = 0
    for w in query.lower().split():
        if w in doc.lower():
            score += 0.3
    return score

# -----------------------------
# Boost por título del documento
# -----------------------------
def title_boost(query, meta):
    score = 0
    title = meta.get("documento", "").lower()
    for w in query.lower().split():
        if w in title:
            score += 0.5
    return score

# -----------------------------
# Boost por categoría
# -----------------------------
def category_boost(query, meta):
    score = 0
    category = meta.get("categoría", "").lower()
    for w in query.lower().split():
        if w in category:
            score += 0.4
    return score

# -----------------------------
# Boost por sección
# -----------------------------
def section_boost(query, meta):
    score = 0
    section = meta.get("sección", "").lower()
    for w in query.lower().split():
        if w in section:
            score += 0.2
    return score

# -----------------------------
# Reranking final
# -----------------------------
def rerank(query, docs, metas):
    ranked = []

    for doc, meta in zip(docs, metas):
        tfidf_score = 1.0
        kw = keyword_boost(query, doc)
        tb = title_boost(query, meta)
        cb = category_boost(query, meta)
        sb = section_boost(query, meta)

        final_score = tfidf_score + kw + tb + cb + sb
        ranked.append((final_score, doc, meta))

    ranked.sort(key=lambda x: x[0], reverse=True)

    top_docs = [d for _, d, _ in ranked[:5]]
    top_metas = [m for _, _, m in ranked[:5]]

    return top_docs, top_metas

# -----------------------------
# Construcción del contexto
# -----------------------------
def build_context(docs, metas):
    context = ""
    for doc, meta in zip(docs, metas):
        context += f"Documento: {meta.get('documento')}\n"
        context += f"Categoría: {meta.get('categoría')}\n"
        context += f"Chunk: {meta.get('chunk')}\n"
        context += f"Sección: {meta.get('sección')}\n"
        context += f"Contenido:\n{doc}\n"
        context += "-" * 60 + "\n"
    return context

# -----------------------------
# Función principal
# -----------------------------
def retrieve(query):
    docs, metas = semantic_search(query)
    top_docs, top_metas = rerank(query, docs, metas)
    return build_context(top_docs, top_metas)
