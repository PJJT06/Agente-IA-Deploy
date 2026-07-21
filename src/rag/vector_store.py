# src/rag/vector_store.py
import chromadb

def init_vector_store():
    """
    Inicializa ChromaDB y devuelve la colección donde guardaremos los embeddings.
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="ia_agente_docs")
    return collection

def add_embeddings(collection, embeddings):
    """
    Agrega los embeddings y sus metadatos a la colección.
    """
    ids = []
    texts = []
    metas = []
    vectors = []

    for i, item in enumerate(embeddings):
        ids.append(f"chunk_{i}")
        texts.append(item["text"])
        metas.append(item["metadata"])
        vectors.append(item["embedding"])

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=vectors
    )
