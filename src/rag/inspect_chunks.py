from chromadb import PersistentClient

# ⚠️ Usa SIEMPRE la ruta ABSOLUTA para evitar leer bases viejas
CHROMA_PATH = r"C:\Users\pjjt0\OneDrive\Escritorio\IA_AGENTE\chroma"

client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("ia_agente_docs")

results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    print("\n-------------------------------")
    print(f"Documento: {meta.get('document_id')}")
    print(f"Chunk: {meta.get('chunk_index')}")
    print("Contenido:")
    print(doc)
