import os
import shutil
import pdfplumber
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# Ruta donde están tus PDFs y MD
BASE_DIR = r"C:\Users\pjjt0\OneDrive\Escritorio\IA_AGENTE\data\raw"

# Modelo de embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Base vectorial
client = PersistentClient(path=r"C:\Users\pjjt0\OneDrive\Escritorio\IA_AGENTE\chroma")

collection = client.get_or_create_collection("ia_agente_docs")


# -----------------------------
# 1. LECTURA DE PDF
# -----------------------------
def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -----------------------------
# 2. LECTURA DE MARKDOWN
# -----------------------------
def read_markdown(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# -----------------------------
# 3. LIMPIEZA DE TEXTO
# -----------------------------
def clean_text(text):
    cleaned = []

    for line in text.splitlines():
        line = line.strip()

        # eliminar separadores
        if line.startswith("---") or line.startswith("==="):
            continue

        # eliminar backslashes al final
        if line.endswith("\\"):
            line = line[:-1]

        # ignorar líneas vacías múltiples
        if line.strip() == "":
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# -----------------------------
# 4. CHUNKING POR ENCABEZADOS
# -----------------------------
def chunk_text(text):
    lines = text.splitlines()
    chunks = []
    current_title = None
    current_content = []

    for line in lines:
        stripped = line.strip()

        # Detectar encabezados Markdown
        if stripped.startswith("#"):
            # Si ya había un chunk, guardarlo
            if current_title is not None:
                chunks.append(current_title + "\n" + "\n".join(current_content))

            current_title = stripped
            current_content = []
        else:
            current_content.append(stripped)

    # Último chunk
    if current_title is not None:
        chunks.append(current_title + "\n" + "\n".join(current_content))

    # Si no hay encabezados → fallback por tamaño
    if len(chunks) == 0:
        big = text
        fallback_chunks = []
        buf = []

        for line in big.splitlines():
            buf.append(line)
            if len("\n".join(buf)) > 800:
                fallback_chunks.append("\n".join(buf))
                buf = []

        if buf:
            fallback_chunks.append("\n".join(buf))

        return fallback_chunks

    return chunks


# -----------------------------
# 5. INDEXACIÓN
# -----------------------------
def index_documents(reset=False):

    if reset:
        shutil.rmtree("./chroma", ignore_errors=True)
        print("🧹 Base Chroma eliminada. Reindexando desde cero...")
        global collection
        client = PersistentClient(path="./chroma")
        collection = client.get_or_create_collection("ia_agente_docs")

    print("📄 Indexando documentos en Chroma...\n")

    for file in os.listdir(BASE_DIR):
        path = os.path.join(BASE_DIR, file)

        if not os.path.isfile(path):
            continue

        print(f"📄 Procesando archivo: {file}")

        # PDF
        if file.lower().endswith(".pdf"):
            raw_text = read_pdf(path)

        # Markdown
        elif file.lower().endswith(".md"):
            raw_text = read_markdown(path)

        else:
            print(f"⚠️ Archivo ignorado (no PDF/MD): {file}")
            continue

        # Limpiar texto
        cleaned = clean_text(raw_text)

        # Chunkear
        chunks = chunk_text(cleaned)

        # Guardar chunks
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            print(f"   ➤ Chunk {i} generado ({len(chunk)} caracteres)")

            embedding = embedding_model.encode([chunk], convert_to_numpy=True).tolist()[0]
            doc_id = f"{file}_chunk_{i}"

            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{
                    "document_id": file,
                    "chunk_index": i
                }]
            )

    print("\n✔️ Indexación completa.")


# -----------------------------
# 6. EJECUCIÓN
# -----------------------------
if __name__ == "__main__":
    import sys
    reset_flag = "--reset" in sys.argv
    index_documents(reset=reset_flag)
