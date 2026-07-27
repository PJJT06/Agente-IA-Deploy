import os
import json
import re

INPUT_FOLDER = "json_chunks"   # carpeta donde tienes tus JSON
OUTPUT_FOLDER = "docs"         # carpeta donde guardaremos los .txt

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def sanitize_filename(name):
    # reemplaza caracteres problemáticos
    name = re.sub(r"[^\w\d]+", "_", name)
    return name

for file in os.listdir(INPUT_FOLDER):
    if file.endswith(".json"):
        path = os.path.join(INPUT_FOLDER, file)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Si el JSON es una lista, recorrer cada item
        if isinstance(data, list):
            items = data
        else:
            items = [data]

        for item in items:
            text = item.get("text", "")
            meta = item.get("metadata", {})

            doc_id = meta.get("document_id", "document")
            category = meta.get("category", "unknown")
            source_path = meta.get("source_path", "")
            chunk_index = meta.get("chunk_index", 1)
            page_number = meta.get("page_number", "")
            section = meta.get("section", "")

            # limpiar nombre del documento
            doc_id_clean = sanitize_filename(doc_id.replace(".pdf", ""))

            filename = f"{doc_id_clean}_chunk{chunk_index}.txt"
            out_path = os.path.join(OUTPUT_FOLDER, filename)

            # Construir contenido con metadatos
            content = (
                f"DOCUMENTO: {doc_id}\n"
                f"CATEGORÍA: {category}\n"
                f"ORIGEN: {source_path}\n"
                f"CHUNK: {chunk_index}\n"
                f"PÁGINA: {page_number}\n"
                f"SECCIÓN: {section}\n"
                f"{'-'*60}\n"
                f"{text}\n"
            )

            with open(out_path, "w", encoding="utf-8") as out:
                out.write(content)

            print(f"✔ Convertido con metadatos: {filename}")
