from groq import Groq
from src.retriever import retrieve
import time
from metrics import log_metrics
import os

# Cliente Groq usando variable de entorno
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def validate_context(context: str) -> bool:
    """Valida que el contexto no esté vacío y tenga contenido útil."""
    if context is None:
        return False
    if len(context.strip()) == 0:
        return False
    if "Contenido:" not in context:
        return False
    return True

def build_prompt(query: str, context: str) -> str:
    """Construye el prompt que se enviará al modelo."""
    return f"""
Eres un asistente que responde únicamente con base en el contexto proporcionado.
Si la información no está en el contexto, debes decir:
"❗ No encontré esta información en los documentos disponibles."

Instrucciones estrictas:
- No inventes información.
- No uses conocimiento externo.
- Cita siempre el documento y el chunk de donde proviene cada parte de la respuesta.
- Si el contexto no es suficiente, dilo claramente.

---------------- CONTEXTO ----------------
{context}
------------------------------------------

Pregunta del usuario: {query}

Responde de forma clara, precisa y citando las fuentes:
"""

def answer_query(query: str):
    """Genera la respuesta final usando llama-3.1-8b-instant en Groq."""

    # Recuperar contexto desde el RAG
    context = retrieve(query)

    # Validar contexto
    if not validate_context(context):
        log_metrics(query, context, 0)
        return "❗ No encontré esta información en los documentos disponibles."

    # Construir prompt
    prompt = build_prompt(query, context)

    # Medir tiempo de respuesta
    start = time.time()

    # Llamada al modelo Groq
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    end = time.time()

    # Log de métricas
    log_metrics(query, context, end - start)

    # Extraer contenido de forma segura
    try:
        content = completion.choices[0].message.content
    except Exception:
        return "❗ El modelo no devolvió contenido. Revisa el prompt o el contexto."

    if content is None:
        return "❗ El modelo devolvió una respuesta vacía."

    return content.strip()
