import streamlit as st
import os
from groq import Groq
from retriever import retrieve
from dotenv import load_dotenv

# -----------------------------
# Configuración inicial
# -----------------------------
load_dotenv()

# Cargar clave de Groq desde .env o desde variables de Render
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ No se encontró la clave GROQ_API_KEY. "
             "Asegúrate de definirla en Render → Environment Variables.")
    st.stop()

# Inicializar cliente Groq
try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Error inicializando Groq: {e}")
    st.stop()

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.set_page_config(page_title="Agente IA", page_icon="🤖", layout="wide")
st.title("🤖 Agente IA con Groq + RAG (TF‑IDF + Metadatos + Reranking)")

st.markdown("""
Este agente usa **TF‑IDF**, **metadatos**, **boosting** y **reranking** para construir contexto
antes de consultar el modelo **Llama‑3.3‑70B** de Groq.
""")

query = st.text_input("🔍 Escribe tu pregunta:")

# -----------------------------
# Botón de ejecución
# -----------------------------
if st.button("Enviar"):
    if not query.strip():
        st.warning("⚠️ Por favor escribe una pregunta válida.")
        st.stop()

    with st.spinner("Buscando información y generando respuesta..."):

        # -----------------------------
        # Recuperar contexto
        # -----------------------------
        context = retrieve(query)

        if not context or "⚠️" in context:
            st.error("❌ No se pudo construir el contexto. "
                     "Revisa si la carpeta 'docs/' contiene documentos válidos.")
            st.text(context)
            st.stop()

        # -----------------------------
        # Construir prompt para Groq
        # -----------------------------
        prompt = f"""
        Usa el siguiente contexto para responder la pregunta del usuario de forma clara, precisa y completa.

        CONTEXTO:
        {context}

        PREGUNTA:
        {query}

        RESPUESTA:
        """

        # -----------------------------
        # Llamar al modelo Groq
        # -----------------------------
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )

            answer = response.choices[0].message.content

        except Exception as e:
            st.error(f"❌ Error llamando al modelo Groq: {e}")
            st.stop()

        # -----------------------------
        # Mostrar resultados
        # -----------------------------
        st.subheader("🧠 Respuesta del agente:")
        st.write(answer)

        st.subheader("📚 Contexto usado:")
        st.text(context)
