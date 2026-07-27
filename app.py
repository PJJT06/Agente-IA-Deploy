import streamlit as st
import os
from groq import Groq
from retriever import retrieve
from dotenv import load_dotenv

# -----------------------------
# Configuración inicial
# -----------------------------
load_dotenv()

# Cargar clave de Groq desde .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ No se encontró la clave GROQ_API_KEY en el archivo .env")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# Interfaz Streamlit
# -----------------------------
st.set_page_config(page_title="Agente IA", page_icon="🤖", layout="wide")
st.title("🤖 Agente IA con Groq + Retriever Simple")

st.markdown("Este agente usa búsqueda TF‑IDF y metadatos para construir contexto antes de consultar Groq.")

query = st.text_input("🔍 Escribe tu pregunta:")

if st.button("Enviar") or query:
    with st.spinner("Buscando información y generando respuesta..."):
        # Recuperar contexto
        context = retrieve(query)

        # Construir prompt para Groq
        prompt = f"""
        Usa el siguiente contexto para responder la pregunta del usuario de forma clara y completa.

        CONTEXTO:
        {context}

        PREGUNTA:
        {query}

        RESPUESTA:
        """

        # Llamar al modelo de Groq
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )

        answer = response.choices[0].message.content

        # Mostrar resultados
        st.subheader("🧠 Respuesta del agente:")
        st.write(answer)

        st.subheader("📚 Contexto usado:")
        st.text(context)
