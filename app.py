import streamlit as st
from src.retriever import retrieve
from src.answer_query import answer_query

st.set_page_config(page_title="IA Agente", layout="wide")

# -----------------------------
# Título
# -----------------------------
st.title("🤖 IA Agente — Sistema RAG con Groq + Llama 3.1")

st.write("Ingresa tu pregunta y el agente buscará la información en tus documentos usando RAG.")

# -----------------------------
# Input del usuario
# -----------------------------
query = st.text_input(
    "Pregunta:",
    placeholder="Ejemplo: ¿Por qué una transferencia puede ser rechazada?"
)

# -----------------------------
# Botón de ejecutar
# -----------------------------
if st.button("Consultar"):
    if not query.strip():
        st.warning("Por favor ingresa una pregunta.")
        st.stop()

    # Recuperar contexto
    with st.spinner("🔍 Buscando información relevante en tus documentos..."):
        context = retrieve(query)

    # Panel colapsable para ver el contexto
    with st.expander("📄 Ver contexto utilizado (opcional)"):
        st.text(context)

    # Generar respuesta
    with st.spinner("🤖 Generando respuesta con Llama 3.1..."):
        answer = answer_query(query)

    st.subheader("💬 Respuesta del agente")
    st.write(answer)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Desarrollado por Jose — RAG + Groq + Llama 3.1")
