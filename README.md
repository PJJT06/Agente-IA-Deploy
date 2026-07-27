# 🧠 Agente Inteligente con RAG  
Challenge Alura + Oracle ONE

Este proyecto implementa un **Agente de Inteligencia Artificial** capaz de responder preguntas basadas en documentos internos utilizando la técnica **RAG (Retrieval-Augmented Generation)**.  
El agente lee documentos, los indexa y genera respuestas fundamentadas en su contenido


---

## 🏗️ Arquitectura de la Solución

### 1. **Ingesta de Documentos**
Los documentos se almacenan en la carpeta `docs/`.  
El sistema:
- Lee todos los archivos `.txt`
- Limpia y normaliza el texto
- Divide el contenido en fragmentos (chunks)

### 2. **Indexación (TF‑IDF)**
Se construye un índice vectorial usando:
- `TfidfVectorizer`
- Similaridad de coseno

Esto permite encontrar los fragmentos más relevantes para cada pregunta.

### 3. **Recuperación de Contexto**
El archivo `retriever.py`:
- Carga los documentos
- Construye el índice TF‑IDF
- Busca los mejores fragmentos
- Devuelve el contexto al modelo

### 4. **Generación de Respuestas**
El archivo `app.py`:
- Expone un endpoint `/ask`
- Recibe la pregunta del usuario
- Llama al retriever
- Construye una respuesta basada en el contexto

### 5. **API con FastAPI**
La API permite:
- Probar el agente desde `/docs`
- Enviar preguntas vía JSON
- Recibir respuestas estructuradas

---

## 🛠️ Tecnologías Utilizadas

- Python 3.10  
- FastAPI  
- Uvicorn  
- scikit-learn  
- NLTK  
- GitHub (control de versiones)  
- Entorno virtual (venv)
