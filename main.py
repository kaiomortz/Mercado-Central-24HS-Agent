import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ========== 1. CONFIGURACIÓN ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FOLDER = "data"

# ========== 2. CARGAR DOCUMENTOS ==========
print("📂 Leyendo documentos...")
all_docs = []

# Leer PDFs
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, filename))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        all_docs.extend(pages)
        print(f"   ✅ PDF: {filename}")

# Leer Excel
for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(os.path.join(DATA_FOLDER, filename))
        for idx, row in df.iterrows():
            text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            all_docs.append(Document(page_content=text, metadata={"source": filename}))
        print(f"   ✅ Excel: {filename}")

print(f"📊 Total de fragmentos: {len(all_docs)}")

# ========== 3. VECTOR STORE ==========
print("🧠 Creando memoria vectorial (puede tardar la primera vez)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
vectorstore = FAISS.from_documents(all_docs, embeddings)
print("✅ Memoria lista")

# ========== 4. MODELO DE LENGUAJE ==========
print("🤖 Conectando con Groq...")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)
print("✅ Groq conectado")

# ========== 5. CADENA RAG (sin RetrievalQA) ==========
template = """Responde la pregunta basándote ÚNICAMENTE en el siguiente contexto.
Si la respuesta no está en el contexto, di "No encontré información sobre eso en los documentos".

Contexto:
{context}

Pregunta: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Esta es la magia: busca documentos similares, arma el prompt y le pregunta a Gemini
rag_chain = (
    {
        "context": vectorstore.as_retriever(search_kwargs={"k": 4}) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ========== 6. API ==========
app = FastAPI(title="Agente Mercado Central 24h")

class Question(BaseModel):
    pregunta: str

@app.post("/preguntar")
async def preguntar(q: Question):
    # Buscamos los documentos usados para dar las fuentes
    docs_usados = vectorstore.similarity_search(q.pregunta, k=4)
    respuesta = rag_chain.invoke(q.pregunta)
    
    return {
        "respuesta": respuesta,
        "fuentes": list(set([doc.metadata["source"] for doc in docs_usados]))
    }

@app.get("/")
async def root():
    return {
        "mensaje": "Agente Mercado Central 24h activo",
        "instruccion": "Hacé un POST a /preguntar con JSON {'pregunta': 'tu pregunta'}"
    }