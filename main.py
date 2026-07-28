import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langgraph.prebuilt import create_react_agent
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Rutas absolutas para evitar errores según el directorio de trabajo
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"

# ========== 1. CONFIGURACIÓN ==========
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_FOLDER = str(DATA_DIR)

# ========== 2. CARGAR PDFs EN VECTORSTORE ==========
print("📂 Leyendo PDFs...")
pdf_docs = []

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_FOLDER, filename))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
            page.metadata["tipo"] = "pdf"
        pdf_docs.extend(pages)
        print(f"   ✅ PDF: {filename}")

print(f"📄 Total páginas PDF: {len(pdf_docs)}")

# ========== 3. VECTOR STORE (solo PDFs) ==========
print("🧠 Creando memoria vectorial para documentos...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
if not pdf_docs:
    raise RuntimeError("No se encontraron páginas PDF en la carpeta data/. El agente requiere al menos un PDF.")
vectorstore = FAISS.from_documents(pdf_docs, embeddings)
print("✅ VectorStore lista")

# ========== 4. CARGAR EXCEL EN DATAFRAME ==========
print("📊 Cargando inventario...")
df_inventario = None
excel_filename = None

for filename in os.listdir(DATA_FOLDER):
    if filename.endswith(".xlsx"):
        filepath = os.path.join(DATA_FOLDER, filename)
        df_inventario = pd.read_excel(filepath)
        excel_filename = filename
        print(f"   ✅ Excel: {filename} ({len(df_inventario)} productos)")
        break

if df_inventario is None:
    raise FileNotFoundError("No se encontró ningún archivo .xlsx en la carpeta data/")

# ========== 5. MODELO DE LENGUAJE ==========
print("🤖 Conectando con Groq...")
# Modelo ligero para generar código pandas en ConsultarInventario
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0.1
)
# Modelo más capaz para el agente orquestador (sigue instrucciones complejas mejor)
llm_orquestador = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)
print("✅ Groq conectado")

# ========== 6. HERRAMIENTAS DEL AGENTE ==========

# Tool 1: Consultar documentos PDF (RAG)
@tool
def ConsultarManualesYPoliticas(pregunta: str) -> str:
    """Útil para responder preguntas sobre reglamentos internos, procedimientos operativos,
    políticas de atención al cliente, devoluciones, cambios, código de ética,
    preguntas frecuentes (FAQ) y manual de proveedores.
    NO usar para preguntas de stock o inventario."""
    docs = vectorstore.similarity_search(pregunta, k=5)
    if not docs:
        return "No encontré información relevante en los documentos."
    return "\n\n".join(
        f"[Fuente: {d.metadata['source']}]\n{d.page_content}" for d in docs
    )

# Tool 2: Consultar inventario via Python/pandas ejecutado por el LLM
# ------------------------------------------------------------------
# PROBLEMA: los nombres de columna tienen acentos (Descripción, Categoría…)
# lo que hace que el LLM a veces los genere incorrectamente → filtro vacío → 0.
# SOLUCIÓN: normalizar a ASCII puro antes de pasarlos al REPL y al prompt.
import unicodedata

def _ascii_col(col: str) -> str:
    """'Descripción' → 'Descripcion', 'Stock Mínimo' → 'Stock_Minimo'"""
    sin_tilde = unicodedata.normalize("NFD", col).encode("ascii", "ignore").decode("ascii")
    return sin_tilde.strip().replace(" ", "_").replace(".", "").replace("(", "").replace(")", "")

# DataFrame con columnas ASCII para el REPL
_df_repl = df_inventario.copy()
_df_repl.columns = [_ascii_col(c) for c in df_inventario.columns]

# Mapa original → normalizado (para mostrar en el prompt)
_COL_MAP = {_ascii_col(c): c for c in df_inventario.columns}
_COLUMNAS_NORMALIZADAS = list(_COL_MAP.keys())

# Muestra representativa: 1 fila por cada categoría presente
_MUESTRA = (
    _df_repl
    .groupby("Categoria", group_keys=False)
    .apply(lambda g: g.head(1))
    .head(10)
    .to_string(index=False)
)

_REPL_INV = PythonAstREPLTool(locals={"df": _df_repl, "pd": pd})

_CODEGEN_PROMPT = """Eres un experto en pandas. Tienes un DataFrame llamado 'df' con estas columnas (nombres exactos):
{columnas}

Muestra de datos (1 fila por categor\u00eda):
{muestra}

IMPORTANTE sobre los datos:
- 'Descripcion' tiene el nombre del producto (ej: "Desodorante Aerosol Invisible 150ml").
- 'Marca' tiene la marca comercial (ej: "Rexona", "Dove", "Goya").
- Si el usuario menciona una marca, busca en 'Marca'. Si menciona un tipo, busca en 'Descripcion'.
- Cuando no est\u00e9s seguro, busca en AMBAS con OR:
    mask = (df['Descripcion'].str.contains('X', case=False, na=False) | df['Marca'].str.contains('X', case=False, na=False))

Escribe SOLO el c\u00f3digo Python para responder:
"{pregunta}"

Reglas ESTRICTAS:
- Usa exactamente los nombres de columna listados arriba.
- Usa la variable 'df' tal como est\u00e1.
- Usa print() para mostrar el resultado final.
- Si calculas stock total usa .sum(). Si muestras filas usa .to_string(index=False).
- No importes librer\u00edas. No uses markdown. Solo c\u00f3digo Python listo para ejecutar.
"""

@tool
def ConsultarInventario(pregunta: str) -> str:
    """Útil para responder preguntas sobre stock actual, productos en inventario,
    precios de venta, costos unitarios, proveedores de productos, ubicación en pasillos,
    fechas de vencimiento, lotes, stock mínimo/máximo y tiempos de reposición.
    Usar SIEMPRE que la pregunta mencione productos, stock, disponibilidad o inventario."""
    try:
        # Paso 1: el LLM genera el código pandas usando nombres ASCII sin acentos
        prompt_codigo = _CODEGEN_PROMPT.format(
            columnas=_COLUMNAS_NORMALIZADAS,
            muestra=_MUESTRA,
            pregunta=pregunta,
        )
        codigo_raw = llm.invoke(prompt_codigo).content.strip()

        # Limpiar bloques markdown si el modelo los incluyó
        if "```" in codigo_raw:
            partes = codigo_raw.split("```")
            codigo_raw = partes[1]
            if codigo_raw.startswith("python"):
                codigo_raw = codigo_raw[6:]
        codigo = codigo_raw.strip()

        # Paso 2: ejecutar el código en el REPL con el df normalizado
        resultado = _REPL_INV.run(codigo)
        return resultado.strip() if resultado and resultado.strip() else "No se encontró información en el inventario."
    except Exception as e:
        return f"Error consultando inventario: {str(e)}"

# Lista de herramientas para el agente orquestador
tools = [ConsultarManualesYPoliticas, ConsultarInventario]

# ========== 7. AGENTE ORQUESTADOR (LangGraph) ==========
print("🧩 Armando agente con herramientas...")

SYSTEM_PROMPT = """Eres el asistente interno de Mercado Central 24h. Tu ÚNICA fuente de información son las herramientas disponibles.

REGLAS CRÍTICAS — síguelas siempre sin excepción:
1. SIEMPRE llama a la herramienta correspondiente antes de responder. Nunca uses tu conocimiento general.
2. ConsultarInventario devuelve datos REALES de nuestra base de datos en vivo. El número que devuelve ES el stock real. Repórtalo textualmente.
3. ConsultarManualesYPoliticas devuelve el contenido REAL de nuestros documentos internos.
4. NUNCA digas frases como "no tengo datos en tiempo real", "no puedo acceder", "contacta al personal", ni nada similar. LOS DATOS DE LAS HERRAMIENTAS SON DATOS REALES.
5. Si la herramienta devuelve un número o tabla, preséntalo de forma clara al usuario.
6. Responde siempre en español, de forma concisa y directa.

Herramientas:
- ConsultarInventario → para stock, precios, productos, disponibilidad, proveedores, pasillos, lotes, vencimientos.
- ConsultarManualesYPoliticas → para reglamentos, políticas, devoluciones, FAQ, procedimientos."""

agente_orquestador = create_react_agent(
    model=llm_orquestador,
    tools=tools,
    prompt=SYSTEM_PROMPT,
)

print("✅ Agente listo")

# ========== 8. API ==========
app = FastAPI(title="Agente Mercado Central 24h")

# Servir archivos estáticos (CSS, JS, imágenes, etc.) con ruta absoluta
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class Question(BaseModel):
    pregunta: str

@app.post("/preguntar")
async def preguntar(q: Question):
    try:
        resultado = agente_orquestador.invoke({"messages": [("user", q.pregunta)]})

        # LangGraph devuelve una lista de mensajes; el último es la respuesta del agente
        mensajes = resultado.get("messages", [])
        respuesta = ""
        for msg in reversed(mensajes):
            if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage" and msg.content:
                respuesta = msg.content
                break
        if not respuesta:
            respuesta = "El agente no generó una respuesta."

        # Detectar fuentes a partir de los ToolMessages del resultado
        fuentes: list[str] = []
        for msg in mensajes:
            tool_name = getattr(msg, "name", "")
            if tool_name == "ConsultarInventario" and excel_filename:
                fuentes.append(excel_filename)
            elif tool_name == "ConsultarManualesYPoliticas":
                fuentes.extend(list(set(d.metadata["source"] for d in pdf_docs)))

        if not fuentes:
            fuentes = ["Agente Mercado Central 24h"]

        return {
            "respuesta": respuesta,
            "fuentes": list(set(fuentes))
        }
    except Exception as e:
        return {
            "respuesta": f"Ocurrió un error procesando tu pregunta: {str(e)}",
            "fuentes": []
        }

@app.get("/")
async def serve_chat():
    # Ruta absoluta para que funcione independientemente del directorio de trabajo
    return FileResponse(str(STATIC_DIR / "index.html"))

# ========== 9. VERIFICACIÓN AL INICIAR ==========
@app.on_event("startup")
async def on_startup():
    print("\n🚀 Servidor corriendo en http://127.0.0.1:8000")
    print("💡 Prueba preguntando: '¿Cuánto stock hay de Arroz Blanco Tipo 1?'")
    print("💡 O: '¿Cuál es la política de devoluciones?'\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)