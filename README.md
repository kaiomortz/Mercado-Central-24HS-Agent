# Agente de IA — Mercado Central 24h

Este proyecto es un agente de inteligencia artificial que responde preguntas sobre los documentos internos de un supermercado: políticas de devolución, reglamento, proveedores, preguntas frecuentes y el inventario de productos.

Cargas de documentos, el agente los lee, los entiende y después responde lo que le preguntes.

---

## Cómo funciona

El agente utiliza **RAG** (Retrieval-Augmented Generation)

**Tecnologías:**
- **FastAPI** para la API web
- **LangChain** para armar el pipeline de RAG
- **FAISS** como base de datos vectorial (para buscar rápido entre los documentos)
- **HuggingFace Embeddings** (modelo multilingüe) para convertir texto en vectores
- **Groq** (Llama 3.1 8B) como modelo de lenguaje para generar las respuestas
- **PyPDF y Pandas** para leer los PDFs y el Excel

---

## Cómo correrlo en tu máquina

1. Clonar el repo y entrar a la carpeta
2. Crear el entorno virtual:
   python -m venv venv


## ACTIVAR:
### Windows
venv\Scripts\activate

### Mac/Linux
source venv/bin/activate

## INSTALAR DEPENDENCIAS
pip install -r requirements.txt

## CREAR .env CON LA API KEY
GROQ_API_KEY= API_KEY_ACA

## CORRER LA APLICACION
uvicorn main:app --host 127.0.0.1 --port 8000

## URL PARA REALIZAR LAS CONSULTAS CUANDO ESTE CORRIENDO
http://127.0.0.1:8000/docs


#####
Ejemplos de uso:
Averiguar politica de devoluciones
Cantidad de productos en stock
Beneficios de clientes VIP
requisitos para ser proveedor
#####