# ============================================================
# Agente RAG - Consulta Tecnica Avionica LATAM Airlines
# ISY0101 - Ingenieria de Soluciones con IA
# ============================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- Configuracion GitHub Models (segun curso) ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("GITHUB_TOKEN"),
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("GITHUB_TOKEN"),
)

# --- System Prompt (Few-Shot CoT) ---
SYSTEM_PROMPT = """
Eres un asistente tecnico especializado en mantenimiento aeronautico
para tecnicos del Taller de Avionica de LATAM Airlines.

Tu funcion es apoyar la busqueda de documentacion tecnica complementaria
a los manuales CMM disponibles en MyBoeingFleet ToolBox.

REGLAS ESTRICTAS:
1. Responde UNICAMENTE usando la informacion del CONTEXTO proporcionado.
2. Si el contexto no contiene informacion suficiente, responde exactamente:
   "No se encontro documentacion que respalde esta consulta. 
    Verifique con su supervisor o consulte directamente la fuente oficial."
3. Cada respuesta DEBE incluir la cita de fuente con formato:
   [Documento | Seccion | Revision/Fecha]
4. Incluye siempre el URL de la fuente cuando este disponible en el contexto.
5. Nunca inventes ni inferyas datos tecnicos no presentes en el contexto.
6. Usa lenguaje tecnico apropiado para tecnicos certificados AME.

CONTEXTO:
{context}

PREGUNTA:
{question}
"""

# --- Carga y procesamiento de documentos ---
def cargar_documentos(fuentes: list) -> list:
    """Carga documentos desde URLs o archivos PDF."""
    documentos = []
    for fuente in fuentes:
        try:
            if fuente.endswith(".pdf"):
                loader = PyPDFLoader(fuente)
            else:
                loader = WebBaseLoader(fuente)
            documentos.extend(loader.load())
            print(f"[OK] Cargado: {fuente}")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar {fuente}: {e}")
    return documentos

def crear_vectorstore(documentos: list) -> Chroma:
    """Fragmenta y vectoriza los documentos."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documentos)
    print(f"[OK] {len(chunks)} chunks generados")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def construir_rag_chain(vectorstore: Chroma):
    """Construye el pipeline RAG completo."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    
    def format_docs(docs):
        resultado = []
        for doc in docs:
            fuente = doc.metadata.get("source", "Fuente desconocida")
            pagina = doc.metadata.get("page", "")
            texto = f"[Fuente: {fuente} | Pagina: {pagina}]\n{doc.page_content}"
            resultado.append(texto)
        return "\n\n---\n\n".join(resultado)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Demo con fuentes publicas ---
if __name__ == "__main__":
    print("=" * 60)
    print("AGENTE RAG - CONSULTA TECNICA AVIONICA")
    print("Taller Avionica - LATAM Airlines")
    print("=" * 60)
    
    # Fuentes demo (publicas y verificables)
    fuentes_demo = [
        "https://rgl.faa.gov/Regulatory_and_Guidance_Library/rgAdvisoryCircular.nsf/0/153f764f6835ee6f862569b9006e87bc/$FILE/AC43.13-1B.pdf"
    ]
    
    print("\n[1] Cargando documentos...")
    docs = cargar_documentos(fuentes_demo)
    
    if docs:
        print("\n[2] Creando vector store...")
        vs = crear_vectorstore(docs)
        
        print("\n[3] Construyendo pipeline RAG...")
        chain = construir_rag_chain(vs)
        
        # Consulta de ejemplo
        consulta = "What are the requirements for wire splicing in aircraft wiring repairs?"
        print(f"\n[CONSULTA]: {consulta}")
        print("\n[RESPUESTA]:")
        respuesta = chain.invoke(consulta)
        print(respuesta)
    else:
        print("[!] No se pudieron cargar documentos. Verifique su conexion y GITHUB_TOKEN.")
