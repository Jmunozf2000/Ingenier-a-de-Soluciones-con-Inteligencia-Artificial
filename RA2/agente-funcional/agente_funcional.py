import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

import herramientas
from herramientas import HERRAMIENTAS, set_retriever
from memoria import crear_memoria_corto_plazo, MemoriaLargoPlazo

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("GITHUB_TOKEN"),
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=os.getenv("OPENAI_EMBEDDINGS_URL", os.getenv("OPENAI_BASE_URL")),
    api_key=os.getenv("GITHUB_TOKEN"),
)

SYSTEM_PROMPT = """Eres un Agente Planificador de Mantenimiento del Taller de Avionica
de LATAM Airlines. Transformas una orden de alto nivel en un plan ejecutable y seguro.

ESQUEMA DE PLANIFICACION (siguelo SIEMPRE en orden):
1. DESCOMPONER: divide la orden en subtareas concretas.
2. PRIORIZAR: ordena por prioridad (seguridad > inspeccion > ajuste > registro).
3. FUNDAMENTAR: usa 'consulta_requisitos_mantenimiento' para respaldo documental.
4. VALIDAR: usa 'validar_parametro_tecnico' para comprobar calibres o torques.
5. GENERAR: usa 'generar_plan_mantenimiento' para emitir el checklist/OT final.

REGLAS DE DECISION ADAPTATIVA:
- Si la consulta NO devuelve respaldo, NO inventes: marca la subtarea como
  "requiere verificacion con supervisor" y continua con las demas.
- Si una validacion resulta [NO APTO], NO generes el plan con ese valor:
  corrige el parametro o marca la subtarea como bloqueada y explica por que.
- Solo invoca 'generar_plan_mantenimiento' cuando las subtareas criticas esten
  fundamentadas y validadas. Cita siempre la fuente en el plan final.

Responde en espanol tecnico, claro y conciso."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


def construir_retriever(fuentes: list, persist_dir: str = "./chroma_db"):
    documentos = []
    for fuente in fuentes:
        try:
            if fuente.endswith(".pdf"):
                loader = PyPDFLoader(fuente)
            elif fuente.startswith("http"):
                loader = WebBaseLoader(fuente)
            else:
                loader = TextLoader(fuente, encoding="utf-8")
            documentos.extend(loader.load())
            print(f"[OK] Cargado: {fuente}")
        except Exception as e:
            print(f"[ERROR] {fuente}: {e}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documentos)
    print(f"[OK] {len(chunks)} chunks indexados")
    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_dir
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def construir_agente(retriever, memoria_corto):
    set_retriever(retriever)
    agente = create_tool_calling_agent(llm, HERRAMIENTAS, PROMPT)
    return AgentExecutor(
        agent=agente, tools=HERRAMIENTAS, memory=memoria_corto,
        verbose=True, handle_parsing_errors=True, max_iterations=8,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("AGENTE FUNCIONAL - PLANIFICADOR DE MANTENIMIENTO")
    print("Taller de Avionica - LATAM Airlines")
    print("=" * 60)
    memoria_largo = MemoriaLargoPlazo()
    print("\n[CONTEXTO PREVIO]")
    print(memoria_largo.recuperar_contexto())
    print("\n[1] Indexando documentacion tecnica...")
    retriever = construir_retriever(["doc_demo.txt"])
    print("\n[2] Inicializando memoria de corto plazo...")
    memoria_corto = crear_memoria_corto_plazo(k=5)
    print("\n[3] Construyendo agente funcional...")
    agente = construir_agente(retriever, memoria_corto)
    orden = ("Planifica la inspeccion de cableado del componente de avionica: "
             "revisar empalmes, validar que el cable AWG 16 soporte 18 A y "
             "confirmar el torque del terminal_anillo_10 a 22 lb-in.")
    print(f"\n[ORDEN]: {orden}\n")
    resultado = agente.invoke({"input": orden})
    print("\n[PLAN GENERADO]:")
    print(resultado["output"])
    memoria_largo.guardar_interaccion(orden, resultado["output"])
    print("\n[OK] Interaccion guardada en memoria de largo plazo.")
