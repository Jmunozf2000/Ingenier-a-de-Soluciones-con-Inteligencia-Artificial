import os
import sys
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

import herramientas
from herramientas import (
    consulta_requisitos_mantenimiento,
    validar_parametro_tecnico,
    generar_plan_mantenimiento,
)

load_dotenv()


def crear_llm():
    return LLM(
        model="openai/gpt-4o-mini",
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("GITHUB_TOKEN"),
        temperature=0,
    )


def construir_crew(retriever=None):
    if retriever is not None:
        herramientas.set_retriever(retriever)
    llm = crear_llm()

    manager = Agent(
        role="Manager de Mantenimiento",
        goal=("Descomponer la orden en subtareas priorizadas y delegar cada una "
              "al especialista adecuado, integrando los resultados en un plan seguro."),
        backstory=("Jefe de taller de avionica con 20 anios de experiencia. No ejecuta "
                   "tareas tecnicas: planifica, prioriza (seguridad primero) y coordina."),
        llm=llm,
        allow_delegation=True,
        verbose=True,
    )
    documental = Agent(
        role="Especialista Documental",
        goal="Encontrar y citar los requisitos tecnicos de cada subtarea en la documentacion.",
        backstory="Domina normas FAA/EASA y manuales OEM. Siempre cita la fuente.",
        tools=[consulta_requisitos_mantenimiento],
        llm=llm, allow_delegation=False, verbose=True,
    )
    validador = Agent(
        role="Ingeniero Validador",
        goal="Validar que calibres y torques cumplan rangos seguros antes de aprobar.",
        backstory="Ingeniero de confiabilidad. Ninguna cifra pasa sin verificarse.",
        tools=[validar_parametro_tecnico],
        llm=llm, allow_delegation=False, verbose=True,
    )
    redactor = Agent(
        role="Redactor Tecnico",
        goal="Redactar y emitir el plan final con subtareas priorizadas y respaldo.",
        backstory="Convierte el trabajo del equipo en una OT clara y trazable.",
        tools=[generar_plan_mantenimiento],
        llm=llm, allow_delegation=False, verbose=True,
    )

    tarea = Task(
        description=(
            "Planifica el mantenimiento solicitado: '{orden}'.\n"
            "1. Descomponlo en subtareas y priorizalas (seguridad > inspeccion > ajuste > registro).\n"
            "2. Delega al Especialista Documental la consulta de requisitos.\n"
            "3. Delega al Ingeniero Validador la verificacion de calibres/torques.\n"
            "4. Delega al Redactor Tecnico la emision del plan final.\n"
            "Si algo no tiene respaldo o resulta NO APTO, marcalo y ajusta el plan."
        ),
        expected_output=(
            "Un plan con subtareas priorizadas, cada una con respaldo documental y "
            "validacion tecnica, y la confirmacion de que el plan final fue generado."
        ),
        agent=manager,
    )

    return Crew(
        agents=[documental, validador, redactor],
        tasks=[tarea],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )


def describir_crew():
    print("=" * 60)
    print("EQUIPO DE ORQUESTACION JERARQUICA (CrewAI)")
    print("=" * 60)
    print("Proceso: Process.hierarchical (manager planifica y delega)\n")
    print("MANAGER (sin herramientas, allow_delegation=True):")
    print("  - Manager de Mantenimiento -> planifica y prioriza\n")
    print("ESPECIALISTAS:")
    print("  1. Especialista Documental -> consulta_requisitos_mantenimiento (RAG)")
    print("  2. Ingeniero Validador     -> validar_parametro_tecnico (razonamiento)")
    print("  3. Redactor Tecnico        -> generar_plan_mantenimiento (escritura)")


if __name__ == "__main__":
    if "--dry-run" in sys.argv:
        describir_crew()
        sys.exit(0)
    from agente_funcional import construir_retriever
    print("[1] Indexando documentacion...")
    retriever = construir_retriever(["doc_demo.txt"])
    print("[2] Construyendo Crew jerarquico...")
    crew = construir_crew(retriever)
    orden = ("Inspeccion de 100 horas del cableado del componente de avionica: "
             "revisar empalmes, validar AWG 16 para 18 A y torque del "
             "terminal_anillo_10 a 22 lb-in.")
    print(f"[3] Ejecutando orquestacion:\n    {orden}\n")
    resultado = crew.kickoff(inputs={"orden": orden})
    print("\n" + "=" * 60)
    print("RESULTADO DE LA ORQUESTACION:")
    print("=" * 60)
    print(resultado)
