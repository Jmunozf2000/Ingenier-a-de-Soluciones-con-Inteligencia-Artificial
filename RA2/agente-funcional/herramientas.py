import os
from typing import Union
from datetime import datetime
from langchain_core.tools import tool

_RETRIEVER = None


def set_retriever(retriever):
    global _RETRIEVER
    _RETRIEVER = retriever


@tool
def consulta_requisitos_mantenimiento(tarea: str) -> str:
    """Busca en la documentacion tecnica los REQUISITOS de una tarea de
    mantenimiento: intervalos, herramientas, torque, calibres, materiales
    y procedimientos. Fundamenta cada subtarea con respaldo documental.

    Args:
        tarea: Descripcion de la subtarea de mantenimiento a consultar.
    """
    if _RETRIEVER is None:
        return "[ERROR] El retriever no esta inicializado."
    docs = _RETRIEVER.invoke(tarea)
    if not docs:
        return ("No se encontro documentacion que respalde esta tarea. "
                "Verifique con su supervisor o la fuente oficial.")
    fragmentos = []
    for doc in docs:
        fuente = doc.metadata.get("source", "Fuente desconocida")
        pagina = doc.metadata.get("page", "s/p")
        fragmentos.append(f"[Fuente: {fuente} | Pagina: {pagina}]\n{doc.page_content}")
    return "\n\n---\n\n".join(fragmentos)


_TABLA_AWG = {20: 11.0, 18: 16.0, 16: 22.0, 14: 32.0, 12: 41.0, 10: 55.0, 8: 73.0, 6: 101.0}
_RANGO_TORQUE = {
    "terminal_anillo_10": (20, 25),
    "terminal_anillo_1/4": (50, 70),
    "conector_circular": (15, 20),
}


@tool
def validar_parametro_tecnico(tipo: str, valor: float, referencia: Union[str, float, int] = "") -> str:
    """Valida un parametro tecnico. tipo='calibre': si un AWG (valor) soporta
    la corriente en 'referencia' (en amperios, numero). tipo='torque': si un
    valor (lb-in) esta en el rango del terminal en 'referencia' (nombre del terminal).

    Args:
        tipo: 'calibre' o 'torque'.
        valor: AWG para calibre, lb-in para torque.
        referencia: corriente en amperios (calibre) o nombre del terminal (torque).
    """
    tipo = str(tipo).lower().strip()
    referencia = str(referencia)
    if tipo == "calibre":
        awg = int(valor)
        if awg not in _TABLA_AWG:
            disp = ", ".join(str(k) for k in sorted(_TABLA_AWG))
            return f"[ERROR] Calibre AWG {awg} no tabulado. Disponibles: {disp}."
        capacidad = _TABLA_AWG[awg]
        try:
            corriente = float(referencia)
        except (ValueError, TypeError):
            return "[ERROR] Para 'calibre' indique la corriente (A) en 'referencia'."
        if corriente <= capacidad:
            margen = round((capacidad - corriente) / capacidad * 100, 1)
            return f"[APTO] AWG {awg} soporta {capacidad} A para {corriente} A (margen {margen}%)."
        return (f"[NO APTO] AWG {awg} soporta solo {capacidad} A, insuficiente para "
                f"{corriente} A. Use un calibre menor (mas grueso).")
    if tipo == "torque":
        if referencia not in _RANGO_TORQUE:
            disp = ", ".join(_RANGO_TORQUE)
            return f"[ERROR] Terminal '{referencia}' no tabulado. Disponibles: {disp}."
        minimo, maximo = _RANGO_TORQUE[referencia]
        if minimo <= valor <= maximo:
            return f"[APTO] Torque {valor} lb-in dentro del rango [{minimo}, {maximo}] para {referencia}."
        return (f"[NO APTO] Torque {valor} lb-in fuera del rango [{minimo}, {maximo}] "
                f"para {referencia}. Ajuste al rango especificado.")
    return "[ERROR] tipo debe ser 'calibre' o 'torque'."


@tool
def generar_plan_mantenimiento(titulo: str, subtareas_priorizadas: str, fuente: str) -> str:
    """Genera y guarda un Plan de Mantenimiento (checklist/OT) en un archivo.
    Usar al final, con las subtareas ya priorizadas, fundamentadas y validadas.

    Args:
        titulo: Titulo de la orden de mantenimiento.
        subtareas_priorizadas: Lista ordenada de subtareas por prioridad.
        fuente: Cita(s) de la documentacion que respalda el plan.
    """
    carpeta = "planes_mantenimiento"
    os.makedirs(carpeta, exist_ok=True)
    plan_id = datetime.now().strftime("PLAN-%Y%m%d-%H%M%S")
    contenido = (
        f"=========================================\n"
        f"PLAN DE MANTENIMIENTO {plan_id}\n"
        f"Taller de Avionica - LATAM Airlines\n"
        f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"=========================================\n\n"
        f"ORDEN: {titulo}\n\n"
        f"SUBTAREAS PRIORIZADAS:\n{subtareas_priorizadas}\n\n"
        f"RESPALDO DOCUMENTAL:\n{fuente}\n"
        f"=========================================\n"
    )
    ruta = os.path.join(carpeta, f"{plan_id}.txt")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(contenido)
    return f"[OK] Plan de mantenimiento generado: {ruta}\n\n{contenido}"


HERRAMIENTAS = [
    consulta_requisitos_mantenimiento,
    validar_parametro_tecnico,
    generar_plan_mantenimiento,
]
