"""
Bateria de escenarios con variabilidad de datos (RA3)
=====================================================
Genera un flujo de ordenes de mantenimiento para ejercitar al agente en
condiciones diversas: casos nominales, calibres insuficientes, torques fuera
de rango, parametros no tabulados y ordenes sin respaldo documental.

Cada escenario incluye la VERDAD DE TERRENO (veredicto esperado), lo que permite
medir precision (IE1) comparando lo observado contra lo esperado.

Las tablas espejan las de RA2/agente-funcional/herramientas.py. En modo real, el
harness llama a la herramienta original; estas tablas son la referencia de verdad.
"""

from __future__ import annotations
import random

# Espejo de _TABLA_AWG (capacidad de corriente en amperios por calibre)
TABLA_AWG = {20: 11.0, 18: 16.0, 16: 22.0, 14: 32.0, 12: 41.0, 10: 55.0, 8: 73.0, 6: 101.0}
# Espejo de _RANGO_TORQUE (lb-in) por terminal
RANGO_TORQUE = {
    "terminal_anillo_10": (20, 25),
    "terminal_anillo_1/4": (50, 70),
    "conector_circular": (15, 20),
}

CATEGORIAS = [
    "nominal",            # todo apto, con documentacion
    "awg_insuficiente",   # el calibre no soporta la corriente -> NO_APTO
    "torque_fuera_rango", # torque fuera del rango del terminal -> NO_APTO
    "no_tabulado",        # AWG/terminal inexistente -> ERROR de parametro
    "sin_documentacion",  # RAG no devuelve respaldo -> subtarea bloqueada
    "repetido",           # orden identica repetida -> mide consistencia (IE1)
]


def _veredicto_calibre(awg: int, corriente: float) -> str:
    if awg not in TABLA_AWG:
        return "ERROR"
    return "APTO" if corriente <= TABLA_AWG[awg] else "NO_APTO"


def _veredicto_torque(valor: float, terminal: str) -> str:
    if terminal not in RANGO_TORQUE:
        return "ERROR"
    lo, hi = RANGO_TORQUE[terminal]
    return "APTO" if lo <= valor <= hi else "NO_APTO"


def generar_bateria(n_por_categoria: int = 8, semilla: int = 42) -> list[dict]:
    """Construye una bateria reproducible de escenarios variados."""
    rng = random.Random(semilla)
    escenarios: list[dict] = []
    awgs = list(TABLA_AWG)
    terminales = list(RANGO_TORQUE)

    def base(idx, cat, awg, corriente, torque, terminal, doc):
        return {
            "id": f"{cat[:3].upper()}-{idx:02d}",
            "categoria": cat,
            "orden": (f"Planifica inspeccion de cableado: validar AWG {awg} para "
                      f"{corriente} A y torque de {torque} lb-in en {terminal}."),
            "awg": awg, "corriente": corriente,
            "torque": torque, "terminal": terminal,
            "doc_disponible": doc,
            "verdad_calibre": _veredicto_calibre(awg, corriente),
            "verdad_torque": _veredicto_torque(torque, terminal),
        }

    # nominal: apto y con documentacion
    for i in range(n_por_categoria):
        awg = rng.choice(awgs)
        corriente = round(TABLA_AWG[awg] * rng.uniform(0.4, 0.85), 1)
        terminal = rng.choice(terminales)
        lo, hi = RANGO_TORQUE[terminal]
        torque = round(rng.uniform(lo, hi), 1)
        escenarios.append(base(i, "nominal", awg, corriente, torque, terminal, True))

    # awg_insuficiente
    for i in range(n_por_categoria):
        awg = rng.choice(awgs)
        corriente = round(TABLA_AWG[awg] * rng.uniform(1.1, 1.8), 1)
        terminal = rng.choice(terminales)
        lo, hi = RANGO_TORQUE[terminal]
        torque = round(rng.uniform(lo, hi), 1)
        escenarios.append(base(i, "awg_insuficiente", awg, corriente, torque, terminal, True))

    # torque_fuera_rango
    for i in range(n_por_categoria):
        awg = rng.choice(awgs)
        corriente = round(TABLA_AWG[awg] * rng.uniform(0.4, 0.8), 1)
        terminal = rng.choice(terminales)
        lo, hi = RANGO_TORQUE[terminal]
        torque = round(hi + rng.uniform(3, 15), 1)
        escenarios.append(base(i, "torque_fuera_rango", awg, corriente, torque, terminal, True))

    # no_tabulado (AWG inexistente)
    for i in range(n_por_categoria):
        awg = rng.choice([22, 24, 4, 2])  # fuera de la tabla
        corriente = round(rng.uniform(5, 30), 1)
        terminal = rng.choice(terminales)
        lo, hi = RANGO_TORQUE[terminal]
        torque = round(rng.uniform(lo, hi), 1)
        escenarios.append(base(i, "no_tabulado", awg, corriente, torque, terminal, True))

    # sin_documentacion (RAG miss)
    for i in range(n_por_categoria):
        awg = rng.choice(awgs)
        corriente = round(TABLA_AWG[awg] * rng.uniform(0.4, 0.8), 1)
        terminal = rng.choice(terminales)
        lo, hi = RANGO_TORQUE[terminal]
        torque = round(rng.uniform(lo, hi), 1)
        escenarios.append(base(i, "sin_documentacion", awg, corriente, torque, terminal, False))

    # repetido: misma orden N veces para medir consistencia
    plantilla = base(0, "repetido", 16, 18.0, 22.0, "terminal_anillo_10", True)
    for i in range(n_por_categoria):
        rep = dict(plantilla)
        rep["id"] = f"REP-{i:02d}"
        escenarios.append(rep)

    rng.shuffle(escenarios)
    return escenarios


if __name__ == "__main__":
    bateria = generar_bateria()
    print(f"{len(bateria)} escenarios generados")
    for e in bateria[:5]:
        print(e["id"], e["categoria"], "->", e["verdad_calibre"], e["verdad_torque"])
