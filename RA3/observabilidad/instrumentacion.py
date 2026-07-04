"""
Instrumentacion de Observabilidad — Agente Planificador de Mantenimiento (RA3)
==============================================================================
Capa ADITIVA de observabilidad para el agente funcional de la RA2. No modifica
el agente original: expone metricas (Prometheus), traza estructurada (JSONL) y
muestreo de recursos (psutil).

Cubre:
  IE1  precision, consistencia, frecuencia de errores
  IE2  latencia y uso de recursos
  IE3  eventos y latencias por herramienta (para deteccion de cuellos de botella)
  IE4  registros base para deteccion de patrones/anomalias

Modos de exposicion:
  - Servidor /metrics para scrape de Prometheus  ->  iniciar_servidor_metricas(port)
  - Push a Pushgateway (jobs batch)              ->  empujar_metricas(gateway, job)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from prometheus_client import (
    Counter, Gauge, Histogram, CollectorRegistry,
    start_http_server, push_to_gateway,
)

try:
    import psutil  # muestreo real de CPU/memoria
    _PROC = psutil.Process(os.getpid())
except Exception:  # pragma: no cover
    psutil = None
    _PROC = None

# ---------------------------------------------------------------------------
# Registro de metricas (un unico registry para poder empujar a Pushgateway)
# ---------------------------------------------------------------------------
REGISTRO = CollectorRegistry()

# --- IE2: latencia -----------------------------------------------------------
LATENCIA_INTERACCION = Histogram(
    "agente_interaccion_latencia_segundos",
    "Latencia end-to-end de una interaccion (orden -> plan).",
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21),
    registry=REGISTRO,
)
LATENCIA_HERRAMIENTA = Histogram(
    "agente_herramienta_latencia_segundos",
    "Latencia por invocacion de herramienta.",
    ["herramienta"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5),
    registry=REGISTRO,
)

# --- IE1 / IE3: invocaciones, validaciones, errores --------------------------
INVOCACIONES = Counter(
    "agente_herramienta_invocaciones_total",
    "Invocaciones por herramienta.",
    ["herramienta"],
    registry=REGISTRO,
)
VALIDACIONES = Counter(
    "agente_validaciones_total",
    "Resultados de validar_parametro_tecnico.",
    ["resultado"],  # APTO | NO_APTO | ERROR
    registry=REGISTRO,
)
RAG_CONSULTAS = Counter(
    "agente_rag_consultas_total",
    "Consultas RAG por resultado.",
    ["resultado"],  # HIT | MISS
    registry=REGISTRO,
)
ERRORES = Counter(
    "agente_errores_total",
    "Errores por tipo.",
    ["tipo"],  # parametro_no_tabulado | sin_documentacion | plan_bloqueado | excepcion
    registry=REGISTRO,
)
TOKENS = Counter(
    "agente_tokens_estimados_total",
    "Tokens estimados consumidos (aprox. 4 chars/token).",
    registry=REGISTRO,
)
INTERACCIONES = Counter(
    "agente_interacciones_total",
    "Interacciones procesadas por resultado.",
    ["resultado"],  # completado | bloqueado
    registry=REGISTRO,
)

# --- IE1: precision y consistencia (gauges de estado agregado) ---------------
PRECISION = Gauge(
    "agente_precision_ratio",
    "Fraccion de validaciones cuyo veredicto coincide con la verdad de terreno.",
    registry=REGISTRO,
)
CONSISTENCIA = Gauge(
    "agente_consistencia_ratio",
    "Fraccion de ordenes repetidas que producen el mismo veredicto.",
    registry=REGISTRO,
)
TASA_ERROR = Gauge(
    "agente_tasa_error_ratio",
    "Fraccion de interacciones con al menos un error.",
    registry=REGISTRO,
)

# --- IE2: uso de recursos ----------------------------------------------------
CPU = Gauge("agente_cpu_porcentaje", "Uso de CPU del proceso (%).", registry=REGISTRO)
MEM = Gauge("agente_memoria_mb", "Memoria RSS del proceso (MB).", registry=REGISTRO)


# ---------------------------------------------------------------------------
# Traza estructurada (JSONL)  ->  insumo de IE3 / IE4
# ---------------------------------------------------------------------------
class TrazaJSONL:
    """Escribe un evento por linea en formato JSON. Cada evento es autocontenido
    y trazable (timestamp UTC, escenario, herramienta, latencia, resultado)."""

    def __init__(self, ruta: str = "evidencia/traza_ejecucion.jsonl"):
        self.ruta = ruta
        os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
        # trunca al iniciar una corrida nueva
        open(self.ruta, "w", encoding="utf-8").close()

    def evento(self, **campos) -> None:
        campos.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="milliseconds"))
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(campos, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Utilidades de medicion
# ---------------------------------------------------------------------------
@contextmanager
def medir_herramienta(nombre: str):
    """Context manager que cronometra una herramienta y actualiza sus metricas."""
    INVOCACIONES.labels(herramienta=nombre).inc()
    inicio = time.perf_counter()
    try:
        yield
    finally:
        LATENCIA_HERRAMIENTA.labels(herramienta=nombre).observe(time.perf_counter() - inicio)


def muestrear_recursos() -> dict:
    """Actualiza los gauges de CPU/memoria y devuelve el snapshot."""
    if _PROC is None:
        return {"cpu": 0.0, "mem_mb": 0.0}
    cpu = _PROC.cpu_percent(interval=None)
    mem_mb = _PROC.memory_info().rss / (1024 * 1024)
    CPU.set(cpu)
    MEM.set(mem_mb)
    return {"cpu": round(cpu, 1), "mem_mb": round(mem_mb, 1)}


def estimar_tokens(texto: str) -> int:
    """Estimacion barata de tokens (~4 caracteres por token)."""
    n = max(1, len(texto) // 4)
    TOKENS.inc(n)
    return n


def iniciar_servidor_metricas(port: int = 8000) -> None:
    """Expone /metrics para que Prometheus lo scrapee."""
    start_http_server(port, registry=REGISTRO)
    print(f"[obs] Servidor de metricas en http://localhost:{port}/metrics")


def empujar_metricas(gateway: str, job: str = "agente_avionica") -> None:
    """Empuja el estado actual del registro a un Pushgateway (util para jobs batch)."""
    push_to_gateway(gateway, job=job, registry=REGISTRO)
    print(f"[obs] Metricas empujadas a {gateway} (job={job})")
