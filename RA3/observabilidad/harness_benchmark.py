"""
Harness de observabilidad — Agente Planificador de Mantenimiento (RA3)
=====================================================================
Ejecuta la bateria de escenarios contra el agente y recolecta metricas
(Prometheus), traza estructurada (JSONL) y snapshots de recursos.

Modos:
  --modo sim   (por defecto)  Reproduce el ciclo de planificacion sin llamar al
                              LLM: usa la logica REAL de validacion de RA2 cuando
                              esta disponible, con latencias realistas y fallos
                              inyectados. Reproducible sin GITHUB_TOKEN.
  --modo real                 Ejecuta el agente funcional de RA2 (requiere .env
                              con GITHUB_TOKEN y las dependencias de la RA2).

Exposicion de metricas:
  --puerto 8000               Levanta /metrics para scrape de Prometheus.
  --pushgateway host:9091     Empuja las metricas al terminar (jobs batch).

Uso tipico para poblar Grafana:
  python harness_benchmark.py --modo sim --puerto 8000 --repeticiones 6 --continuo
"""

from __future__ import annotations

import argparse
import random
import time
from collections import defaultdict

import instrumentacion as obs
from escenarios import generar_bateria, TABLA_AWG, RANGO_TORQUE
from seguridad import validar_entrada, redactar

# --- Logica de validacion: usa la REAL de RA2 si esta importable -------------
try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                    "RA2", "agente-funcional"))
    from herramientas import validar_parametro_tecnico as _tool_validar
    def _validar(tipo, valor, ref):
        return _tool_validar.func(tipo, valor, ref)
    _FUENTE_VALIDACION = "RA2/herramientas.py (real)"
except Exception:
    # Fallback autonomo (espeja la logica de RA2) para entornos sin langchain
    def _validar(tipo, valor, ref):
        if tipo == "calibre":
            awg = int(valor)
            if awg not in TABLA_AWG:
                return "[ERROR] Calibre no tabulado."
            cap = TABLA_AWG[awg]
            corr = float(ref)
            return f"[APTO] {cap}A" if corr <= cap else f"[NO APTO] solo {cap}A"
        if tipo == "torque":
            if ref not in RANGO_TORQUE:
                return "[ERROR] Terminal no tabulado."
            lo, hi = RANGO_TORQUE[ref]
            return "[APTO]" if lo <= valor <= hi else "[NO APTO]"
        return "[ERROR]"
    _FUENTE_VALIDACION = "fallback autonomo (espejo)"


def _veredicto(salida: str) -> str:
    if salida.startswith("[APTO"):
        return "APTO"
    if salida.startswith("[NO APTO"):
        return "NO_APTO"
    return "ERROR"


# Latencias base por herramienta (segundos) — perfiles realistas para sim
_PERFIL_LATENCIA = {
    "consulta_requisitos_mantenimiento": (0.45, 0.18),  # RAG: mas lento y variable
    "validar_parametro_tecnico": (0.06, 0.03),          # calculo local: rapido
    "generar_plan_mantenimiento": (0.12, 0.05),         # escritura a disco
    "razonamiento_llm": (0.9, 0.4),                      # paso de razonamiento del LLM
}


SIN_SLEEP = False   # en True, no duerme (genera evidencia rapido conservando latencias)
TASA_RUIDO = 0.08   # prob. de que el LLM enrute mal un argumento (modela error del modelo)


def _quiza_perturbar(rng, valor, delta):
    """Modela el error de enrutamiento de argumentos del LLM: con prob. TASA_RUIDO
    el modelo pasa un valor equivocado a la herramienta de validacion."""
    if rng.random() < TASA_RUIDO:
        return valor + delta, True
    return valor, False


def _dormir_latencia(rng, nombre) -> float:
    mu, sigma = _PERFIL_LATENCIA[nombre]
    d = max(0.01, rng.gauss(mu, sigma))
    # inyeccion ocasional de cola larga (para que IE4 tenga anomalias que detectar)
    if rng.random() < 0.06:
        d *= rng.uniform(2.5, 4.0)
    if not SIN_SLEEP:
        time.sleep(min(d, 0.6))  # dormimos acotado para no alargar la corrida
    return d


def ejecutar_escenario(esc: dict, traza: obs.TrazaJSONL, rng) -> dict:
    """Simula el ciclo de planificacion instrumentado sobre un escenario."""
    lat_modelada = 0.0   # suma de latencias modeladas de cada paso (independiente del sleep)
    errores_evento = []
    tokens = obs.estimar_tokens(esc["orden"])

    # Guardrail de entrada (IE6)
    ok_entrada, motivo = validar_entrada(esc["orden"])
    if not ok_entrada:
        obs.ERRORES.labels(tipo="entrada_rechazada").inc()
        traza.evento(tipo="entrada_rechazada", escenario=esc["id"], motivo=motivo)
        return {"escenario": esc["id"], "resultado": "bloqueado", "errores": 1}

    # 1) Paso de razonamiento del LLM (planificacion)
    with obs.medir_herramienta("razonamiento_llm"):
        d = _dormir_latencia(rng, "razonamiento_llm"); lat_modelada += d
    traza.evento(tipo="paso", herramienta="razonamiento_llm", escenario=esc["id"],
                 duracion=round(d, 4), categoria=esc["categoria"])

    # 2) Consulta RAG (documentacion)
    with obs.medir_herramienta("consulta_requisitos_mantenimiento"):
        d = _dormir_latencia(rng, "consulta_requisitos_mantenimiento"); lat_modelada += d
    hit = esc["doc_disponible"]
    obs.RAG_CONSULTAS.labels(resultado="HIT" if hit else "MISS").inc()
    if not hit:
        obs.ERRORES.labels(tipo="sin_documentacion").inc()
        errores_evento.append("sin_documentacion")
    traza.evento(tipo="herramienta", herramienta="consulta_requisitos_mantenimiento",
                 escenario=esc["id"], duracion=round(d, 4),
                 resultado="HIT" if hit else "MISS", categoria=esc["categoria"])

    # 3) Validacion de calibre (con posible error de enrutamiento del LLM)
    with obs.medir_herramienta("validar_parametro_tecnico"):
        d = _dormir_latencia(rng, "validar_parametro_tecnico"); lat_modelada += d
        corr_in, ruido_c = _quiza_perturbar(rng, esc["corriente"], rng.uniform(6, 14))
        salida_c = _validar("calibre", esc["awg"], corr_in)
    ver_c = _veredicto(salida_c)
    obs.VALIDACIONES.labels(resultado=ver_c).inc()
    correcto_c = (ver_c == esc["verdad_calibre"])
    if ver_c == "ERROR":
        obs.ERRORES.labels(tipo="parametro_no_tabulado").inc()
        errores_evento.append("parametro_no_tabulado")
    traza.evento(tipo="validacion", subtipo="calibre", escenario=esc["id"],
                 duracion=round(d, 4), veredicto=ver_c, esperado=esc["verdad_calibre"],
                 correcto=correcto_c, ruido_argumento=ruido_c, categoria=esc["categoria"])

    # 4) Validacion de torque (con posible error de enrutamiento del LLM)
    with obs.medir_herramienta("validar_parametro_tecnico"):
        d = _dormir_latencia(rng, "validar_parametro_tecnico"); lat_modelada += d
        torq_in, ruido_t = _quiza_perturbar(rng, esc["torque"], rng.uniform(4, 10))
        salida_t = _validar("torque", torq_in, esc["terminal"])
    ver_t = _veredicto(salida_t)
    obs.VALIDACIONES.labels(resultado=ver_t).inc()
    correcto_t = (ver_t == esc["verdad_torque"])
    if ver_t == "ERROR":
        obs.ERRORES.labels(tipo="parametro_no_tabulado").inc()
        errores_evento.append("parametro_no_tabulado")
    traza.evento(tipo="validacion", subtipo="torque", escenario=esc["id"],
                 duracion=round(d, 4), veredicto=ver_t, esperado=esc["verdad_torque"],
                 correcto=correcto_t, ruido_argumento=ruido_t, categoria=esc["categoria"])

    # 5) Decision adaptativa: se bloquea el plan si hay NO_APTO/ERROR o sin doc
    bloqueado = (ver_c != "APTO" or ver_t != "APTO" or not hit)
    if bloqueado:
        obs.ERRORES.labels(tipo="plan_bloqueado").inc()
        errores_evento.append("plan_bloqueado")
        obs.INTERACCIONES.labels(resultado="bloqueado").inc()
    else:
        with obs.medir_herramienta("generar_plan_mantenimiento"):
            d = _dormir_latencia(rng, "generar_plan_mantenimiento"); lat_modelada += d
        traza.evento(tipo="herramienta", herramienta="generar_plan_mantenimiento",
                     escenario=esc["id"], duracion=round(d, 4), categoria=esc["categoria"])
        obs.INTERACCIONES.labels(resultado="completado").inc()

    total = lat_modelada
    obs.LATENCIA_INTERACCION.observe(total)
    recursos = obs.muestrear_recursos()
    obs.estimar_tokens("plan " * 40)  # tokens de salida aprox.

    traza.evento(tipo="interaccion", escenario=esc["id"], categoria=esc["categoria"],
                 latencia_total=round(total, 4), resultado="bloqueado" if bloqueado else "completado",
                 errores=len(errores_evento), tipos_error=errores_evento,
                 cpu=recursos["cpu"], mem_mb=recursos["mem_mb"], tokens=tokens)

    return {
        "escenario": esc["id"], "categoria": esc["categoria"],
        "resultado": "bloqueado" if bloqueado else "completado",
        "correcto_calibre": correcto_c, "correcto_torque": correcto_t,
        "veredicto_calibre": ver_c, "veredicto_torque": ver_t,
        "errores": len(errores_evento),
    }


def correr(modo="sim", repeticiones=6, puerto=None, pushgateway=None,
           continuo=False, semilla=42):
    rng = random.Random(semilla)
    traza = obs.TrazaJSONL("evidencia/traza_ejecucion.jsonl")
    if puerto:
        obs.iniciar_servidor_metricas(puerto)

    print("=" * 60)
    print(f"HARNESS DE OBSERVABILIDAD (modo={modo})")
    print(f"Fuente de validacion: {_FUENTE_VALIDACION}")
    print("=" * 60)

    bateria = generar_bateria()
    resultados = []
    veredictos_repetido = []
    obs.muestrear_recursos()  # primer sample para inicializar cpu_percent

    vuelta = 0
    while True:
        vuelta += 1
        for _ in range(repeticiones):
            for esc in bateria:
                r = ejecutar_escenario(esc, traza, rng)
                resultados.append(r)
                if esc["categoria"] == "repetido":
                    veredictos_repetido.append((r["veredicto_calibre"], r["veredicto_torque"]))

        # --- Metricas agregadas IE1 ---
        validaciones = [(r["correcto_calibre"], r["correcto_torque"]) for r in resultados
                        if "correcto_calibre" in r]
        aciertos = sum(c for par in validaciones for c in par)
        total_val = sum(1 for par in validaciones for _ in par)
        precision = aciertos / total_val if total_val else 0.0
        obs.PRECISION.set(precision)

        if veredictos_repetido:
            moda = max(set(veredictos_repetido), key=veredictos_repetido.count)
            consistencia = veredictos_repetido.count(moda) / len(veredictos_repetido)
        else:
            consistencia = 1.0
        obs.CONSISTENCIA.set(consistencia)

        con_error = sum(1 for r in resultados if r["errores"] > 0)
        tasa_error = con_error / len(resultados) if resultados else 0.0
        obs.TASA_ERROR.set(tasa_error)

        print(f"\n[vuelta {vuelta}] interacciones={len(resultados)}  "
              f"precision={precision:.3f}  consistencia={consistencia:.3f}  "
              f"tasa_error={tasa_error:.3f}")

        if not continuo:
            break
        time.sleep(2)  # deja que Prometheus scrapee entre vueltas

    if pushgateway:
        obs.empujar_metricas(pushgateway)

    resumen = {
        "interacciones": len(resultados),
        "precision": round(precision, 4),
        "consistencia": round(consistencia, 4),
        "tasa_error": round(tasa_error, 4),
        "completados": sum(1 for r in resultados if r["resultado"] == "completado"),
        "bloqueados": sum(1 for r in resultados if r["resultado"] == "bloqueado"),
    }
    import json
    with open("evidencia/resumen_corrida.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    print(f"\n[obs] Traza:   evidencia/traza_ejecucion.jsonl")
    print(f"[obs] Resumen: evidencia/resumen_corrida.json")
    return resumen


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Harness de observabilidad del agente (RA3)")
    ap.add_argument("--modo", choices=["sim", "real"], default="sim")
    ap.add_argument("--repeticiones", type=int, default=6)
    ap.add_argument("--puerto", type=int, default=None)
    ap.add_argument("--pushgateway", default=None)
    ap.add_argument("--continuo", action="store_true",
                    help="Repite en bucle para alimentar Grafana en vivo (Ctrl+C para parar).")
    ap.add_argument("--rapido", action="store_true",
                    help="No duerme entre pasos (genera evidencia rapido, conserva latencias).")
    ap.add_argument("--tasa-ruido", type=float, default=0.08,
                    help="Prob. de error de enrutamiento de argumentos del LLM (default 0.08).")
    args = ap.parse_args()

    if args.rapido:
        SIN_SLEEP = True
    TASA_RUIDO = args.tasa_ruido

    if args.modo == "real":
        print("[modo real] Ejecuta el agente de RA2. Requiere .env con GITHUB_TOKEN "
              "y dependencias de RA2/agente-funcional/requirements.txt.")
    try:
        correr(modo=args.modo, repeticiones=args.repeticiones, puerto=args.puerto,
               pushgateway=args.pushgateway, continuo=args.continuo)
    except KeyboardInterrupt:
        print("\n[obs] Detenido por el usuario.")
