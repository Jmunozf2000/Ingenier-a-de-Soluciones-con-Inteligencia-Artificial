"""
Analisis de trazabilidad y deteccion de anomalias (RA3 — IE3, IE4)
==================================================================
Lee la traza estructurada (JSONL) producida por el harness y:

  IE3  Examina logs y eventos: latencia por herramienta, identifica el cuello
       de botella y la concentracion de errores.
  IE4  Detecta patrones y anomalias: outliers de latencia (regla IQR),
       correlacion sin_documentacion -> plan_bloqueado, precision por categoria.

Salidas:
  evidencia/hallazgos.md         informe de hallazgos en Markdown
  evidencia/resumen_herramientas.csv
  evidencia/resumen_categorias.csv
"""

from __future__ import annotations
import json
import pandas as pd

RUTA_TRAZA = "evidencia/traza_ejecucion.jsonl"


def cargar(ruta: str = RUTA_TRAZA) -> pd.DataFrame:
    filas = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                filas.append(json.loads(linea))
    return pd.DataFrame(filas)


def analizar(df: pd.DataFrame) -> dict:
    r = {}

    # --- IE3: latencia por herramienta y cuello de botella -------------------
    pasos = df[df["tipo"].isin(["herramienta", "paso", "validacion"])].copy()
    pasos["herramienta"] = pasos["herramienta"].fillna(
        pasos.get("subtipo").map(lambda s: f"validar_{s}" if isinstance(s, str) else None)
    )
    lat = (pasos.groupby("herramienta")["duracion"]
           .agg(n="count", media="mean",
                p50=lambda s: s.quantile(0.50),
                p95=lambda s: s.quantile(0.95),
                maximo="max")
           .sort_values("p95", ascending=False).round(4))
    r["latencia_herramientas"] = lat
    r["cuello_botella"] = lat.index[0]

    # --- IE3: concentracion de errores ---------------------------------------
    inter = df[df["tipo"] == "interaccion"].copy()
    tipos_error = inter["tipos_error"].explode().dropna()
    r["errores_por_tipo"] = tipos_error.value_counts()

    # --- IE1 por categoria: precision y tasa de bloqueo ----------------------
    val = df[df["tipo"] == "validacion"].copy()
    val["correcto"] = val["correcto"].astype(bool)
    prec_cat = val.groupby("categoria")["correcto"].mean().round(4)
    bloqueo_cat = (inter.assign(bloq=inter["resultado"].eq("bloqueado"))
                   .groupby("categoria")["bloq"].mean().round(4))
    lat_cat = inter.groupby("categoria")["latencia_total"].mean().round(4)
    r["por_categoria"] = pd.DataFrame({
        "precision": prec_cat, "tasa_bloqueo": bloqueo_cat, "latencia_media": lat_cat
    })

    # --- IE4: anomalias de latencia (regla IQR sobre latencia total) ---------
    q1, q3 = inter["latencia_total"].quantile([0.25, 0.75])
    iqr = q3 - q1
    umbral = q3 + 1.5 * iqr
    anomalias = inter[inter["latencia_total"] > umbral]
    r["umbral_anomalia"] = round(umbral, 4)
    r["n_anomalias"] = len(anomalias)
    r["pct_anomalias"] = round(100 * len(anomalias) / max(1, len(inter)), 2)
    r["anomalias_top"] = (anomalias.nlargest(5, "latencia_total")
                          [["escenario", "categoria", "latencia_total"]])

    # --- IE4: correlacion sin_documentacion -> bloqueo -----------------------
    sindoc = inter[inter["categoria"] == "sin_documentacion"]
    r["sindoc_bloqueo"] = round(sindoc["resultado"].eq("bloqueado").mean(), 4) if len(sindoc) else 0.0

    # --- Ruido de enrutamiento (errores del LLM detectados) ------------------
    if "ruido_argumento" in val.columns:
        ruido = val["ruido_argumento"].fillna(False).astype(bool)
        incorrecto = ~val["correcto"]
        r["fallos_por_ruido_llm"] = int((ruido & incorrecto).sum())
        r["total_incorrectos"] = int(incorrecto.sum())

    r["totales"] = {
        "eventos": len(df),
        "interacciones": len(inter),
        "completadas": int(inter["resultado"].eq("completado").sum()),
        "bloqueadas": int(inter["resultado"].eq("bloqueado").sum()),
        "cpu_media": round(inter["cpu"].mean(), 1),
        "mem_media_mb": round(inter["mem_mb"].mean(), 1),
    }
    return r


def escribir_hallazgos(r: dict, ruta_md: str = "evidencia/hallazgos.md") -> None:
    t = r["totales"]
    lat = r["latencia_herramientas"]
    lineas = [
        "# Hallazgos de trazabilidad y anomalias (RA3)",
        "",
        f"- Eventos analizados: **{t['eventos']}** | Interacciones: **{t['interacciones']}** "
        f"(completadas {t['completadas']}, bloqueadas {t['bloqueadas']}).",
        f"- Uso de recursos promedio: CPU {t['cpu_media']}% | Memoria {t['mem_media_mb']} MB.",
        "",
        "## IE3 — Latencia por herramienta y cuello de botella",
        "",
        f"El cuello de botella es **{r['cuello_botella']}** (mayor p95).",
        "",
        lat.to_markdown(),
        "",
        "## IE3 — Concentracion de errores",
        "",
        r["errores_por_tipo"].to_frame("ocurrencias").to_markdown(),
        "",
        "## IE4 — Anomalias de latencia (regla IQR)",
        "",
        f"Umbral de anomalia (Q3 + 1.5·IQR): **{r['umbral_anomalia']} s**. "
        f"Se detectaron **{r['n_anomalias']}** interacciones anomalas "
        f"({r['pct_anomalias']}% del total).",
        "",
        r["anomalias_top"].to_markdown(index=False),
        "",
        "## IE4 — Patrones y correlaciones",
        "",
        f"- Correlacion **sin_documentacion -> plan_bloqueado**: "
        f"{r['sindoc_bloqueo']*100:.0f}% de las ordenes sin respaldo terminan bloqueadas "
        "(comportamiento seguro esperado: el agente no inventa datos).",
    ]
    if "fallos_por_ruido_llm" in r:
        lineas += [
            f"- De **{r['total_incorrectos']}** validaciones incorrectas, "
            f"**{r['fallos_por_ruido_llm']}** se explican por error de enrutamiento de "
            "argumentos del LLM (parametro mal pasado a la herramienta).",
        ]
    lineas += [
        "",
        "## Precision y bloqueo por categoria",
        "",
        r["por_categoria"].to_markdown(),
        "",
    ]
    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    r["latencia_herramientas"].to_csv("evidencia/resumen_herramientas.csv")
    r["por_categoria"].to_csv("evidencia/resumen_categorias.csv")


if __name__ == "__main__":
    df = cargar()
    r = analizar(df)
    escribir_hallazgos(r)
    print(f"[analisis] Cuello de botella: {r['cuello_botella']}")
    print(f"[analisis] Anomalias de latencia: {r['n_anomalias']} ({r['pct_anomalias']}%)")
    print(f"[analisis] sin_documentacion -> bloqueo: {r['sindoc_bloqueo']*100:.0f}%")
    print("[analisis] Salidas: evidencia/hallazgos.md, resumen_herramientas.csv, resumen_categorias.csv")
