"""
Generacion de figuras de observabilidad (RA3 — IE8)
===================================================
Produce figuras a partir de la traza JSONL, como evidencia visual para el
informe y como vista previa de los paneles del dashboard de Grafana.

Salidas: evidencia/figuras/*.png
"""

from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RUTA = "evidencia/traza_ejecucion.jsonl"
DIR = "evidencia/figuras"

# Paleta sobria (coherente con el estilo tecnico del proyecto)
AZUL, VERDE, AMBAR, ROJO, GRIS = "#1f4e79", "#2e8b57", "#d98c00", "#b3402e", "#6b7280"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})


def cargar():
    filas = [json.loads(l) for l in open(RUTA, encoding="utf-8") if l.strip()]
    return pd.DataFrame(filas)


def fig_latencia_herramientas(df):
    pasos = df[df["tipo"].isin(["herramienta", "paso", "validacion"])].copy()
    pasos["herramienta"] = pasos["herramienta"].fillna(
        pasos.get("subtipo").map(lambda s: f"validar_{s}" if isinstance(s, str) else None))
    orden = pasos.groupby("herramienta")["duracion"].median().sort_values().index
    datos = [pasos[pasos["herramienta"] == h]["duracion"].values for h in orden]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bp = ax.boxplot(datos, vert=False, patch_artist=True, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for p in bp["boxes"]:
        p.set(facecolor=AZUL, alpha=0.55)
    for m in bp["medians"]:
        m.set(color=AMBAR, linewidth=2)
    ax.set_yticklabels(orden)
    ax.set_xlabel("Latencia por invocacion (s)")
    ax.set_title("Latencia por herramienta — cuello de botella: razonamiento LLM y RAG")
    fig.tight_layout(); fig.savefig(f"{DIR}/01_latencia_herramientas.png"); plt.close(fig)


def fig_invocaciones(df):
    pasos = df[df["tipo"].isin(["herramienta", "paso", "validacion"])].copy()
    pasos["herramienta"] = pasos["herramienta"].fillna(
        pasos.get("subtipo").map(lambda s: f"validar_{s}" if isinstance(s, str) else None))
    conteo = pasos["herramienta"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(conteo.index[::-1], conteo.values[::-1], color=AZUL, alpha=0.85)
    ax.set_xlabel("Invocaciones")
    ax.set_title("Invocaciones por herramienta")
    for i, v in enumerate(conteo.values[::-1]):
        ax.text(v + 2, i, str(v), va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{DIR}/02_invocaciones.png"); plt.close(fig)


def fig_bloqueo_categoria(df):
    inter = df[df["tipo"] == "interaccion"].copy()
    tasa = (inter.assign(b=inter["resultado"].eq("bloqueado"))
            .groupby("categoria")["b"].mean().sort_values())
    colores = [VERDE if v < 0.5 else ROJO for v in tasa.values]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.barh(tasa.index, tasa.values * 100, color=colores, alpha=0.85)
    ax.set_xlabel("% de planes bloqueados")
    ax.set_xlim(0, 105)
    ax.set_title("Tasa de bloqueo por categoria (seguridad: bloquear lo no-apto)")
    for i, v in enumerate(tasa.values * 100):
        ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{DIR}/03_bloqueo_categoria.png"); plt.close(fig)


def fig_veredictos(df):
    val = df[df["tipo"] == "validacion"]
    conteo = val["veredicto"].value_counts()
    colores = {"APTO": VERDE, "NO_APTO": AMBAR, "ERROR": ROJO}
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.pie(conteo.values, labels=conteo.index, autopct="%1.0f%%",
           colors=[colores.get(k, GRIS) for k in conteo.index],
           wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.set_title("Distribucion de veredictos de validacion")
    fig.tight_layout(); fig.savefig(f"{DIR}/04_veredictos.png"); plt.close(fig)


def fig_anomalias(df):
    inter = df[df["tipo"] == "interaccion"].reset_index(drop=True)
    q1, q3 = inter["latencia_total"].quantile([0.25, 0.75])
    umbral = q3 + 1.5 * (q3 - q1)
    normal = inter[inter["latencia_total"] <= umbral]
    anom = inter[inter["latencia_total"] > umbral]
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.scatter(normal.index, normal["latencia_total"], s=10, color=AZUL, alpha=0.5, label="normal")
    ax.scatter(anom.index, anom["latencia_total"], s=28, color=ROJO, alpha=0.9,
               label=f"anomalia (n={len(anom)})", zorder=3)
    ax.axhline(umbral, color=AMBAR, ls="--", lw=1.5, label=f"umbral IQR = {umbral:.2f}s")
    ax.set_xlabel("Interaccion (secuencia)")
    ax.set_ylabel("Latencia total (s)")
    ax.set_title("Deteccion de anomalias de latencia (regla IQR)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{DIR}/05_anomalias_latencia.png"); plt.close(fig)


def fig_kpis(df):
    val = df[df["tipo"] == "validacion"].copy()
    val["correcto"] = val["correcto"].astype(bool)
    prec = val["correcto"].mean()
    inter = df[df["tipo"] == "interaccion"]
    tasa_err = (inter["errores"] > 0).mean()
    completa = inter["resultado"].eq("completado").mean()
    kpis = {"Precision": prec, "Completadas": completa, "Tasa de error": tasa_err}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    cols = [VERDE, AZUL, ROJO]
    barras = ax.bar(list(kpis), [v * 100 for v in kpis.values()], color=cols, alpha=0.85, width=0.55)
    ax.set_ylabel("%"); ax.set_ylim(0, 100)
    ax.set_title("Indicadores agregados del agente")
    for b, v in zip(barras, kpis.values()):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 2, f"{v*100:.1f}%", ha="center", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{DIR}/06_kpis.png"); plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(DIR, exist_ok=True)
    df = cargar()
    for fn in (fig_latencia_herramientas, fig_invocaciones, fig_bloqueo_categoria,
               fig_veredictos, fig_anomalias, fig_kpis):
        fn(df)
        print(f"[fig] {fn.__name__} OK")
    print(f"[fig] Figuras en {DIR}/")
