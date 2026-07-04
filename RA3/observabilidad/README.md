# RA3 — Observabilidad del Agente Planificador de Mantenimiento

Capa **aditiva** de observabilidad, trazabilidad y monitoreo sobre el agente
funcional de la RA2 (`RA2/agente-funcional/`). No modifica el agente original:
lo instrumenta, ejecuta una batería de escenarios con variabilidad de datos,
recolecta métricas (Prometheus), analiza la traza y las visualiza en Grafana.

## Contenido

| Archivo | Rol | Indicadores |
|---|---|---|
| `instrumentacion.py` | Métricas Prometheus + traza JSONL + recursos (psutil) | IE1, IE2, IE3, IE4 |
| `escenarios.py` | Batería de escenarios con verdad de terreno | IE1 |
| `harness_benchmark.py` | Ejecuta la batería instrumentada (modos `sim` / `real`) | IE1, IE2 |
| `analisis_trazabilidad.py` | Cuellos de botella, errores y anomalías (IQR) | IE3, IE4 |
| `seguridad.py` | Redacción de secretos/PII y guardrails de entrada | IE6 |
| `generar_figuras.py` | Figuras de evidencia a partir de la traza | IE8 |
| `monitoring/` | Stack Docker: Prometheus + Grafana + dashboard | IE5 |

## Requisitos

```bash
pip install -r requirements.txt      # capa de observabilidad
# Docker + Docker Compose para el stack de monitoreo
```

## 1) Generar métricas y evidencia (sin API, reproducible)

El modo `sim` reproduce el ciclo de planificación del agente usando la lógica
**real** de validación de la RA2 (si `langchain` está instalado; si no, un espejo
autónomo), con latencias realistas y un modelo de error de enrutamiento de
argumentos del LLM. No requiere `GITHUB_TOKEN`.

```bash
# corrida rápida que genera traza + resumen + análisis + figuras
python harness_benchmark.py --modo sim --repeticiones 6 --rapido
python analisis_trazabilidad.py
python generar_figuras.py
```

Salidas en `evidencia/`:
- `traza_ejecucion.jsonl` — un evento por línea (insumo de IE3/IE4)
- `resumen_corrida.json` — precisión, consistencia, tasa de error, bloqueados
- `hallazgos.md`, `resumen_herramientas.csv`, `resumen_categorias.csv`
- `figuras/*.png`

## 2) Dashboard en Grafana (monitoreo en vivo)

```bash
# a) Levanta Prometheus (9090) y Grafana (3000)
cd monitoring
docker compose up -d

# b) En otra terminal, alimenta métricas en vivo desde el host
cd ..
python harness_benchmark.py --modo sim --puerto 8000 --continuo
```

- Grafana: <http://localhost:3000>  (usuario `admin` / clave `admin`)
- El dashboard **"Observabilidad — Agente Planificador de Mantenimiento (RA3)"**
  aparece provisionado en la carpeta *Agente Avionica*.
- Prometheus scrapea `host.docker.internal:8000` cada 5 s.

> Para capturas del informe: deja la corrida `--continuo` unos minutos y toma las
> capturas de las tres filas del dashboard (KPIs, Latencia/recursos, Errores).

## 3) Ejecución real (opcional, con el LLM)

```bash
python harness_benchmark.py --modo real --puerto 8000 --continuo
```

Requiere `.env` con `GITHUB_TOKEN` y las dependencias de
`RA2/agente-funcional/requirements.txt`. En este modo la precisión se mide sobre
las decisiones reales del agente.

## Métricas expuestas (`/metrics`)

`agente_interaccion_latencia_segundos` · `agente_herramienta_latencia_segundos{herramienta}`
· `agente_herramienta_invocaciones_total{herramienta}` · `agente_validaciones_total{resultado}`
· `agente_rag_consultas_total{resultado}` · `agente_errores_total{tipo}`
· `agente_tokens_estimados_total` · `agente_interacciones_total{resultado}`
· `agente_precision_ratio` · `agente_consistencia_ratio` · `agente_tasa_error_ratio`
· `agente_cpu_porcentaje` · `agente_memoria_mb`

## Seguridad y uso responsable (IE6)

- Secretos vía `.env` (ignorado por git); `seguridad.redactar()` enmascara tokens
  y PII (correos, matrículas `CC-XXX`, números de serie) antes de logs/métricas.
- `seguridad.validar_entrada()` bloquea órdenes vacías, sobredimensionadas o con
  patrones de inyección de instrucciones.
- El agente de la RA2 no inventa datos sin respaldo: las órdenes sin documentación
  se bloquean (verificable en la correlación *sin_documentacion → plan_bloqueado*).
