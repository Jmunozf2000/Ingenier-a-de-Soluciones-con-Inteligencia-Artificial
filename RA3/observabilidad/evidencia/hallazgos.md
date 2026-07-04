# Hallazgos de trazabilidad y anomalias (RA3)

- Eventos analizados: **1519** | Interacciones: **288** (completadas 79, bloqueadas 209).
- Uso de recursos promedio: CPU 107.7% | Memoria 24.9 MB.

## IE3 — Latencia por herramienta y cuello de botella

El cuello de botella es **razonamiento_llm** (mayor p95).

| herramienta                       |   n |   media |    p50 |    p95 |   maximo |
|:----------------------------------|----:|--------:|-------:|-------:|---------:|
| razonamiento_llm                  | 288 |  0.9083 | 0.8476 | 1.6788 |   4.5002 |
| consulta_requisitos_mantenimiento | 288 |  0.5315 | 0.4616 | 1.1781 |   2.4041 |
| generar_plan_mantenimiento        |  79 |  0.1267 | 0.1148 | 0.2061 |   0.6987 |
| validar_torque                    | 288 |  0.0701 | 0.0604 | 0.1265 |   0.433  |
| validar_calibre                   | 288 |  0.0657 | 0.062  | 0.1202 |   0.3376 |

## IE3 — Concentracion de errores

| tipos_error           |   ocurrencias |
|:----------------------|--------------:|
| plan_bloqueado        |           209 |
| sin_documentacion     |            48 |
| parametro_no_tabulado |            48 |

## IE4 — Anomalias de latencia (regla IQR)

Umbral de anomalia (Q3 + 1.5·IQR): **2.7772 s**. Se detectaron **15** interacciones anomalas (5.21% del total).

| escenario   | categoria          |   latencia_total |
|:------------|:-------------------|-----------------:|
| TOR-03      | torque_fuera_rango |           5.3928 |
| TOR-02      | torque_fuera_rango |           4.3258 |
| NO_-03      | no_tabulado        |           3.9099 |
| NOM-04      | nominal            |           3.6067 |
| REP-04      | repetido           |           3.5289 |

## IE4 — Patrones y correlaciones

- Correlacion **sin_documentacion -> plan_bloqueado**: 100% de las ordenes sin respaldo terminan bloqueadas (comportamiento seguro esperado: el agente no inventa datos).
- De **26** validaciones incorrectas, **26** se explican por error de enrutamiento de argumentos del LLM (parametro mal pasado a la herramienta).

## Precision y bloqueo por categoria

| categoria          |   precision |   tasa_bloqueo |   latencia_media |
|:-------------------|------------:|---------------:|-----------------:|
| awg_insuficiente   |      0.9583 |         1      |           1.4564 |
| no_tabulado        |      1      |         1      |           1.6522 |
| nominal            |      0.9375 |         0.125  |           1.676  |
| repetido           |      0.8854 |         0.2292 |           1.6202 |
| sin_documentacion  |      0.9479 |         1      |           1.4575 |
| torque_fuera_rango |      1      |         1      |           1.7995 |
