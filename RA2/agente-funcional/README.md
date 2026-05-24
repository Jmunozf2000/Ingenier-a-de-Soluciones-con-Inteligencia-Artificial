# 🛠️ Agente Funcional — Planificador de Mantenimiento

**Taller de Aviónica · LATAM Airlines**
ISY0101 — Ingeniería de Soluciones con IA · Evaluación Parcial 2

Este proyecto es la segunda parte del agente iniciado en la EP1. Si en la primera
evaluación construimos un agente RAG que respondía consultas técnicas citando su
fuente, ahora damos el salto: lo convertimos en un **agente funcional** que no solo
consulta, sino que **planifica, razona, decide y escribe**. En otras palabras, pasó
de "biblioteca que responde" a "colega que arma el plan de trabajo".

---

## 📌 ¿Qué hace este agente?

Recibe una **orden de mantenimiento de alto nivel** (por ejemplo: *"planifica la
inspección de cableado del componente X"*) y la transforma en un **plan ejecutable,
priorizado y respaldado por documentación**. Para lograrlo:

1. **Descompone** la orden en subtareas.
2. Las **prioriza** (la seguridad va primero, siempre).
3. **Consulta** la documentación técnica para fundamentar cada paso.
4. **Valida** parámetros críticos como calibres de cable y pares de apriete.
5. **Genera** el checklist / orden de trabajo final, listo para ejecutar.

Y lo bonito es que se adapta: si la documentación no respalda algo, no se lo inventa;
y si un parámetro técnico no es seguro, frena y lo marca en vez de seguir derecho.

---

## 🧩 Arquitectura general

![Diagrama de orquestación](evidencia/diagrama_orquestacion.png)

El sistema combina **dos frameworks** que se complementan:

| Capa | Tecnología | Para qué |
|------|-----------|----------|
| Agente individual | **LangChain** (`AgentExecutor` + function calling) | Razona y decide qué herramienta usar en cada momento |
| Orquestación de equipo | **CrewAI** (`Process.hierarchical`) | Un manager planifica y delega a especialistas |
| Memoria | `ConversationBufferWindowMemory` + JSON persistente | Mantiene el hilo dentro y entre sesiones |
| Recuperación semántica | **ChromaDB** + `text-embedding-3-small` | Encuentra el respaldo documental |
| Motor de razonamiento | **GitHub Models** (`gpt-4o-mini`) | El cerebro detrás de todo |

---

## 🔧 Componentes del proyecto

### 1. Herramientas (`herramientas.py`)
El agente tiene tres herramientas, una por cada tipo de acción que pide el encargo:

- **Consulta** — `consulta_requisitos_mantenimiento`: busca en la documentación
  indexada los requisitos de cada subtarea (recuperación semántica vía RAG).
- **Razonamiento** — `validar_parametro_tecnico`: valida calibres AWG contra la
  corriente requerida, y pares de apriete (torque) contra el rango seguro del terminal.
- **Escritura** — `generar_plan_mantenimiento`: redacta y guarda el plan/checklist
  final con sus subtareas priorizadas y la cita documental que lo respalda.

Cada herramienta tiene argumentos tipados y una descripción clara, para que el LLM
sepa cuándo conviene usar cada una (que es justo lo que recomienda la buena práctica).

### 2. Memoria (`memoria.py`)
Dos niveles, porque un buen asistente necesita memoria corta *y* larga:

- **Corto plazo** — una ventana deslizante (`k=5`) que recuerda las últimas
  interacciones del flujo actual sin gastar tokens de más.
- **Largo plazo** — persistencia en JSON que sobrevive entre sesiones. Así el agente
  "se acuerda" de planes anteriores y puede retomar tareas largas sin empezar de cero.

### 3. Agente principal (`agente_funcional.py`)
El corazón del sistema. Aquí vive el **esquema de planificación** (los 5 pasos:
descomponer → priorizar → fundamentar → validar → generar) y las **reglas de decisión
adaptativa** que hacen que el agente cambie de comportamiento según lo que va encontrando.

### 4. Orquestador jerárquico (`orquestador_crew.py`)
La versión "en equipo". Un **Manager** (que solo planifica y delega, sin tocar
herramientas) reparte el trabajo entre tres especialistas:

- 📚 **Especialista Documental** → consulta requisitos
- 🔬 **Ingeniero Validador** → valida calibres y torques
- ✍️ **Redactor Técnico** → emite el plan final

Esto sigue el patrón jerárquico de CrewAI y aporta flexibilidad, resiliencia y
especialización (si algo falla, el manager puede reorganizar el plan).

---

## ▶️ Cómo ejecutarlo

### Requisitos previos
- Python 3.10+
- Un token de **GitHub Models** (el mismo de la EP1)

### Instalación
```bash
# (recomendado) crea un entorno virtual para evitar choques de versiones
python -m venv .venv
source .venv/bin/activate   # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Variables de entorno
Crea un archivo `.env` en esta carpeta con:
```
GITHUB_TOKEN=tu_token_aqui
OPENAI_BASE_URL=https://models.inference.ai.azure.com
```

### Ejecución
```bash
# Agente individual (LangChain)
python agente_funcional.py

# Orquestador en equipo (CrewAI) — ejecución real
python orquestador_crew.py

# Inspeccionar la arquitectura del equipo SIN credenciales
python orquestador_crew.py --dry-run

# Correr las pruebas de componentes (no requieren token)
python pruebas/test_componentes.py
```

> 💡 **Ojo con las versiones:** LangChain 0.3 y CrewAI tienen preferencias distintas
> sobre algunas dependencias. Si los instalas juntos y pip se queja, lo más cómodo es
> usar el entorno virtual y, si hace falta, ejecutar el agente y el orquestador en
> entornos separados. En la práctica ambos corren bien con las versiones del
> `requirements.txt`.

---

## ✅ Evidencia de pruebas

El archivo `pruebas/test_componentes.py` valida —sin necesidad de token— toda la
lógica determinista del agente:

- Validación de calibres y torques (casos APTO y NO APTO)
- Generación y persistencia del plan
- Memoria de corto plazo (ventana)
- Memoria de largo plazo (persistencia entre sesiones + búsqueda)

Todas las pruebas pasan. La salida del ciclo completo agente↔LLM (con el razonamiento
paso a paso en modo `verbose`) se documenta en el informe como evidencia de las
decisiones adaptativas.

---

## 🗂️ Estructura de archivos
```
agente-funcional/
├── agente_funcional.py        # Agente principal (LangChain)
├── orquestador_crew.py        # Orquestador jerárquico (CrewAI)
├── herramientas.py            # Las 3 herramientas (consulta/razonamiento/escritura)
├── memoria.py                 # Memoria de corto y largo plazo
├── doc_demo.txt               # Documentación técnica de ejemplo
├── requirements.txt           # Dependencias
├── pruebas/
│   └── test_componentes.py    # Pruebas reproducibles
└── evidencia/
    └── diagrama_orquestacion.png
```

---

## 📚 Decisiones de diseño (el porqué de cada cosa)

- **¿Por qué LangChain + CrewAI y no solo uno?** LangChain es excelente para un agente
  individual potente con function calling confiable; CrewAI brilla cuando hay que
  coordinar varios especialistas. Usamos cada uno donde rinde mejor.
- **¿Por qué memoria en dos niveles?** Porque un plan de mantenimiento puede tomar
  varias sesiones. La memoria corta mantiene la conversación fluida; la larga asegura
  continuidad real en el tiempo.
- **¿Por qué validación determinista (tablas) y no dejar todo al LLM?** Porque en
  aviónica un calibre mal elegido es un riesgo de seguridad. Las cifras críticas se
  validan contra tablas de referencia, no contra la intuición del modelo.

---

## 📖 Referencias
Ver la sección de referencias (formato APA) en el informe técnico del proyecto.
