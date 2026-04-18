# Agente de Consulta Técnica Aeronáutica — RAG
## ISY0101 · Evaluación Parcial N°1 | LATAM Airlines · Taller de Aviónica

## Descripción
Agente LLM + RAG que responde consultas técnicas de mantenimiento aeronáutico citando siempre la fuente documental y su URL. Si no encuentra respaldo documental, declina responder.

**Caso de uso real:** El CMM indica "crimpar cable según standard practices". El técnico consulta al agente para obtener la herramienta, terminal y procedimiento correcto — con cita del documento y URL verificable.

---

## Stack
| Componente | Tecnología |
|---|---|
| Framework | LangChain (Python) |
| LLM | GPT-4o-mini · GitHub Models |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB (persistente) |

---

## Instalación

```bash
git clone https://github.com/Jmunozf2000/Ingenier-a-de-Soluciones-con-Inteligencia-Artificial.git
cd Ingenier-a-de-Soluciones-con-Inteligencia-Artificial/RA1/IL1.3/agente-avionica
pip install -r requirements.txt
```

## Configuración

Crear archivo `.env`:
```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
OPENAI_BASE_URL=https://models.inference.ai.azure.com
GITHUB_BASE_URL=https://models.inference.ai.azure.com
```

> Obtener token en: GitHub → Settings → Developer Settings → Tokens (classic) → scope: `repo`
>
> ## Agregar documentos
>
> Colocar PDFs técnicos en la carpeta `docs/`. El agente los indexa automáticamente.
>
> Fuentes recomendadas:
> - TE Connectivity Crimping Guide: https://www.te.com/content/dam/te-com/documents/appliances/global/guide-to-crimping.pdf
> - - FAA AC 43.13-1B: https://www.faa.gov/regulations_policies/advisory_circulars
>   - - EASA Part-145: https://www.easa.europa.eu
>    
>     - El archivo `doc_demo.txt` (incluido) permite probar el agente sin necesidad de descargar PDFs.
>    
>     - ---
>
> ## Ejecución
>
> ```bash
> python agente_avionica.py
> ```
>
> **Salida esperada:**
> ```
> [CONSULTA]: ¿Qué herramienta uso para crimpar cable AWG 20?
> [RESPUESTA]:
> - Herramienta: M22520/2-01 con posicionador M22520/2-09
> - Terminal aprobado: MS20659-225
> - Pull test mínimo: 40 lb
>
> [FAA AC 43.13-1B | Sección 11-114 | 1998 (con revisiones)]
> URL: https://www.faa.gov/regulations_policies/advisory_circulars
> ```
>
> ---
>
> ## Estructura
> ```
> agente-avionica/
> ├── agente_avionica.py   # Código principal
> ├── requirements.txt     # Dependencias
> ├── doc_demo.txt         # Documento demo incluido
> ├── docs/                # Agregar PDFs aquí
> ├── chroma_db/           # Vector store (auto-generado)
> └── .env                 # Variables de entorno (no subir)
> ```
>
> ---
>
> ## Referencias
> - FAA. (1998). *AC 43.13-1B*. https://www.faa.gov
> - - LangChain. (2024). *RAG Docs*. https://python.langchain.com/docs/
>   - - TE Connectivity. (2024). *Guide to Crimping*. https://www.te.com
>    
>     - ## Uso de IA
>     - Claude (Anthropic) usado como apoyo para código y documentación. Citado según: https://bibliotecas.duoc.cl/ia
