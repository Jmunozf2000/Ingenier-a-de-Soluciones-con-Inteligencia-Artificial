# Agente RAG - Consulta Tecnica Avionica
## LATAM Airlines - Taller de Avionica
### ISY0101 - Ingenieria de Soluciones con IA

## Descripcion
Agente de consulta tecnica basado en LLM + RAG para el Taller de Avionica de LATAM Airlines.
Permite buscar documentacion tecnica complementaria a los CMM disponibles en MyBoeingFleet ToolBox.

## Fuentes indexadas
- FAA Advisory Circular AC 43.13-1B (Wiring)
- Hojas SDS de materiales aeronauticos
- Normas MIL-SPEC aplicables
- Regulaciones DGAC / EASA / ANAC

## Instalacion
```bash
pip install -r requirements.txt
```

## Configuracion
Crea un archivo `.env` con:
```
GITHUB_TOKEN=tu_token_aqui
GITHUB_BASE_URL=https://models.inference.ai.azure.com
OPENAI_BASE_URL=https://models.inference.ai.azure.com
```

## Uso
```bash
python agente_avionica.py
```

## Stack tecnologico
- LLM: GitHub Models (gpt-4o-mini via Azure inference)
- Framework: LangChain
- Vector Store: ChromaDB
- Embeddings: text-embedding-3-small
- Prompt: Few-Shot Chain-of-Thought
