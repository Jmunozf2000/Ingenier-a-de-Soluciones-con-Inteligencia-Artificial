import os
import json
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory


def crear_memoria_corto_plazo(k: int = 5) -> ConversationBufferWindowMemory:
    """Crea la memoria de ventana deslizante para el agente.

    Args:
        k: numero de interacciones recientes a conservar (default 5).
    """
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
    )


class MemoriaLargoPlazo:
    """Almacena y recupera el historial de ordenes/planes entre sesiones.
    Persiste en un archivo JSON para dar continuidad a tareas prolongadas."""

    def __init__(self, ruta: str = "memoria_largo_plazo.json"):
        self.ruta = ruta
        self.registros = self._cargar()

    def _cargar(self) -> list:
        if os.path.exists(self.ruta):
            try:
                with open(self.ruta, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def guardar_interaccion(self, orden: str, plan: str) -> None:
        self.registros.append({
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "orden": orden,
            "plan": plan,
        })
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self.registros, f, ensure_ascii=False, indent=2)

    def recuperar_contexto(self, n: int = 3) -> str:
        if not self.registros:
            return "Sin antecedentes de planes de mantenimiento previos."
        recientes = self.registros[-n:]
        lineas = [f"- [{r['fecha']}] Orden: {r['orden']}" for r in recientes]
        return "Antecedentes de planes previos:\n" + "\n".join(lineas)

    def buscar(self, palabra_clave: str) -> list:
        clave = palabra_clave.lower()
        return [r for r in self.registros
                if clave in r["orden"].lower() or clave in r["plan"].lower()]
