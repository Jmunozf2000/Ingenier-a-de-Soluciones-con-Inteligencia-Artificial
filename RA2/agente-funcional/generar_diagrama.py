# ============================================================
# Sistema de Memoria del Agente Funcional
# Taller de Avionica LATAM Airlines
# ISY0101 - Ingenieria de Soluciones con IA - Evaluacion Parcial 2
# ------------------------------------------------------------
# Implementa la memoria del agente (Apartado B / IE3, IE4):
#
#   - CORTO PLAZO: ConversationBufferWindowMemory. Mantiene las
#     ultimas N interacciones del flujo de planificacion en curso,
#     dando coherencia inmediata sin gastar tokens innecesarios.
#
#   - LARGO PLAZO: persistencia en disco (JSON) del historial entre
#     sesiones. Permite que el agente "recuerde" ordenes y planes
#     previos, asegurando continuidad en tareas prolongadas (IE3).
#     Ademas se apoya en la recuperacion semantica del vector store
#     (ChromaDB) para el contexto documental (IE4).
# ============================================================

import os
import json
from datetime import datetime
from langchain.memory import ConversationBufferWindowMemory


# ------------------------------------------------------------
# MEMORIA DE CORTO PLAZO
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# MEMORIA DE LARGO PLAZO (persistente entre sesiones)
# ------------------------------------------------------------
class MemoriaLargoPlazo:
    """Almacena y recupera el historial de ordenes/planes entre sesiones.

    Persiste en un archivo JSON. Cada registro guarda la orden recibida,
    el plan generado y la marca de tiempo, de modo que en una sesion
    posterior el agente pueda recuperar el contexto de tareas previas.
    """

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
        """Registra una orden y su plan generado en la memoria persistente."""
        self.registros.append({
            "fecha": datetime.now().isoformat(timespec="seconds"),
            "orden": orden,
            "plan": plan,
        })
        with open(self.ruta, "w", encoding="utf-8") as f:
            json.dump(self.registros, f, ensure_ascii=False, indent=2)

    def recuperar_contexto(self, n: int = 3) -> str:
        """Devuelve un resumen textual de las ultimas n interacciones,
        para inyectar contexto de tareas previas al iniciar una sesion."""
        if not self.registros:
            return "Sin antecedentes de planes de mantenimiento previos."
        recientes = self.registros[-n:]
        lineas = []
        for r in recientes:
            lineas.append(f"- [{r['fecha']}] Orden: {r['orden']}")
        return "Antecedentes de planes previos:\n" + "\n".join(lineas)

    def buscar(self, palabra_clave: str) -> list:
        """Busca registros previos que contengan una palabra clave."""
        clave = palabra_clave.lower()
        return [r for r in self.registros
                if clave in r["orden"].lower() or clave in r["plan"].lower()]
