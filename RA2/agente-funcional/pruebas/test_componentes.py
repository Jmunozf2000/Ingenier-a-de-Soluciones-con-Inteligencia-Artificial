import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from herramientas import validar_parametro_tecnico, generar_plan_mantenimiento
from memoria import MemoriaLargoPlazo, crear_memoria_corto_plazo


def test_razonamiento():
    print("\n[TEST] Herramienta de razonamiento (validar_parametro_tecnico)")
    apto = validar_parametro_tecnico.invoke({"tipo": "calibre", "valor": 16, "referencia": "18"})
    noapto = validar_parametro_tecnico.invoke({"tipo": "calibre", "valor": 20, "referencia": "18"})
    torque_ok = validar_parametro_tecnico.invoke({"tipo": "torque", "valor": 22, "referencia": "terminal_anillo_10"})
    assert apto.startswith("[APTO]")
    assert noapto.startswith("[NO APTO]")
    assert torque_ok.startswith("[APTO]")
    print("   ", apto)
    print("   ", noapto)
    print("   ", torque_ok)
    print("   PASS")


def test_escritura():
    print("\n[TEST] Herramienta de escritura (generar_plan_mantenimiento)")
    salida = generar_plan_mantenimiento.invoke({
        "titulo": "Inspeccion de cableado - componente avionica",
        "subtareas_priorizadas": "1. (Seguridad) Cortar energia\n2. (Inspeccion) Revisar empalmes",
        "fuente": "doc_demo.txt | AS50881",
    })
    assert "[OK] Plan de mantenimiento generado" in salida
    print("    Plan generado y persistido en archivo.")
    print("   PASS")


def test_memoria_corto():
    print("\n[TEST] Memoria de corto plazo (ventana k=5)")
    m = crear_memoria_corto_plazo(k=5)
    m.save_context({"input": "Orden A"}, {"output": "Plan A"})
    m.save_context({"input": "Orden B"}, {"output": "Plan B"})
    hist = m.load_memory_variables({})["chat_history"]
    assert len(hist) == 4
    print("    Ventana mantiene", len(hist), "mensajes (2 turnos).")
    print("   PASS")


def test_memoria_largo():
    print("\n[TEST] Memoria de largo plazo (persistente entre sesiones)")
    ruta = "_test_memoria.json"
    if os.path.exists(ruta):
        os.remove(ruta)
    ml = MemoriaLargoPlazo(ruta)
    ml.guardar_interaccion("Inspeccion 100h componente Y", "Plan 4 subtareas")
    ml.guardar_interaccion("Revision torque terminal Z", "Plan 3 subtareas")
    ml2 = MemoriaLargoPlazo(ruta)
    assert len(ml2.registros) == 2
    assert len(ml2.buscar("torque")) == 1
    print("    Persistencia OK: 2 registros tras recarga. Busqueda 'torque': 1.")
    os.remove(ruta)
    print("   PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("PRUEBAS DE COMPONENTES - AGENTE FUNCIONAL (EP2)")
    print("=" * 60)
    test_razonamiento()
    test_escritura()
    test_memoria_corto()
    test_memoria_largo()
    print("\n" + "=" * 60)
    print("TODAS LAS PRUEBAS PASARON")
    print("=" * 60)
