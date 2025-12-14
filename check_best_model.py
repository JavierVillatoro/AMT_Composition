import json
import os
from pathlib import Path

# CONFIGURACIÓN
MODEL_DIR = Path("./model_final")  # Tu carpeta de salida

def find_best_checkpoint():
    state_file = MODEL_DIR / "trainer_state.json"
    
    if not state_file.exists():
        print(f"❌ No encuentro el archivo {state_file}")
        print("¿Ha terminado el entrenamiento ya?")
        return

    with open(state_file, "r") as f:
        data = json.load(f)

    # Buscamos en el historial todas las veces que se evaluó
    history = data["log_history"]
    eval_steps = [entry for entry in history if "eval_loss" in entry]
    
    if not eval_steps:
        print("⚠️ No hay datos de evaluación en el historial.")
        return

    # Ordenamos por menor loss (el mejor)
    best_step = min(eval_steps, key=lambda x: x["eval_loss"])
    
    print("\n" + "="*40)
    print("🏆 EL GANADOR (MEJOR MODELO)")
    print("="*40)
    print(f"📉 Loss más baja: {best_step['eval_loss']:.4f}")
    print(f"👣 Paso (Step):    {best_step['step']}")
    print(f"🔄 Epoch:          {best_step['epoch']}")
    print("="*40)
    
    print(f"\n👉 Para usar este modelo, en tu script de inferencia pon:")
    print(f'   model_path = "./model_final/checkpoint-{best_step["step"]}"')
    
    # Comprobación del modelo final
    last_step = eval_steps[-1]
    print(f"\n(Nota: El modelo final 'model_final' tiene Loss: {last_step['eval_loss']:.4f})")

if __name__ == "__main__":
    find_best_checkpoint()