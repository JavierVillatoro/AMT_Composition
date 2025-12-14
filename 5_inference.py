import torch
from pathlib import Path
from miditok import REMI
from transformers import GPT2LMHeadModel
import time

# --- CONFIGURACIÓN ---
MODEL_PATH = Path("./model_final")       # Tu modelo entrenado
TOKENIZER_PATH = Path("./model_final")   # El tokenizer también se guardó ahí al final del train
OUTPUT_DIR = Path("./generated_music")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_music():
    print("--- 🎹 CARGANDO CEREBRO MUSICAL (INFERENCIA) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device.upper()}")

    # 1. CARGAR
    try:
        # Cargamos tokenizer desde la misma carpeta del modelo
        tokenizer = REMI.from_pretrained(TOKENIZER_PATH)
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
        model.eval() # Modo evaluación (apaga el aprendizaje)
    except Exception as e:
        print(f"❌ Error: {e}")
        print("¿Seguro que terminaste el entrenamiento y existe la carpeta model_final?")
        return

    print("✅ Modelo listo. ¡A componer!")

    while True:
        # 2. CONFIGURACIÓN USUARIO
        print("\n" + "-"*30)
        try:
            num_tokens = int(input("Longitud (tokens, ej. 500): ") or "500")
            temp = float(input("Temperatura (Creatividad 0.8 - 1.2): ") or "1.0")
        except:
            num_tokens, temp = 500, 1.0

        print(f"🎵 Generando obra de Chopin AI... (T={temp})")
        start_time = time.time()

        # 3. GENERACIÓN
        # Creamos una secuencia vacía (o con token de inicio si existe)
        # BOS = Beginning Of Sequence. Si no hay, usamos el id 0.
        bos_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.tensor([[bos_token]]).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids,
                max_length=num_tokens,
                do_sample=True,      # ¡Clave! Permite variedad
                temperature=temp,    # Controla el "caos"
                top_k=50,            # Se queda con las 50 mejores opciones
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.pad_token_id
            )

        # 4. DECODIFICACIÓN (Números -> Música)
        # Convertimos el tensor de GPU a lista de Python
        gen_seq = generated_ids[0].tolist()
        
        # CORRECCIÓN IMPORTANTE: Usamos .decode()
        # Esto crea un objeto Score (música)
        midi_output = tokenizer.decode(gen_seq)
        
        # 5. GUARDAR
        timestamp = int(time.time())
        filename = OUTPUT_DIR / f"chopin_ai_{timestamp}_t{temp}.mid"
        
        # Dump midi guarda el archivo
        midi_output.dump_midi(filename)
        
        print(f"✨ ¡Terminado en {time.time() - start_time:.1f}s!")
        print(f"💾 Guardado en: {filename}")
        
        if input("¿Otra? [s/n]: ").lower() == 'n':
            break

if __name__ == "__main__":
    generate_music()