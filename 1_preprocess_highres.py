import os
import torch
from miditok import REMI, TokenizerConfig
from symusic import Score
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN (RUTAS SEGURAS)
# ==========================================
# Ruta donde están tus midis originales
MIDI_PATHS = Path("./dataset/clean_midi") 

# ⚠️ CAMBIO DE NOMBRE: Nueva carpeta para no borrar lo anterior
SAVE_DIR = Path("./data_processed_highres") 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 512  # Tamaño de contexto GPT-2  1024 para contexto pedal??? 

def preprocess():
    print("🚀 INICIANDO PREPROCESAMIENTO DE ALTA RESOLUCIÓN (ANTI-GRID)")
    print(f"📂 Los datos se guardarán en: {SAVE_DIR}")
    
    # 1. Configuración del Tokenizador (MODO CHOPIN REALISTA)
    config = TokenizerConfig(
        pitch_range=(21, 109),
        
        # --- EL SECRETO DEL RUBATO ---
        # 96 ticks por tiempo. 
        # Esto permite que la nota se mueva "fuera" del golpe matemático.
        beat_res={(0, 4): 96, (4, 12): 32}, 
        
        # --- MÁS DINÁMICA ---
        # 64 niveles de volumen en lugar de 32 para más expresividad
        num_velocities=64, 
        
        use_chords=True,
        use_programs=False,
        use_tempos=True,
        
        # --- EL ALMA DEL PIANO ---
        # Activamos el pedal para capturar la resonancia
        use_sustain_pedals=True, 
    )

    tokenizer = REMI(config)
    print("🎵 Tokenizando MIDIs con resolución 1/96 y Pedales...")
    
    midi_files = list(MIDI_PATHS.glob("**/*.mid"))
    all_tokens = []
    
    # Contadores para estadísticas
    files_ok = 0
    files_error = 0

    for midi_path in tqdm(midi_files):
        try:
            # Usamos symusic para cargar rápido
            midi = Score(midi_path)
            
            # Tokenizamos
            tokens = tokenizer(midi)
            
            # Verificación de formato para evitar errores de versión de miditok
            if isinstance(tokens, list):
                # Si devuelve lista de tracks, cogemos el primero o aplanamos
                ids = tokens[0].ids 
            elif hasattr(tokens, "ids"):
                ids = tokens.ids
            else:
                continue # Formato desconocido
                
            all_tokens.extend(ids)
            files_ok += 1
        except Exception as e:
            # Si un archivo falla, no paramos todo el proceso
            files_error += 1
            # print(f"Error en {midi_path.name}: {e}") 

    print(f"\n📊 Resumen:")
    print(f"   - Archivos procesados: {files_ok}")
    print(f"   - Archivos fallidos: {files_error}")
    print(f"   - Total de tokens: {len(all_tokens)}")

    # 2. Guardar Tokenizador
    print(f"💾 Guardando tokenizer en {SAVE_DIR}...")
    tokenizer.save_pretrained(SAVE_DIR / "tokenizer.json")

    # 3. Crear Tensor Gigante
    if len(all_tokens) == 0:
        print("❌ ERROR: No se generaron tokens. Revisa la carpeta de MIDIs.")
        return

    print("📦 Convirtiendo a Tensores (Training Data)...")
    data_tensor = torch.tensor(all_tokens, dtype=torch.long)
    
    # Cortar para que encaje perfecto en bloques de 512
    num_sequences = len(data_tensor) // SEQ_LEN
    data_tensor = data_tensor[:num_sequences * SEQ_LEN]
    data_tensor = data_tensor.view(-1, SEQ_LEN)
    
    print(f"🔥 Dataset final listo: {data_tensor.shape}")
    torch.save(data_tensor, SAVE_DIR / "dataset.pt")
    print("✅ ¡LISTO! Ahora tienes datos de Alta Definición.")

if __name__ == "__main__":
    preprocess()