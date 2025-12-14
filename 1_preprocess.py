import os
import torch
from miditok import REMI, TokenizerConfig
from symusic import Score
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
MIDI_PATHS = Path("./dataset/clean_midi")
SAVE_DIR = Path("./data_processed")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
SEQ_LEN = 512  # Tamaño de contexto para GPT-2

def preprocess():
    # 1. Configuración del Tokenizador
    config = TokenizerConfig(
        pitch_range=(21, 109),
        #beat_res={(0, 4): 32, (4, 12): 16},#1/32 , Subir 96/32
        beat_res={(0, 4): 96, (4, 12): 32},
        #num_velocities=32,                    #Subir a 64
        num_velocities=64,
        use_chords=True,
        use_programs=False,
        use_tempos=True,
        use_sustain_pedals=True,
    )

    tokenizer = REMI(config)
    print("🎵 Iniciando tokenización de MIDIs...")
    
    midi_files = list(MIDI_PATHS.glob("**/*.mid"))
    all_tokens = []

    for midi_path in tqdm(midi_files):
        try:
            midi = Score(midi_path)
            tokens = tokenizer(midi)
            # Manejo robusto de la salida de miditok
            ids = tokens[0].ids if isinstance(tokens, list) else tokens.ids
            all_tokens.extend(ids)
        except Exception as e:
            pass # Ignoramos errores corruptos

    # 2. Guardar Tokenizador
    tokenizer.save_pretrained(SAVE_DIR / "tokenizer.json")

    # 3. Crear Tensor Gigante (Más rápido para UMAP y Training)
    print("📦 Convirtiendo a Tensores...")
    data_tensor = torch.tensor(all_tokens, dtype=torch.long)
    
    # Cortar en bloques exactos de SEQ_LEN
    num_sequences = len(data_tensor) // SEQ_LEN
    data_tensor = data_tensor[:num_sequences * SEQ_LEN]
    data_tensor = data_tensor.view(-1, SEQ_LEN)
    
    print(f"Dataset final: {data_tensor.shape}")
    torch.save(data_tensor, SAVE_DIR / "dataset.pt")
    print("✅ Preprocesado completado.")

if __name__ == "__main__":
    preprocess()