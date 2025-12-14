import torch
import time
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from transformers import GPT2LMHeadModel
from symusic import Score, Track, Note, Tempo

# --- CONFIGURACIÓN Y CONSTANTES GLOBALES ---
MODEL_PATH = Path("./model_final") # O la ruta de tu modelo preferido
OUTPUT_DIR = Path("./generated_music")
OUTPUT_DIR.mkdir(exist_ok=True)
TICKS_PER_BEAT = 960 
MAX_CONTEXT = 512 # Límite físico del modelo

def get_exact_tokenizer():
    # Sincronización con tu entrenamiento
    config = TokenizerConfig(
        pitch_range=(21, 109), beat_res={(0, 4): 32, (4, 12): 16}, 
        num_velocities=32, use_chords=True, use_programs=False, use_tempos=True,
    )
    return REMI(config)

# --- TU DECODIFICADOR ORIGINAL (INTACTO) ---
def decode_by_state_machine(tokenizer, tokens):
    print("🤖 Aplicando Máquina de Estados (Ritmo lento)...")
    
    score = Score(); score.ticks_per_quarter = TICKS_PER_BEAT 
    track = Track(name="Piano AI", program=0, is_drum=False)
    score.tracks.append(track)
    score.tempos.append(Tempo(time=0, qpm=120))
    
    current_time = 0; PITCH = None; VELOCITY = None; DURATION = None; notes_counter = 0

    for t in tokens:
        if t == 0: continue
        try:
            event_str = str(tokenizer[t])
            etype, value = event_str.split('_')
        except: continue

        if etype == "Pitch":
            PITCH = int(value); VELOCITY = None; DURATION = None
        elif etype == "Velocity":
            if PITCH is not None: VELOCITY = int(value)
            
        elif etype == "Duration":
            if PITCH is not None and VELOCITY is not None:
                val_parts = value.split('.')
                try:
                    beats = int(val_parts[0]) if len(val_parts) > 0 else 0
                    subbeats = int(val_parts[1]) if len(val_parts) > 1 else 0
                    ticks = int(val_parts[2]) if len(val_parts) > 2 else 0
                    duration_ticks = (beats * TICKS_PER_BEAT) + (subbeats * TICKS_PER_BEAT // 4) + (ticks * 30) 
                except:
                    duration_ticks = TICKS_PER_BEAT 

                new_note = Note(
                    time=current_time, duration=duration_ticks if duration_ticks > 10 else 120,
                    pitch=PITCH, velocity=VELOCITY
                )
                track.notes.append(new_note); notes_counter += 1
                PITCH = None 
                
        elif etype == "Position":
            try:
                steps = int(value)
                max_shift = TICKS_PER_BEAT 
                shift = steps * (TICKS_PER_BEAT // 32)
                current_time += min(shift, max_shift) 
            except: pass
            
        elif etype == "Bar":
            current_time += TICKS_PER_BEAT * 4
        elif etype == "Tempo":
            pass 

    print(f"   ✅ Decodificación manual terminada. Notas añadidas: {notes_counter}")
    return score, notes_counter

# --- NUEVA FUNCIÓN DE GENERACIÓN LARGA ---
def generate_music():
    print("\n" + "="*50)
    print("🎹  CHOPIN AI - MODO EXTENDIDO (SLIDING WINDOW) 🎹")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = get_exact_tokenizer()
        BOS_TOKEN = tokenizer.vocab.get("Bar_None", 0) 
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print(f"✅ Sistema listo. Ventana de contexto: {MAX_CONTEXT}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    while True:
        print("-" * 30)
        try:
            # ¡Ahora puedes pedir 2000 o 5000 tokens!
            target_length = int(input(f"Longitud total deseada (Ej. 2000): ") or 1000)
            temp = float(input("Temperatura (RECOMENDADO 0.7 - 0.8): ") or 0.7)
        except: target_length, temp = 1000, 0.7

        print(f"🎵 Componiendo pieza larga de {target_length} tokens...")
        start_time = time.time()

        # --- LÓGICA DE VENTANA DESLIZANTE ---
        # 1. Iniciamos la historia completa con el token de inicio
        full_song_tokens = [BOS_TOKEN]
        
        # 2. El contexto actual es lo que ve el modelo (máximo 512)
        current_context = torch.tensor([[BOS_TOKEN]]).to(device)
        
        # Barra de progreso simple
        from tqdm import tqdm
        pbar = tqdm(total=target_length, desc="Generando", unit="tok")

        while len(full_song_tokens) < target_length:
            
            # Si el contexto está lleno (512), cortamos y nos quedamos con los últimos 511
            # para dejar espacio a 1 nuevo.
            if current_context.shape[1] >= MAX_CONTEXT:
                current_context = current_context[:, -MAX_CONTEXT:]

            with torch.no_grad():
                # Generamos UN solo token nuevo a la vez (o bloques pequeños)
                # Generar de 1 en 1 es más lento pero más seguro para la ventana
                outputs = model(current_context)
                next_token_logits = outputs.logits[:, -1, :] / temp
                
                # Sampling
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Actualizamos listas
                next_token_item = next_token.item()
                full_song_tokens.append(next_token_item)
                
                # Añadimos al tensor de contexto para la siguiente vuelta
                current_context = torch.cat([current_context, next_token], dim=1)
                
                pbar.update(1)

        pbar.close()
        
        # --- DECODIFICACIÓN DE LA LISTA COMPLETA ---
        print("\n🔨 Decodificando secuencia completa...")
        score_obj, n_notes = decode_by_state_machine(tokenizer, full_song_tokens)

        if n_notes < 50: 
             print(f"⚠️ Pocas notas ({n_notes}). Intenta de nuevo.")
             continue

        timestamp = int(time.time())
        filename = OUTPUT_DIR / f"chopin_long_{target_length}_{timestamp}.mid"
        
        try:
            score_obj.dump_midi(filename) 
            print(f"✨ ¡ÉXITO! ({time.time() - start_time:.1f}s)")
            print(f"💾 Guardado: {filename.name}")
        except Exception as e:
            print(f"❌ Error al guardar MIDI: {e}")

        if input("\n¿Otra? [s/n]: ").lower() == 'n': break

if __name__ == "__main__":
    generate_music()