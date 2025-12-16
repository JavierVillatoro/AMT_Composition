import torch
import time
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from transformers import GPT2LMHeadModel
from symusic import Score, Track, Note, Tempo

# --- CONFIGURACIÓN Y CONSTANTES GLOBALES ---
MODEL_PATH = Path("./model_final")
OUTPUT_DIR = Path("./generated_music_prueba")
OUTPUT_DIR.mkdir(exist_ok=True)
TICKS_PER_BEAT = 960 
MAX_CONTEXT = 512

def get_exact_tokenizer():
    # Sincronización con tu entrenamiento
    config = TokenizerConfig(
        pitch_range=(21, 109), beat_res={(0, 4): 32, (4, 12): 16}, 
        num_velocities=32, use_chords=True, use_programs=False, use_tempos=True,
    )
    return REMI(config)

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
                # Duración (Cálculo avanzado de V18)
                val_parts = value.split('.')
                try:
                    beats = int(val_parts[0]) if len(val_parts) > 0 else 0
                    subbeats = int(val_parts[1]) if len(val_parts) > 1 else 0
                    ticks = int(val_parts[2]) if len(val_parts) > 2 else 0
                    duration_ticks = (beats * TICKS_PER_BEAT) + (subbeats * TICKS_PER_BEAT // 4) + (ticks * 30) 
                except:
                    duration_ticks = TICKS_PER_BEAT 

                # CREAR NOTA 
                new_note = Note(
                    time=current_time, duration=duration_ticks if duration_ticks > 10 else 120,
                    pitch=PITCH, velocity=VELOCITY
                )
                track.notes.append(new_note); notes_counter += 1
                PITCH = None # Consumimos el Pitch
                
        elif etype == "Position":
            try:
                steps = int(value)
                # --- FIX CRÍTICO DEL SALTO DE TIEMPO ---
                # Limitamos el avance de tiempo a 1 beat (960 ticks)
                max_shift = TICKS_PER_BEAT 
                shift = steps * (TICKS_PER_BEAT // 32)
                
                current_time += min(shift, max_shift) # <--- SOLO AVANZA 1 BEAT MÁXIMO
            except: pass
            
        elif etype == "Bar":
            # Si detecta Bar, solo avanzamos 4 beats y no más
            current_time += TICKS_PER_BEAT * 4
        elif etype == "Tempo":
            pass 

    print(f"   ✅ Decodificación manual terminada. Notas añadidas: {notes_counter}")
    return score, notes_counter

def generate_music():
    print("\n" + "="*50)
    print("🎹  CHOPIN AI - V19 (PREVENCIÓN DE SILENCIO) 🎹")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = get_exact_tokenizer()
        BOS_TOKEN = tokenizer.vocab.get("Bar_None", 0) 
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print(f"✅ Sistema listo. Límite de tokens de la GPU: {MAX_CONTEXT}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    while True:
        print("-" * 30)
        try:
            length = int(input(f"Longitud (Máx {MAX_CONTEXT}, Default 500): ") or 500)
            length = min(length, MAX_CONTEXT) 
            temp = float(input("Temperatura (RECOMENDADO 0.7): ") or 0.7)
        except: length, temp = 500, 0.7

        print(f"🎵 Componiendo {length} tokens con T={temp}...")
        start_time = time.time()

        # Generación (Limitada a 512 tokens para evitar el error CUDA)
        input_ids = torch.tensor([[BOS_TOKEN]]).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=length, temperature=temp, do_sample=True, top_k=25,
                pad_token_id=0, eos_token_id=0
            )
        
        tokens = gen_ids[0].tolist()
        
        # --- DECODIFICACIÓN ---
        score_obj, n_notes = decode_by_state_machine(tokenizer, tokens)

        if n_notes < 50: # Subimos el límite de notas
             print(f"⚠️ Generación fallida. Solo {n_notes} notas creadas. Intenta con T=1.0")
             continue

        timestamp = int(time.time())
        filename = OUTPUT_DIR / f"chopin_distribuido_{timestamp}.mid"
        
        try:
            score_obj.dump_midi(filename) 
            print(f"✨ ¡ÉXITO! ({time.time() - start_time:.1f}s)")
            print(f"💾 Archivo: {filename.name}")
        except Exception as e:
            print(f"❌ Error al guardar MIDI: {e}")

        if input("\n¿Otra? [s/n]: ").lower() == 'n': break

if __name__ == "__main__":
    generate_music()