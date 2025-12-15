import torch
import time
import os
from pathlib import Path
from miditok import REMI, TokenizerConfig
from transformers import GPT2LMHeadModel
# 1. AÑADIDO: ControlChange para manejar el pedal
from symusic import Score, Track, Note, Tempo, ControlChange

# --- CONFIGURACIÓN Y CONSTANTES GLOBALES ---
MODEL_PATH = Path("./model_highres")
OUTPUT_DIR = Path("./generated_music_highres")
OUTPUT_DIR.mkdir(exist_ok=True)
TICKS_PER_BEAT = 960 
MAX_CONTEXT = 1024 # Te recomiendo subir esto a 1024 si puedes, para dar espacio a los pedales

def get_exact_tokenizer():
    # Sincronización con tu entrenamiento
    config = TokenizerConfig(
        pitch_range=(21, 109), 
        beat_res={(0, 4): 96, (4, 12): 32}, 
        num_velocities=64, 
        use_chords=True, 
        use_programs=False, 
        use_tempos=True,
        # 2. AÑADIDO: Activamos pedales en la configuración
        use_pedals=True 
    )
    return REMI(config)

def decode_by_state_machine(tokenizer, tokens):
    print("🤖 Aplicando Máquina de Estados (Con Pedal)...")
    
    score = Score(); score.ticks_per_quarter = TICKS_PER_BEAT 
    track = Track(name="Piano AI", program=0, is_drum=False)
    score.tracks.append(track)
    score.tempos.append(Tempo(time=0, qpm=120))
    
    current_time = 0; PITCH = None; VELOCITY = None; DURATION = None
    notes_counter = 0
    pedal_counter = 0 # Para contar pedales

    for t in tokens:
        if t == 0: continue
        try:
            event_str = str(tokenizer[t])
            # Filtro de seguridad por si hay tokens raros
            if '_' not in event_str: continue 
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
                max_shift = TICKS_PER_BEAT 
                # Ajuste: Si tu beat_res es 96, hay que ajustar el divisor. 
                # Pero si usas la lógica standard de REMI (32 bins), esto suele estar bien.
                shift = steps * (TICKS_PER_BEAT // 32) 
                
                current_time += min(shift, max_shift) 
            except: pass
            
        elif etype == "Bar":
            # Si detecta Bar, solo avanzamos 4 beats y no más
            current_time += TICKS_PER_BEAT * 4
        
        elif etype == "Tempo":
            pass 

        # 3. AÑADIDO: Lógica de PEDAL
        elif etype == "Pedal":
            # Pedal On (Sustain CC 64 = 127)
            track.controls.append(ControlChange(time=current_time, number=64, value=127))
            pedal_counter += 1
            
        elif etype == "PedalOff":
            # Pedal Off (Sustain CC 64 = 0)
            track.controls.append(ControlChange(time=current_time, number=64, value=0))
            pedal_counter += 1

    print(f"   ✅ Decodificación terminada. Notas: {notes_counter} | Pedales: {pedal_counter}")
    return score, notes_counter

def generate_music():
    print("\n" + "="*50)
    print("🎹  CHOPIN AI - V19 (CON PEDAL) 🎹")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        tokenizer = get_exact_tokenizer()
        # Intentamos obtener el token de barra o usamos padding/0
        BOS_TOKEN = tokenizer.vocab.get("Bar_None", 0) 
        
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
        model.eval()
        print(f"✅ Sistema listo. Límite de tokens de la GPU: {MAX_CONTEXT}")
    except Exception as e:
        print(f"❌ Error cargando modelo/tokenizer: {e}")
        return

    while True:
        print("-" * 30)
        try:
            length = int(input(f"Longitud (Máx {MAX_CONTEXT}, Default 500): ") or 500)
            length = min(length, MAX_CONTEXT) 
            temp = float(input("Temperatura (RECOMENDADO 0.8): ") or 0.8)
        except: length, temp = 500, 0.8

        print(f"🎵 Componiendo {length} tokens con T={temp}...")
        start_time = time.time()

        input_ids = torch.tensor([[BOS_TOKEN]]).to(device)
        
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids, 
                max_length=length, 
                temperature=temp, 
                do_sample=True, 
                top_k=40,
                pad_token_id=0, 
                eos_token_id=0
            )
        
        tokens = gen_ids[0].tolist()
        
        # --- DECODIFICACIÓN ---
        score_obj, n_notes = decode_by_state_machine(tokenizer, tokens)

        if n_notes < 20: 
             print(f"⚠️ Generación pobre ({n_notes} notas). Intenta subir T=1.0")
        else:
            timestamp = int(time.time())
            filename = OUTPUT_DIR / f"chopin_pedal_{timestamp}.mid"
            
            try:
                score_obj.dump_midi(str(filename)) 
                print(f"✨ ¡ÉXITO! ({time.time() - start_time:.1f}s)")
                print(f"💾 Archivo: {filename.name}")
            except Exception as e:
                print(f"❌ Error al guardar MIDI: {e}")

        if input("\n¿Otra? [s/n]: ").lower() == 'n': break

if __name__ == "__main__":
    generate_music()