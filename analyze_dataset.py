import os
import pretty_midi
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# La ruta a tu carpeta con los MIDIs de Chopin
DATASET_DIR = Path("./dataset/clean_midi") 
# ---------------------

def analizar_midis():
    print(f"🔍 Analizando dataset en: {DATASET_DIR}")
    
    # Buscamos archivos .mid y .midi recursivamente
    archivos = list(DATASET_DIR.rglob("*.mid")) + list(DATASET_DIR.rglob("*.midi"))
    
    if not archivos:
        print("❌ No se encontraron archivos MIDI en esa carpeta.")
        return

    total_files = len(archivos)
    print(f"📄 Archivos encontrados: {total_files}")
    print("⏳ Procesando... (esto puede tardar un poco dependiendo del número de notas)")

    total_duration_sec = 0
    total_notes = 0
    corrupt_files = 0
    
    # Variables para calcular promedios
    max_duration = 0
    min_duration = float('inf')

    # Usamos tqdm para ver el progreso
    for midi_path in tqdm(archivos):
        try:
            # Cargar el MIDI
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            
            # Duración del archivo actual
            duration = pm.get_end_time()
            
            # Contar notas (sumando las notas de todos los instrumentos/pistas)
            notes_count = sum([len(instrument.notes) for instrument in pm.instruments])
            
            # Acumuladores
            total_duration_sec += duration
            total_notes += notes_count
            
            # Stats de extremos
            if duration > max_duration: max_duration = duration
            if duration < min_duration: min_duration = duration

        except Exception as e:
            # Si el archivo está corrupto o vacío
            corrupt_files += 1
            # print(f"Error en {midi_path.name}: {e}")

    # --- RESULTADOS ---
    if total_files - corrupt_files == 0:
        print("Todos los archivos dieron error.")
        return

    valid_files = total_files - corrupt_files
    total_hours = total_duration_sec / 3600
    avg_notes_per_song = total_notes / valid_files
    avg_duration = total_duration_sec / valid_files

    print("\n" + "="*40)
    print(f"🎹 REPORTE DEL DATASET: CHOPIN")
    print("="*40)
    print(f"✅ Archivos válidos:      {valid_files}")
    print(f"❌ Archivos corruptos:    {corrupt_files}")
    print(f"🎵 Total de notas (tokens): {total_notes:,}")
    print(f"⏱️  Duración total:        {total_hours:.2f} horas ({total_duration_sec/60:.0f} minutos)")
    print("-" * 40)
    print(f"📊 Promedios:")
    print(f"   - Notas por canción:   {int(avg_notes_per_song)}")
    print(f"   - Duración promedio:   {avg_duration/60:.2f} minutos")
    print(f"   - Canción más larga:   {max_duration/60:.2f} minutos")
    print("="*40)
    
    # INTERPRETACIÓN RÁPIDA
    print("\n💡 DIAGNÓSTICO PARA ENTRENAMIENTO:")
    estimate_tokens = total_notes * 3  # Estimación burda: 1 nota ≈ 3 tokens (NoteOn, NoteOff, TimeShift)
    print(f"   Estimas aprox. {estimate_tokens:,} tokens de entrenamiento.")
    
    if total_hours < 1:
        print("   ⚠️ MUY POCO: Menos de 1 hora. El modelo sobreajustará (memorizará) enseguida.")
    elif total_hours < 5:
        print("   ⚠️ POCO: Entre 1 y 5 horas. Bueno para pruebas rápidas, pero el modelo será limitado.")
    elif total_hours < 20:
        print("   ✅ DECENTE: Entre 5 y 20 horas. Suficiente para un modelo de juguete que suene bien.")
    else:
        print("   🚀 EXCELENTE: Más de 20 horas. Tienes datos para hacer algo serio.")

if __name__ == "__main__":
    analizar_midis()