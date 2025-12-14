import json
from pathlib import Path
from miditok import REMI

# --- RUTA DEL MODELO ---
TOKENIZER_PATH = Path("./model_final")  # Asegúrate de que apunta a donde está tokenizer.json

def inspeccionar_cerebro():
    print("🕵️‍♂️  INICIANDO DIAGNÓSTICO PROFUNDO...")
    print(f"📂 Buscando tokenizer en: {TOKENIZER_PATH.absolute()}")
    
    try:
        # 1. CARGAR TOKENIZADOR
        tokenizer = REMI.from_pretrained(TOKENIZER_PATH)
        print("✅ Tokenizador cargado.")
        
        # 2. PROBAR EL TOKEN MALDITO (598)
        TOKEN_MALDITO = 598
        print(f"\n🔎  ANALIZANDO TOKEN {TOKEN_MALDITO}...")
        
        # Verificamos si está en el vocabulario básico
        vocab_size = len(tokenizer)
        print(f"   - Tamaño del vocabulario: {vocab_size}")
        
        if TOKEN_MALDITO >= vocab_size:
            print("   🚨  ¡ALERTA ROJA! El token 598 está FUERA del rango del vocabulario.")
            print("        El modelo está alucinando números que no existen.")
            return

        # Intentamos obtener el evento asociado (Traducción Inversa)
        # Probamos varios métodos según la versión de miditok
        event = None
        try:
            # Método A: Acceso directo (versiones nuevas)
            event = tokenizer[TOKEN_MALDITO]
            print(f"   - Método A (Directo): {event}")
        except:
            try:
                # Método B: Vocabulario interno
                # miditok suele guardar el vocab como lista de eventos
                if hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, list):
                     # Buscar en la lista el valor
                     pass 
                elif hasattr(tokenizer, 'vocab') and isinstance(tokenizer.vocab, dict):
                    # Invertir diccionario
                    inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
                    event = inv_vocab.get(TOKEN_MALDITO, "No encontrado")
                    print(f"   - Método B (Dict): {event}")
            except Exception as e:
                print(f"   - Método B falló: {e}")

        # 3. INTENTO DE DECODIFICACIÓN AISLADA
        print("\n🧪  PRUEBA DE DECODIFICACIÓN AISLADA:")
        fake_seq = [TOKEN_MALDITO]
        try:
            # Intentamos convertir solo ese token a MIDI
            midi = tokenizer.decode(fake_seq)
            print("   ✅  ¡INCREÍBLE! El token se decodificó correctamente solo.")
            print("        (El problema podría ser la secuencia, no el token individual)")
        except Exception as e:
            print(f"   ❌  FALLÓ AL DECODIFICAR: {e}")
            print("        Este es el problema: El token existe en el vocabulario,")
            print("        pero miditok no sabe cómo convertirlo a nota.")

        # 4. EXPORTAR VOCABULARIO (Para que yo lo vea)
        print("\n📝  EXPORTANDO DICCIONARIO...")
        debug_file = "vocabulario_debug.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            if hasattr(tokenizer, "vocab") and isinstance(tokenizer.vocab, dict):
                # Ordenar por ID
                sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda item: item[1])
                for k, v in sorted_vocab:
                    f.write(f"{v}: {k}\n")
            else:
                f.write("No se pudo extraer el vocabulario como diccionario simple.")
                
        print(f"   ✅ Guardado en '{debug_file}'.")
        print("   -> Por favor, abre ese archivo y busca qué pone en la línea 598.")

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO EN EL DIAGNÓSTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspeccionar_cerebro()