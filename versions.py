import sys
import platform
import importlib

# Lista de librerías clave para tu proyecto
libraries = [
    "miditok",
    "symusic",
    "torch",
    "transformers",
    "matplotlib",
    "tqdm",
    "numpy",
    "accelerate", # Importante para el Trainer de HuggingFace
    "packaging"
]

print("--- 📋 REPORTE DE VERSIONES ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"Platform: {platform.system()} {platform.release()}")
print("-" * 30)

found_versions = []

for lib in libraries:
    try:
        # Intentamos importar la librería
        module = importlib.import_module(lib)
        
        # Buscamos la versión
        version = getattr(module, '__version__', 'Desconocida')
        
        # Formato listo para requirements.txt
        output = f"{lib}=={version}"
        print(f"✅ {output}")
        found_versions.append(output)
        
    except ImportError:
        print(f"❌ {lib}: NO INSTALADO")
    except Exception as e:
        print(f"⚠️ {lib}: Error al leer versión ({e})")

print("-" * 30)
print("INFORMACIÓN EXTRA (CUDA/GPU):")
try:
    import torch
    print(f"Torch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("Estás usando CPU (el entrenamiento será lento).")
except:
    pass
print("-" * 30)