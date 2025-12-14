import torch
import umap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIGURACIÓN ---
DATA_PATH = Path("./data_processed/dataset.pt")
SAMPLES_TO_PLOT = 3000  # UMAP es lento, usamos una muestra de 3000 secuencias

def visualize_data():
    if not DATA_PATH.exists():
        print("❌ Ejecuta preprocess primero.")
        return

    print("📊 Cargando datos para análisis UMAP...")
    data = torch.load(DATA_PATH)
    
    # Convertimos a numpy y cogemos una muestra aleatoria
    # (Tu GTX 1060 / CPU te lo agradecerá)
    indices = torch.randperm(len(data))[:SAMPLES_TO_PLOT]
    subset = data[indices].numpy()

    print(f"Calculando proyección UMAP de {SAMPLES_TO_PLOT} secuencias...")
    print("Esto puede tardar unos minutos...")
    
    # UMAP reduce las 512 dimensiones (tokens) a 2 para poder pintar
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='manhattan')
    embedding = reducer.fit_transform(subset)

    # Visualización
    plt.figure(figsize=(12, 8))
    plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7, c='royalblue')
    plt.title(f"Distribución de Patrones de Chopin (UMAP) - {SAMPLES_TO_PLOT} Muestras")
    plt.xlabel("Dimensión UMAP 1")
    plt.ylabel("Dimensión UMAP 2")
    plt.grid(True, alpha=0.3)
    
    output_file = "umap_visualization.png"
    plt.savefig(output_file)
    print(f"✅ Gráfico guardado en: {output_file}")
    plt.show()

if __name__ == "__main__":
    visualize_data()