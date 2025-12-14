import torch
data = torch.load("./data_processed/dataset.pt")
print(f"Tipo de dato: {data.dtype}")
print(f"Muestra: {data[0][:10]}") # Vemos los primeros 10 valores
print(f"Min: {data.min()}, Max: {data.max()}")


print(f"¿PyTorch detecta CUDA?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Tarjeta detectada: {torch.cuda.get_device_name(0)}")
    print("✅ TODO LISTO. Ahora Optuna volará.")
else:
    print("❌ Seguimos sin detectar la GPU. Actualiza tus drivers de NVIDIA.")