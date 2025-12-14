import torch
import json
import matplotlib.pyplot as plt
import os
import sys
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from miditok import REMI
from pathlib import Path
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

# --- CONFIGURACIÓN ---
DATA_DIR = Path("./data_processed")
OUTPUT_DIR = Path("./model_final")
USE_OPTUNA_PARAMS = True
VALIDATION_SPLIT = 0.1  # 10% para validar

# --- CLASES NECESARIAS (Copiadas del fix anterior) ---
class ChopinDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx].long()}

def custom_collator(features):
    batch_tensors = [f["input_ids"] for f in features]
    input_ids = pad_sequence(batch_tensors, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": input_ids}

# --- FUNCIÓN DE GRÁFICA ---
def plot_history(log_history, save_path):
    train_loss = []
    eval_loss = []
    steps = []
    eval_steps = []

    for entry in log_history:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            steps.append(entry['step'])
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_steps.append(entry['step'])

    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_loss, label='Training Loss', alpha=0.6)
    if eval_loss and eval_steps:
        plt.plot(eval_steps, eval_loss, label='Validation Loss', linewidth=2, color='red')
    
    plt.xlabel('Pasos')
    plt.ylabel('Loss')
    plt.title('Curva de Aprendizaje - Chopin AI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"📈 Gráfica guardada en: {save_path}")
    plt.close()

def main():
    # 0. Check GPU
    print("\n" + "="*50)
    if torch.cuda.is_available():
        print(f"🚀 ENTRENANDO EN: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ PRECAUCIÓN: Entrenando en CPU (será lento)")
    print("="*50 + "\n")

    # 1. Cargar Tokenizer y Datos
    tokenizer = REMI.from_pretrained(DATA_DIR / "tokenizer.json")
    
    # Carga segura
    try:
        full_data_tensor = torch.load(DATA_DIR / "dataset.pt", weights_only=False)
    except TypeError:
        full_data_tensor = torch.load(DATA_DIR / "dataset.pt")
        
    # Envolvemos en nuestra clase corregida
    full_dataset = ChopinDataset(full_data_tensor)

    # División Train/Val
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"📊 Datos cargados: {len(full_dataset)} secuencias.")
    print(f"   - Entrenamiento: {train_size}")
    print(f"   - Validación:    {val_size}")

    # 2. Configuración (con Optuna)
    # Valores por defecto por si no existe el json
    params = {
        "n_layer": 6, 
        "n_head": 8, 
        "n_embd": 512, 
        "learning_rate": 5e-4, 
        "batch_size": 8 
    }

    if USE_OPTUNA_PARAMS and os.path.exists("best_params.json"):
        print("✨ Cargando hiperparámetros optimizados de 'best_params.json'...")
        with open("best_params.json", "r") as f:
            best = json.load(f)
            params.update(best)
            # Aseguramos que batch_size sea entero (a veces json lo carga como float)
            params["batch_size"] = int(params["batch_size"]) 
            params["n_layer"] = int(params["n_layer"])
            params["n_head"] = int(params["n_head"])
            params["n_embd"] = int(params["n_embd"])

    config = GPT2Config(
        vocab_size=len(tokenizer.vocab),
        n_positions=512,
        n_ctx=512,
        n_embd=params["n_embd"],
        n_layer=params["n_layer"],
        n_head=params["n_head"],
        bos_token_id=0,
        eos_token_id=0,
        activation_function="gelu_new"
    )
    
    model = GPT2LMHeadModel(config)

    # 3. Argumentos
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,        # Puedes subir esto a 20 o 30 si ves que la loss sigue bajando
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        
        eval_strategy="steps",      # Corregido (antes evaluation_strategy)
        eval_steps=100,             # Evaluamos frecuentemente
        save_steps=500,
        logging_steps=50,
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,   # 0 para evitar errores en Windows
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collator, # ¡Importante! Usamos el collator manual
    )

    print("🔥 Iniciando entrenamiento final...")
    trainer.train()
    
    # 4. Guardado
    print(f"💾 Guardando modelo en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 5. Gráfica
    plot_history(trainer.state.log_history, OUTPUT_DIR / "training_loss.png")
    print("✅ ¡Entrenamiento completado!")

if __name__ == "__main__":
    main()