# PEDAL AÑADIDO Y CUANTIZACION 96
import torch
import json
import matplotlib.pyplot as plt
import os
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from miditok import REMI
from pathlib import Path
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

# ==================================================
# CONFIGURACIÓN (HIGH RESOLUTION)
# ==================================================
# 1. Apuntamos a la carpeta NUEVA con datos de alta calidad
DATA_DIR = Path("./data_processed_highres")

# 2. Carpeta de salida nueva (para no sobrescribir la antigua)
OUTPUT_DIR = Path("./model_highres")
OUTPUT_DIR.mkdir(exist_ok=True)

USE_OPTUNA_PARAMS = True
VALIDATION_SPLIT = 0.1

# ==================================================
# CLASES DE DATOS
# ==================================================
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

# ==================================================
# GRÁFICA DE ENTRENAMIENTO
# ==================================================
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
    plt.title('Curva de Aprendizaje - Chopin High-Res')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    print(f"📈 Gráfica guardada en: {save_path}")
    plt.close()

# ==================================================
# MAIN
# ==================================================
def main():
    print("\n" + "="*50)
    print("🎹 ENTRENAMIENTO CHOPIN: ALTA RESOLUCIÓN + PEDALES 🎹")
    print("="*50 + "\n")

    if torch.cuda.is_available():
        print(f"🚀 GPU Activa: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CPU Activa (Lento)")

    # 1. Cargar Tokenizer NUEVO
    print(f"📂 Cargando datos desde: {DATA_DIR}")
    try:
        tokenizer = REMI.from_pretrained(DATA_DIR / "tokenizer.json")
    except:
        print("❌ Error: No se encuentra tokenizer.json en la carpeta highres.")
        return

    # 2. Cargar Dataset NUEVO
    try:
        full_data_tensor = torch.load(DATA_DIR / "dataset.pt", weights_only=False)
    except:
        full_data_tensor = torch.load(DATA_DIR / "dataset.pt")
        
    full_dataset = ChopinDataset(full_data_tensor)

    # Split
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"📊 Secuencias High-Res: {len(full_dataset)}")

    # 3. Configuración del Modelo (FRESH START)
    # Usamos los mejores parámetros si existen, pero el modelo nace de cero
    params = {
        "n_layer": 6, "n_head": 8, "n_embd": 512, 
        "learning_rate": 5e-4, "batch_size": 8 
    }

    if USE_OPTUNA_PARAMS and os.path.exists("best_params.json"):
        print("✨ Usando hiperparámetros de 'best_params.json'")
        with open("best_params.json", "r") as f:
            best = json.load(f)
            params.update(best)
            params["batch_size"] = int(params["batch_size"]) 
            params["n_layer"] = int(params["n_layer"])
            params["n_head"] = int(params["n_head"])
            params["n_embd"] = int(params["n_embd"])

    # INICIALIZACIÓN DE CERO (Necesario porque cambió el vocabulario)
    config = GPT2Config(
        vocab_size=len(tokenizer.vocab), # ¡El tamaño nuevo será más grande!
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
    print("👶 Modelo inicializado desde cero (adaptado a 96-ticks/beat)")

    # 4. Argumentos de Entrenamiento
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        
        # --- AQUÍ ESTÁN TUS 15 ÉPOCAS DIRECTAS ---
        num_train_epochs=15, 
        
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=params["batch_size"],
        learning_rate=params["learning_rate"],
        
        eval_strategy="steps",
        eval_steps=200,    # Evaluamos un poco menos frecuente para ir rápido
        save_steps=500,
        logging_steps=50,
        
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collator,
    )

    print("🔥 Iniciando entrenamiento (15 Épocas)...")
    trainer.train()
    
    # 5. Guardado Final
    print(f"💾 Guardando modelo High-Res en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    plot_history(trainer.state.log_history, OUTPUT_DIR / "loss_highres.png")
    print("✅ ¡Entrenamiento completado! Ahora tu IA sabe usar pedales y rubato.")

if __name__ == "__main__":
    main()