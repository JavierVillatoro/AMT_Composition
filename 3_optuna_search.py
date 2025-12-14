import optuna
import torch
import json
import logging
import sys
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from miditok import REMI
from pathlib import Path
from torch.utils.data import Dataset, random_split
from torch.nn.utils.rnn import pad_sequence

# --- CONFIGURACIÓN OPTUNA ---
N_TRIALS = 15          
SUBSET_PERCENT = 0.15  
DATA_DIR = Path("./data_processed")

# --- 1. DATASET PERSONALIZADO ---
class ChopinDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Devolvemos tensores Long (enteros) listos para la GPU
        return {"input_ids": self.data[idx].long()}

# --- 2. COLLATOR PERSONALIZADO ---
def custom_collator(features):
    """
    Toma una lista de muestras y crea un batch.
    Reemplaza al DataCollator de Hugging Face para evitar incompatibilidad con miditok.
    """
    batch_tensors = [f["input_ids"] for f in features]
    input_ids = pad_sequence(batch_tensors, batch_first=True, padding_value=0)
    return {"input_ids": input_ids, "labels": input_ids}

def get_dataset_and_tokenizer():
    tokenizer = REMI.from_pretrained(DATA_DIR / "tokenizer.json")
    
    try:
        data = torch.load(DATA_DIR / "dataset.pt", weights_only=False)
    except TypeError:
        data = torch.load(DATA_DIR / "dataset.pt")
    
    subset_size = int(len(data) * SUBSET_PERCENT)
    data = data[:subset_size]
    
    full_dataset = ChopinDataset(data)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    return train_data, val_data, len(tokenizer.vocab)

def objective(trial):
    # --- HIPERPARÁMETROS ---
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8]) 
    n_layer = trial.suggest_int("n_layer", 4, 8)
    n_head = trial.suggest_categorical("n_head", [4, 8])
    n_embd = trial.suggest_categorical("n_embd", [256, 512])

    print(f"\n🔎 Trial {trial.number}: LR={learning_rate:.5f} | Layers={n_layer} | Embed={n_embd}")

    train_dataset, val_dataset, vocab_size = get_dataset_and_tokenizer()
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=0, 
        eos_token_id=0,
        activation_function="gelu_new",
    )
    model = GPT2LMHeadModel(config)

    # Configuración de Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./optuna_temp/trial_{trial.number}",
        num_train_epochs=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="steps",
        eval_steps=40,
        save_strategy="no",
        logging_steps=40,
        fp16=torch.cuda.is_available(), # Activa FP16 si hay GPU
        report_to="none",
        dataloader_num_workers=0 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collator, 
    )

    try:
        trainer.train()
        eval_metrics = trainer.evaluate()
        loss = eval_metrics["eval_loss"]
        
        # Limpieza
        del model
        del trainer
        torch.cuda.empty_cache()
        
        return loss
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("⚠️ OOM: Saltando este trial...")
            torch.cuda.empty_cache()
            return float("inf")
        raise e

if __name__ == "__main__":
    # --- 🔍 VERIFICACIÓN DE GPU AL INICIO ---
    print("\n" + "="*60)
    print("🛠️  COMPROBACIÓN DE HARDWARE")
    print("="*60)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU DETECTADA: {gpu_name}")
        print(f"✅ Versión CUDA: {torch.version.cuda}")
        print("🚀 El entrenamiento utilizará aceleración por GPU.")
    else:
        print("⚠️  ATENCIÓN: NO SE DETECTÓ GPU.")
        print("🐢 El entrenamiento se ejecutará en CPU (será muy lento).")
        print("ℹ️  Si tienes una tarjeta NVIDIA, revisa la instalación de PyTorch.")
    
    print("="*60 + "\n")
    # ---------------------------------------

    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    print("🚀 Iniciando búsqueda de hiperparámetros con Optuna...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*50)
    print("🎉 ¡MEJORES PARÁMETROS ENCONTRADOS!")
    print("="*50)
    best_params = study.best_params
    print(json.dumps(best_params, indent=4))
    
    with open("best_params.json", "w") as f:
        json.dump(best_params, f)
    
    print("\n💾 Guardado en 'best_params.json'.")