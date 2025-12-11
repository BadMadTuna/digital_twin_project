import os
import csv
import json
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

# ==========================================================
# === CONFIGURATION ===
# ==========================================================
PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset")
WAVS_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_FILE = os.path.join(DATASET_PATH, "metadata.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/xtts_finetuned")

# XTTS Training Params
EPOCHS = 10           # XTTS learns extremely fast. 6-10 is usually enough.
BATCH_SIZE = 4        # Keep low to avoid OOM
GRAD_ACC = 1          # Gradient accumulation
LR = 5e-6             # Very low learning rate is mandatory for XTTS
# ==========================================================

os.environ["COQUI_TOS_AGREED"] = "1"

def format_dataset():
    """
    Converts your VITS metadata.csv to the JSON format XTTS requires.
    """
    print("Converting metadata.csv to XTTS JSON format...")
    train_json_path = os.path.join(PROJECT_ROOT, "xtts_train.json")
    eval_json_path = os.path.join(PROJECT_ROOT, "xtts_eval.json")

    data = []
    
    # Check if metadata exists
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Error: {METADATA_FILE} not found.")
        return None, None

    # Read Metadata
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        # Detect delimiter
        line = f.readline()
        f.seek(0)
        delimiter = "|" if "|" in line else ","
        
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if len(row) < 2: continue
            
            filename = row[0].strip()
            text = delimiter.join(row[1:]).strip()
            
            # Resolve Path
            full_wav_path = os.path.join(WAVS_PATH, filename)
            if not os.path.exists(full_wav_path):
                # Try root
                full_wav_path = os.path.join(DATASET_PATH, filename)
            
            if os.path.exists(full_wav_path):
                # XTTS JSON Format
                data.append({
                    "audio_file": full_wav_path,
                    "text": text,
                    "speaker_name": "my_voice",
                    "language": "en"
                })

    # Split Train/Eval (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    # Save JSONs
    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)
    
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4)
        
    print(f"✅ Created {len(train_data)} training samples and {len(eval_data)} eval samples.")
    return train_json_path, eval_json_path

def main():
    # 1. Prepare Data
    train_json, eval_json = format_dataset()
    if not train_json: return

    # 2. Download Base Model
    print("⬇️  Loading XTTS v2 Base Model...")
    ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    checkpoint_dir = os.path.join(os.path.expanduser("~/.local/share/tts"), "tts_models--multilingual--multi-dataset--xtts_v2")
    
    # If the standard download path differs, find it dynamically
    if not os.path.exists(checkpoint_dir):
        # Fallback search
        import glob
        possible = glob.glob(os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"))
        if possible: checkpoint_dir = possible[0]
        else:
             # Last resort: let the config downloader handle it, but define paths manually if needed
             pass

    # 3. Configure XTTS
    config = XttsConfig()
    config.load_json(os.path.join(checkpoint_dir, "config.json"))

    # Override for Fine-Tuning
    config.epochs = EPOCHS
    config.batch_size = BATCH_SIZE
    config.grad_accum_steps = GRAD_ACC
    config.lr = LR
    config.optimizer = "AdamW"
    config.save_step = 500
    config.output_path = OUTPUT_PATH
    
    # IMPORTANT: Point to our custom JSONs
    config.train_datasets = [
        {"name": "custom", "path": "", "meta_file_train": train_json, "language": "en"}
    ]
    config.eval_datasets = [
        {"name": "custom", "path": "", "meta_file_val": eval_json, "language": "en"}
    ]

    # 4. Initialize Model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir, eval=True)

    # 5. Initialize GPT Trainer
    # NOTE: XTTS uses a specialized training routine often, but the Trainer class 
    # in newer Coqui versions handles the "GPT" vs "VITS" switch internally based on config.
    # If this fails, we switch to the explicit GPTTrainer import.
    
    from trainer import Trainer, TrainerArgs
    
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=None, # XTTS loader handles this via config.train_datasets
        eval_samples=None,
    )

    print("🚀 Starting XTTS Fine-Tuning...")
    # XTTS fine-tuning is quirky. It ignores 'train_samples' arg in the trainer 
    # and re-loads from the config JSONs internally.
    trainer.fit()

if __name__ == "__main__":
    main()