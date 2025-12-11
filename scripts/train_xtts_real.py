import os
import csv
import json
import random
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets.formatters import *
from TTS.utils.manage import ModelManager

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
RUN_NAME = "xtts_finetuned"
OUT_PATH = os.path.join(os.getcwd(), "models") 

# --- USE THIS CORRECTED BLOCK ---
print("⏳ Verifying model path...")
manager = ModelManager()
model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")

# Tuple index 1 is the absolute path to 'config.json'. 
# We use dirname() on that file to get the correct folder.
CHECKPOINT_DIR = os.path.dirname(model_path_tuple[1])

# Data Paths
METADATA_CSV = "metadata.csv"
WAVS_DIR = "wavs"  # Assuming your audio files are in a 'wavs' subfolder
LANGUAGE = "en"
SPEAKER_NAME = "my_speaker"

# Training Hyperparameters
BATCH_SIZE = 4  # Lower if you get OutOfMemory errors (try 2)
EPOCHS = 10
LEARNING_RATE = 5e-6

# -------------------------------------------------------------------------
# HELPER: CONVERT CSV TO XTTS JSON
# -------------------------------------------------------------------------
def format_dataset(csv_file, train_json, eval_json):
    """
    Reads metadata.csv (audio_file|text) and converts it to 
    XTTS-compatible JSONL format (train.json and eval.json).
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ Could not find {csv_file}")

    print("Converting metadata.csv to XTTS JSON format...")
    
    items = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) < 2: continue
            audio_name = row[0].strip()
            text = row[1].strip()
            
            # Ensure audio path is correct
            audio_path = os.path.join(WAVS_DIR, audio_name)
            
            # Create XTTS entry
            items.append({
                "text": text,
                "audio_file": audio_path,
                "speaker_name": SPEAKER_NAME,
                "language": LANGUAGE
            })

    # Shuffle and Split (90% train, 10% eval)
    random.shuffle(items)
    split_idx = int(len(items) * 0.9)
    train_items = items[:split_idx]
    eval_items = items[split_idx:]

    # Write JSONL files
    with open(train_json, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item) + "\n")
            
    with open(eval_json, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Created {len(train_items)} training samples and {len(eval_items)} eval samples.")

# -------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# -------------------------------------------------------------------------
def main():
    # 1. Prepare Data Files
    train_json = "metadata_train.json"
    eval_json = "metadata_eval.json"
    format_dataset(METADATA_CSV, train_json, eval_json)

    # 2. Define Model Configuration
    config = XttsConfig()
    
    # Load defaults from the downloaded base model
    config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))

    # Update config for fine-tuning
    config.dataset_config.datasets = [
        BaseDatasetConfig(
            formatter="xtts",  # Uses the internal XTTS formatter
            meta_file_train=train_json,
            meta_file_val=eval_json,
            path=os.getcwd(),
            language=LANGUAGE
        )
    ]

    config.batch_size = BATCH_SIZE
    config.epochs = EPOCHS
    config.lr = LEARNING_RATE
    config.test_batch_size = 1
    
    # Paths for saving results
    config.output_path = OUT_PATH
    config.run_name = RUN_NAME

    # 3. Load the Model
    print("⬇️  Loading XTTS v2 Base Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)

    # =========================================================================
    # 🛠️ CRITICAL FIX: MONKEY-PATCH get_criterion
    # The generic Trainer expects this method, but XTTS calculates loss internally.
    # We add a dummy lambda function to prevent the AttributeError.
    # =========================================================================
    model.get_criterion = lambda: None

    if torch.cuda.is_available():
        model.cuda()

    # 4. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None, # We manually loaded checkpoint above
            skip_train_epoch=False,
            start_with_eval=False,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=None, # Auto-loaded by config
        eval_samples=None,  # Auto-loaded by config
    )

    # 5. Start Training
    print("🚀 Starting Training...")
    trainer.fit()

if __name__ == "__main__":
    main()