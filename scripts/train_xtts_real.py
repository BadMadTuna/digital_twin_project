import os
import csv
import json
import random
import torch
from TTS.utils.manage import ModelManager
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets.formatters import *
from TTS.utils.audio import AudioProcessor

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
RUN_NAME = "xtts_finetuned"
OUT_PATH = os.path.join(os.getcwd(), "models")
METADATA_CSV = "metadata.csv"
WAVS_DIR = "audio_data/dataset/wavs"
LANGUAGE = "en"
SPEAKER_NAME = "my_speaker"

# Training Hyperparameters
BATCH_SIZE = 4  # Set to 2 if you hit OutOfMemory errors
EPOCHS = 10
LEARNING_RATE = 5e-6

# -------------------------------------------------------------------------
# HELPER 1: CONVERT CSV TO XTTS JSON
# -------------------------------------------------------------------------
def format_dataset(csv_file, train_json, eval_json):
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
            audio_path = os.path.join(WAVS_DIR, audio_name)
            
            # Create XTTS entry
            items.append({
                "text": text,
                "audio_file": audio_path,
                "audio_unique_name": audio_name,  # <--- ADD THIS LINE
                "speaker_name": SPEAKER_NAME,
                "language": LANGUAGE
            })

    random.shuffle(items)
    split_idx = int(len(items) * 0.9)
    train_items = items[:split_idx]
    eval_items = items[split_idx:]

    with open(train_json, "w", encoding="utf-8") as f:
        for item in train_items:
            f.write(json.dumps(item) + "\n")
            
    with open(eval_json, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Created {len(train_items)} training samples and {len(eval_items)} eval samples.")

# -------------------------------------------------------------------------
# HELPER 2: LOAD JSON DATA (Prevents NoneType error)
# -------------------------------------------------------------------------
def load_json_data(json_file):
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# -------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# -------------------------------------------------------------------------
def main():
    # 1. Prepare Data Files
    train_json = "metadata_train.json"
    eval_json = "metadata_eval.json"
    format_dataset(METADATA_CSV, train_json, eval_json)

    # 2. Locate Model Path (Robust Method)
    print("⏳ Verifying model path...")
    manager = ModelManager()
    model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    
    # Handle tuple return safely
    model_path = model_path_tuple[0]
    if os.path.isfile(model_path):
        CHECKPOINT_DIR = os.path.dirname(model_path)
    else:
        CHECKPOINT_DIR = model_path
        
    print(f"✅ Found model at: {CHECKPOINT_DIR}")

    # 3. Configure Model
    config = XttsConfig()
    config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))

    config.datasets = [
        BaseDatasetConfig(
            formatter="xtts",
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
    config.output_path = OUT_PATH
    config.run_name = RUN_NAME

    # 4. Load Model (BEFORE Patching)
    print("⬇️  Loading XTTS v2 Base Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # 🛠️ COMPATIBILITY PATCHES (Monkey-Patching)
    # =========================================================================
    
    # Patch 1: Loss Criterion
    model.get_criterion = lambda: None

    # Patch 2: Speaker Manager
    if model.speaker_manager is not None:
        model.speaker_manager.save_ids_to_file = lambda x: None
        
        # 🛠️ NUCLEAR FIX: Delete the class property so we can set a simple dict
        # This resolves the "can't set attribute" and "dict_keys" errors once and for all.
        if hasattr(type(model.speaker_manager), "name_to_id"):
            delattr(type(model.speaker_manager), "name_to_id")
            
        model.speaker_manager.name_to_id = {SPEAKER_NAME: 0}

    # Patch 3: Language Manager
    if model.language_manager is not None:
        model.language_manager.save_ids_to_file = lambda x: None

    # Patch 4: Tokenizer
    if model.tokenizer is not None:
        model.tokenizer.use_phonemes = False
        # Silence print_logs to prevent AttributeError
        model.tokenizer.print_logs = lambda *args, **kwargs: None

        # 🛠️ NEW FIX: Map 'text_to_ids' to the internal 'encode' method
        # We hardcode the language here because the generic loader doesn't pass it.
        model.tokenizer.text_to_ids = lambda t: model.tokenizer.encode(t, lang=LANGUAGE)

    # Patch 5: Config Attributes (Inject missing flags)
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False

    # Patch 6: Audio Processor
    # We explicitly provide FFT settings to prevent the "NoneType" division error.
    if model.ap is None:
        model.ap = AudioProcessor(
            sample_rate=22050,
            win_length=1024,
            hop_length=256,
            num_mels=80,
            n_fft=1024,
            do_trim_silence=True
        )
    
    # =========================================================================

    # 5. Load Data manually to avoid internal loader errors
    print("⏳ Loading data samples...")
    train_samples = load_json_data(train_json)
    eval_samples = load_json_data(eval_json)

    # 6. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(
            restore_path=None, 
            skip_train_epoch=False,
            start_with_eval=False,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples, # Explicitly pass data
        eval_samples=eval_samples,   # Explicitly pass data
    )

    # 7. Start Training
    print("🚀 Starting Training...")
    trainer.fit()

if __name__ == "__main__":
    main()