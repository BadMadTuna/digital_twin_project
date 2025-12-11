import os
import csv
import json
import random
import torch
import types
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets.formatters import *

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
RUN_NAME = "xtts_finetuned"
OUT_PATH = os.path.join(os.getcwd(), "models")
METADATA_CSV = "metadata.csv"

# Path to audio files
WAVS_DIR = "audio_data/dataset/wavs" 

LANGUAGE = "en"
SPEAKER_NAME = "my_speaker"

# Training Hyperparameters
BATCH_SIZE = 2 
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
            
            # Create XTTS entry with 'audio_unique_name'
            items.append({
                "text": text,
                "audio_file": audio_path,
                "audio_unique_name": audio_name, 
                "speaker_name": SPEAKER_NAME,
                "language": LANGUAGE
            })

    # Shuffle and Split
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
# HELPER 2: LOAD JSON DATA
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

    # 2. Locate Model Path
    print("⏳ Verifying model path...")
    manager = ModelManager()
    model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    
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

    # 4. Load Model
    print("⬇️  Loading XTTS v2 Base Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
    if torch.cuda.is_available():
        model.cuda()

    # =========================================================================
    # 🛠️ COMPATIBILITY PATCHES
    # =========================================================================
    
    # Patch 1: Loss Criterion
    model.get_criterion = lambda: None

    # Patch 2: Speaker Manager (Nuclear Fix)
    if model.speaker_manager is not None:
        model.speaker_manager.save_ids_to_file = lambda x: None
        if hasattr(type(model.speaker_manager), "name_to_id"):
            delattr(type(model.speaker_manager), "name_to_id")
        model.speaker_manager.name_to_id = {SPEAKER_NAME: 0}

    # Patch 3: Language Manager
    if model.language_manager is not None:
        model.language_manager.save_ids_to_file = lambda x: None

    # Patch 4: Tokenizer
    if model.tokenizer is not None:
        model.tokenizer.use_phonemes = False
        model.tokenizer.print_logs = lambda *args, **kwargs: None
        model.tokenizer.text_to_ids = lambda t: model.tokenizer.encode(t, lang=LANGUAGE)

    # Patch 5: Config Attributes
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False
    config.r = 1

    # Patch 6: Audio Processor
    if model.ap is None:
        model.ap = AudioProcessor(
            sample_rate=22050,
            num_mels=80,
            do_trim_silence=True,
            n_fft=1024,
            win_length=1024,
            hop_length=256
        )

    # Patch 7: Train Step (Targeting GPT Logic)
    # The standard forward() is blocked. We must call the specific GPT training method.
    def patched_train_step(self, batch, criterion=None):
        # 1. Key Mapping: Fix generic keys to what XTTS expects
        if "text_input" in batch:
            batch["text_inputs"] = batch.pop("text_input")
        if "text_lengths" in batch:
            batch["text_lengths"] = batch.pop("text_lengths")
        if "audio_lengths" in batch:
            batch["audio_lengths"] = batch.pop("audio_lengths")
            
        # 2. Call the specific GPT training logic
        # This calculates the loss for the language model part of XTTS.
        return self.gpt_train_step(batch)
    
    # Bind the new method to the instance
    model.train_step = types.MethodType(patched_train_step, model)
    
    # =========================================================================

    # 5. Load Data
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
        train_samples=train_samples, 
        eval_samples=eval_samples,   
    )

    # 7. Start Training
    print("🚀 Starting Training...")
    trainer.fit()

if __name__ == "__main__":
    main()