import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.config import load_config  # <--- NEW IMPORT

# === CONFIGURATION ===
PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/xtts_finetuned")
METADATA_FILE = "metadata.csv"
LANGUAGE = "en"
# =====================

# Agree to license
os.environ["COQUI_TOS_AGREED"] = "1"

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Adapts your VITS metadata format (wav|text) to XTTS format (wav|text|speaker|lang)
    """
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2: continue
            
            wav_file = parts[0].strip()
            text = "|".join(parts[1:]).strip()
            
            # Find the full path
            wav_path = os.path.join(root_path, wav_file)
            if not os.path.exists(wav_path):
                 wav_path = os.path.join(root_path, "wavs", wav_file)

            if os.path.exists(wav_path):
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "my_voice",  # XTTS needs a speaker ID
                    "language": LANGUAGE,
                    "root_path": root_path
                })
    return items

def train_xtts():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter",
        meta_file_train=METADATA_FILE,
        path=DATASET_PATH,
        language=LANGUAGE
    )

    # 2. DEFINE PATHS (Model already downloaded)
    # Using the path confirmed by your logs:
    checkpoint_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    
    model_path = os.path.join(checkpoint_dir, "model.pth")
    config_path = os.path.join(checkpoint_dir, "config.json")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    speaker_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

    # 3. LOAD & UPDATE CONFIG
    # We load the existing config to match the model architecture exactly
    print(f"⚙️  Loading Config from: {config_path}")
    config = load_config(config_path)

    # --- FIX: Inject missing attribute for BaseTTS compatibility ---
    # The generic trainer checks this flag to decide how to handle speaker IDs,
    # but XTTS config doesn't have it by default.
    config.model_args.use_speaker_embedding = True
    # ---------------------------------------------------------------

    # Update the loaded config with your training preferences
    config.batch_size = 8
    config.eval_batch_size = 2
    config.num_loader_workers = 2
    config.num_eval_loader_workers = 1
    config.run_eval = True
    config.test_delay_epochs = -1
    config.epochs = 15
    config.text_cleaner = "whitespace_cleaner"
    config.use_phonemes = False
    config.print_step = 50
    config.print_eval = True
    config.save_step = 500
    config.output_path = OUTPUT_PATH
    config.datasets = [dataset_config]  # Important: Attach your dataset here

    # 4. Load Model
    print("⬇️  Loading XTTS v2 Base...")
    model = Xtts.init_from_config(config)
    
    print(f"📂 Loading Checkpoint from: {checkpoint_dir}")
    model.load_checkpoint(
        config, 
        checkpoint_path=model_path, 
        vocab_path=vocab_path, 
        speaker_file_path=speaker_path,
        eval=True,
        strict=False  # <--- CRITICAL: Prevents crash on minor metadata mismatches
    )

    # 5. Load Data
    print("📂 Loading Data...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # --- FIX: Patch the model to avoid AttributeErrors ---
    
    # 1. Patch 'get_criterion'
    def get_criterion():
        return None
    model.get_criterion = get_criterion

    # 2. Patch 'save_ids_to_file' for SpeakerManager
    if hasattr(model, "speaker_manager") and model.speaker_manager:
        def save_speaker_ids(path):
            pass # Do nothing
        model.speaker_manager.save_ids_to_file = save_speaker_ids

    # 3. Patch 'save_ids_to_file' for LanguageManager (NEW FIX)
    if hasattr(model, "language_manager") and model.language_manager:
        def save_language_ids(path):
            pass # Do nothing
        model.language_manager.save_ids_to_file = save_language_ids
    
    # -----------------------------------------------------

    # 6. Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting XTTS Fine-Tuning...")
    trainer.fit()

if __name__ == "__main__":
    train_xtts()