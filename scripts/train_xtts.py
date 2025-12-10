import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

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
                    "language": LANGUAGE
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

    # 2. XTTS Configuration
    config = XttsConfig(
        batch_size=4,   # XTTS is heavy, keep batch size low (4-8)
        eval_batch_size=2,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=15,      # <--- XTTS converges VERY fast. 15 is usually enough.
        text_cleaner="whitespace_cleaner",
        use_phonemes=False, # XTTS uses GPT-2 BPE, not phonemes
        language=LANGUAGE,
        print_step=50,
        print_eval=True,
        save_step=500,
        output_path=OUTPUT_PATH,
        datasets=[dataset_config],
    )

    # 3. Load Model
    print("⬇️  Loading XTTS v2 Base...")
    model = Xtts.init_from_config(config)
    
    # Download the checkpoint manually to load pre-trained weights
    manager = ModelManager()
    model_path, config_path, vocab_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    
    model.load_checkpoint(config, checkpoint_path=model_path, vocab_path=vocab_path, eval=True)

    # 4. Load Data
    print("📂 Loading Data...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # 5. Trainer
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