import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()

# 1. POINT THIS TO YOUR BEST MODEL FILE
# (Check the filename in your folder before running!)
PREVIOUS_BEST_MODEL = os.path.join(
    PROJECT_ROOT, 
    "models/voice_model_large/run-December-08-2025_02+11PM-de2b618/best_model.pth"
)

# 2. Paths
DATASET_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/voice_model_large") # Keep same root
CACHE_PATH = os.path.join(PROJECT_ROOT, "audio_data/phoneme_cache")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    if not os.path.exists(manifest_path): return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines: return []
    delimiter = "|" if "|" in lines[0] else ","
    for line in lines:
        line = line.strip()
        if not line: continue
        cols = line.split(delimiter)
        if len(cols) >= 2:
            wav_filename = cols[0].strip()
            text = delimiter.join(cols[1:]).strip()
            wav_path = os.path.join(root_path, wav_filename)
            if not os.path.exists(wav_path):
                wav_path = os.path.join(root_path, "wavs", wav_filename)
            if os.path.exists(wav_path):
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "ljspeech", 
                    "root_path": root_path
                })
    return items

def train_polish_large():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=DATASET_PATH
    )

    # 2. VITS Config (Freezer Mode)
    config = VitsConfig(
        # Keep G5 speeds
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        
        run_eval=True,
        epochs=1000,  # Run for ~50 epochs then stop manually
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-gb", # British!
        phoneme_cache_path=CACHE_PATH,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=OUTPUT_PATH,
        datasets=[dataset_config],
        
        # --- ❄️ THE FREEZER ---
        # 1e-6 forces the model to stop oscillating and settle
        lr=1e-6,       
        lr_gen=1e-6,   
        lr_disc=1e-6,  
        lr_scheduler=None,
    )

    # 3. Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # 4. Load Data
    print("Loading data samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # 5. Initialize Model
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 6. LOAD PREVIOUS WEIGHTS
    if not os.path.exists(PREVIOUS_BEST_MODEL):
        print(f"❌ Error: Could not find model file at {PREVIOUS_BEST_MODEL}")
        return

    print(f"⬇️  Loading model for polishing: {PREVIOUS_BEST_MODEL}")
    model.load_checkpoint(config, PREVIOUS_BEST_MODEL, strict=False)
    print(" -> Weights loaded. Optimizer reset to 1e-6.")

    # 7. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting POLISHING run (1e-6)...")
    trainer.fit()

if __name__ == "__main__":
    train_polish_large()