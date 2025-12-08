import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- PATHS ---
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
output_path = os.path.join(project_root, "models/voice_model_fresh") 
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

# 🔒 POINT TO YOUR BEST MODEL HERE
# This is the file you want to "freeze" and polish
PREVIOUS_BEST_MODEL = "/home/ubuntu/digital_twin_project/models/voice_model_fresh/run-December-08-2025_09+48AM-83efe94/best_model_6534.pth"

os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

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

def train_polish():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. VITS Config
    config = VitsConfig(
        batch_size=8,
        eval_batch_size=4,
        run_eval=True,
        epochs=1000,  # We will manually stop after ~50 epochs
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-gb", # Keep this British!
        phoneme_cache_path=cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        
        # --- ❄️ THE FREEZER SETTINGS (POLISHING) ---
        # 1e-6 is extremely slow. It stops the "bouncing".
        lr=1e-6,       
        lr_gen=1e-6,   
        lr_disc=1e-6, 
        
        # Disable scheduler so it stays frozen at this low speed
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

    # 6. LOAD PREVIOUS BEST WEIGHTS
    # We load normally (no surgical deletion) because this model
    # ALREADY has the British embedding layer shape.
    print(f"⬇️  Loading checkpoint for polishing: {PREVIOUS_BEST_MODEL}")
    
    # We use strict=False to be safe, but keys should match perfectly.
    model.load_checkpoint(config, PREVIOUS_BEST_MODEL, strict=False)
    print(" -> Checkpoint loaded. Optimizer will reset to 1e-6.")

    # 7. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting POLISHING run (LR 1e-6)...")
    trainer.fit()

if __name__ == "__main__":
    train_polish()