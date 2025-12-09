import os
import glob
import torch
import soundfile as sf
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import TTS.tts.models.vits as vits_module

# ==========================================================
# === 🚑 RE-APPLY THE MONKEYPATCH (Crucial for resuming) ===
# ==========================================================
def patched_load_audio(file_path):
    wav_numpy, sr = sf.read(file_path)
    wav_numpy = wav_numpy.astype("float32")
    wav_tensor = torch.from_numpy(wav_numpy)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    else:
        wav_tensor = wav_tensor.transpose(0, 1)
    return wav_tensor, sr

print("🚑 Re-applying 'load_audio' patch...")
vits_module.load_audio = patched_load_audio
# ==========================================================

# --- PATHS ---
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
models_dir = os.path.join(project_root, "models/voice_model_fixed_v2") # MUST match your training folder
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

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

def find_latest_checkpoint(model_dir):
    # Find all run folders
    runs = sorted(glob.glob(os.path.join(model_dir, "run-*")))
    if not runs:
        raise ValueError(f"No run folders found in {model_dir}")
    
    latest_run = runs[-1] # Pick the last one created
    print(f"📍 Found latest run: {os.path.basename(latest_run)}")
    
    # Find all checkpoints in that run
    checkpoints = sorted(glob.glob(os.path.join(latest_run, "checkpoint_*.pth")), key=os.path.getmtime)
    if not checkpoints:
        # Fallback: check for best_model.pth
        best_model = os.path.join(latest_run, "best_model.pth")
        if os.path.exists(best_model):
            return best_model
        raise ValueError(f"No checkpoints found in {latest_run}")
        
    latest_ckpt = checkpoints[-1]
    print(f"♻️  Resuming from: {os.path.basename(latest_ckpt)}")
    return latest_ckpt

def resume_training():
    # 1. Locate the checkpoint
    checkpoint_path = find_latest_checkpoint(models_dir)

    # 2. Config (Must be identical to training)
    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    config = VitsConfig(
        batch_size=32,
        eval_batch_size=16,
        run_eval=True,
        epochs=1000, 
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-gb", 
        phoneme_cache_path=cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=models_dir, # Resume in same parent folder
        datasets=[dataset_config],
        num_loader_workers=4, 
        num_eval_loader_workers=2,
        
        # Keep the FAST learning rate
        lr=2e-4,       
        lr_gen=2e-4,   
        lr_disc=2e-4,  
        lr_scheduler=None, 
    )

    # Apply Audio Fixes
    config.audio.sample_rate = 22050
    config.audio.max_wav_value = 1.0
    config.audio.do_trim_silence = True
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None

    # 3. Initialize Model & Processor
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    
    # Init empty model (weights will be overwritten by checkpoint)
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 4. Load Data
    print("Loading data samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # 5. Initialize Trainer with restore_path
    trainer = Trainer(
        TrainerArgs(restore_path=checkpoint_path), # <--- THIS IS THE MAGIC
        config,
        output_path=models_dir,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Resuming Training...")
    trainer.fit()

if __name__ == "__main__":
    resume_training()