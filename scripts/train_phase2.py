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
# === 🚑 THE MONKEYPATCH (Bypasses TorchCodec errors) ===
# ==========================================================
def patched_load_audio(file_path):
    """
    Directly loads audio as Float32 using soundfile, ignoring Torchaudio issues.
    """
    wav_numpy, sr = sf.read(file_path)
    wav_numpy = wav_numpy.astype("float32")
    wav_tensor = torch.from_numpy(wav_numpy)
    
    # Fix Dimensions: (Time) -> (1, Time)
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    else:
        wav_tensor = wav_tensor.transpose(0, 1)
        
    return wav_tensor, sr

print("🚑 Applying 'load_audio' patch...")
vits_module.load_audio = patched_load_audio
# ==========================================================


# --- PATHS ---
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
output_path = os.path.join(project_root, "models/voice_model")  # Matches your training folder
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Parses the metadata.csv file.
    Expected format: wav_file_name|transcription
    """
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    if not os.path.exists(manifest_path): 
        print(f"Error: Manifest not found at {manifest_path}")
        return []
        
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
    """
    Automatically finds the latest checkpoint to resume from.
    """
    # Find all run folders
    runs = sorted(glob.glob(os.path.join(model_dir, "run-*")))
    if not runs:
        print(f"❌ No run folders found in {model_dir}. Cannot resume.")
        return None
    
    latest_run = runs[-1]
    print(f"📍 Found latest run: {os.path.basename(latest_run)}")
    
    # Find checkpoints inside the latest run
    checkpoints = sorted(glob.glob(os.path.join(latest_run, "checkpoint_*.pth")), key=os.path.getmtime)
    if not checkpoints:
        # Fallback: check for best_model.pth if no checkpoints exist yet
        best_model = os.path.join(latest_run, "best_model.pth")
        if os.path.exists(best_model):
            print(f"♻️  Resuming from best_model.pth")
            return best_model
        return None
        
    latest_ckpt = checkpoints[-1]
    print(f"♻️  Resuming from: {os.path.basename(latest_ckpt)}")
    return latest_ckpt

def train_phase2():
    # 1. Locate Checkpoint
    checkpoint_path = find_latest_checkpoint(output_path)
    if not checkpoint_path:
        print("❌ CRITICAL: No checkpoint found. Please run the fresh training script first.")
        return

    # 2. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter", 
        meta_file_train="metadata.csv", 
        path=dataset_path
    )

    # 3. VITS Config (PHASE 2)
    config = VitsConfig(
        batch_size=32,
        eval_batch_size=16,
        run_eval=True,
        epochs=10000, # Set high to keep going indefinitely
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-gb", 
        phoneme_cache_path=cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        
        num_loader_workers=4, 
        num_eval_loader_workers=2,
        
        # --- PHASE 2 LEARNING RATES (LOWERED) ---
        lr=2e-5,       # 10x lower than Phase 1
        lr_gen=2e-5,   
        lr_disc=2e-5,  
        lr_scheduler=None, 
    )

    # Audio Overrides
    config.audio.sample_rate = 22050
    config.audio.max_wav_value = 1.0
    config.audio.do_trim_silence = True
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None

    # 4. Init Model
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    # Load model structure (weights will be overwritten by checkpoint)
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 5. Load Data
    print("Loading data samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # 6. Trainer with Restore
    trainer = Trainer(
        TrainerArgs(restore_path=checkpoint_path), # <--- This resumes training
        config,
        output_path=output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting PHASE 2 (Fine-Tuning) Training...")
    trainer.fit()

if __name__ == "__main__":
    train_phase2()