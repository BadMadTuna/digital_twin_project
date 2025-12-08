import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- PATHS ---
# We use a NEW output folder to avoid mixing with old attempts
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
output_path = os.path.join(project_root, "models/voice_model_fresh") 
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    """ Reads metadata.csv with | or , delimiter """
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines: return []
    
    # Detect delimiter
    delimiter = "|" if "|" in lines[0] else ","
    
    for line in lines:
        line = line.strip()
        if not line: continue
        cols = line.split(delimiter)
        if len(cols) >= 2:
            wav_filename = cols[0].strip()
            text = delimiter.join(cols[1:]).strip()
            wav_path = os.path.join(root_path, wav_filename)
            # Handle wavs/ subdirectory
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

def train_fresh():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. VITS Architecture Configuration
    config = VitsConfig(
        batch_size=8,
        eval_batch_size=4,
        batch_group_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,  # Set high, we rely on Early Stopping or Scheduler
        text_cleaner="english_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        
        # --- 🧠 THE SCHEDULER MAGIC ---
        # Start at standard speed
        lr=2e-4, 
        
        # Enable Scheduler
        lr_scheduler="StepLR", 
        
        # Decay: Every 10 epochs, multiply LR by 0.9 (slow down by 10%)
        # By epoch 100, LR will be ~6e-5 (perfect for polishing)
        lr_scheduler_params={"step_size": 10, "gamma": 0.9}, 
    )

    # 3. Initialize Audio
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

    # 5. Initialize Model (Fresh Weights)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # 6. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting FRESH training with LR Scheduler...")
    trainer.fit()

if __name__ == "__main__":
    train_fresh()