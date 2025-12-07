import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- Configuration ---
dataset_path = os.path.join(os.getcwd(), "audio_data/dataset")
output_path = os.path.join(os.getcwd(), "models/voice_model")
cache_path = os.path.join(os.getcwd(), "audio_data/phoneme_cache")

# ==========================================
#  UPDATE PATH TO YOUR LATEST CHECKPOINT (EPOCH 25/75)
# ==========================================
PREVIOUS_CHECKPOINT = "/home/ubuntu/digital_twin_project/models/voice_model/run-December-07-2025_11+08AM-7966162/best_model.pth" 
PREVIOUS_CONFIG     = "/home/ubuntu/digital_twin_project/models/voice_model/run-December-07-2025_11+08AM-7966162/config.json"
# ==========================================

# Ensure output and cache directories exist
os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Reads a CSV file and auto-detects if it uses | or , as a separator.
    """
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    
    if not os.path.exists(manifest_path):
        return []
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        return []

    first_line = lines[0].strip()
    delimiter = "|"
    if "|" not in first_line and "," in first_line:
        delimiter = ","

    for line in lines:
        line = line.strip()
        if not line: continue
        cols = line.split(delimiter)
        if len(cols) >= 2:
            wav_filename = cols[0].strip()
            text = delimiter.join(cols[1:]).strip()
            
            wav_path = os.path.join(root_path, wav_filename)
            if not os.path.exists(wav_path):
                wav_path_subdir = os.path.join(root_path, "wavs", wav_filename)
                if os.path.exists(wav_path_subdir):
                    wav_path = wav_path_subdir
            
            if os.path.exists(wav_path):
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "ljspeech", 
                    "root_path": root_path
                })
    return items

def train_model():
    print(f"Resuming with Low Learning Rate from: {PREVIOUS_CHECKPOINT}")

    # 1. Define Dataset Configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. Load the configuration from the PREVIOUS RUN
    config = load_config(PREVIOUS_CONFIG)

    # 3. Update settings for the "Cool Down" run
    config.output_path = output_path
    config.datasets = [dataset_config]
    config.batch_size = 8
    config.epochs = 100         # Give it time to settle
    config.phoneme_cache_path = cache_path
    
    # --- CRITICAL: LOWER THE LEARNING RATE ---
    # Default is 2e-4 (0.0002). We set it to 5e-5 (0.00005).
    # This prevents the model from "overshooting" and helps smooth out robotic noise.
# Update the top-level variable
    config.lr = 0.00005 
    
    # ALSO update the internal dictionary (This is the one ignoring you!)
if hasattr(config, "optimizer_params"):
        if "lr" in config.optimizer_params:
            del config.optimizer_params["lr"]
            print(" -> REMOVED 'lr' from optimizer_params to prevent conflict.")
    
    # 4. Initialize Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # 5. Load Data Samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
        formatter=custom_formatter 
    )

    # 6. Initialize Model 
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    print(" -> Loading previous checkpoint weights...")
    model.load_checkpoint(config, PREVIOUS_CHECKPOINT, strict=False)

    # 7. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # 8. Start Training
    print("Starting Cool-Down Training...")
    trainer.fit()

if __name__ == "__main__":
    if not os.path.exists(PREVIOUS_CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {PREVIOUS_CHECKPOINT}")
        print("Please edit the script and check the path.")
    else:
        train_model()