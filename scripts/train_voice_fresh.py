import os
import torch
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager 

# --- PATHS ---
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
output_path = os.path.join(project_root, "models/voice_model_fixed") 
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

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

def train_fresh_large():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. VITS Config (Initialize with defaults first)
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
        output_path=output_path,
        datasets=[dataset_config],
        num_loader_workers=4, 
        num_eval_loader_workers=2,
        lr=5e-5,       
        lr_gen=5e-5,   
        lr_disc=5e-5,  
        lr_scheduler=None, 
    )

    # --- THE FIX: MANUALLY OVERRIDE AUDIO SETTINGS ---
    # We set these directly on the config object to bypass the constructor error.
    config.audio.sample_rate = 22050
    config.audio.max_wav_value = 1.0   # <--- Forces Float (0.0 - 1.0) handling
    config.audio.do_trim_silence = True
    config.audio.mel_fmin = 0
    config.audio.mel_fmax = None
    # -------------------------------------------------

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

    # 6. DOWNLOAD & SURGICAL LOAD
    print("⬇️  Downloading/Loading LJSpeech base model...")
    manager = ModelManager()
    model_path, _, _ = manager.download_model("tts_models/en/ljspeech/vits")
    
    print(f" -> Loading weights from: {model_path}")
    
    # --- SURGICAL LOADING BLOCK ---
    checkpoint = torch.load(model_path, map_location="cpu")
    model_state = checkpoint["model"]

    # Remove mismatched embedding layer
    bad_keys = []
    for key in model_state.keys():
        if "text_encoder.emb.weight" in key:
            bad_keys.append(key)
    
    for key in bad_keys:
        print(f"   ! Removing mismatched layer: {key}")
        del model_state[key]

    model.load_state_dict(model_state, strict=False)
    print(" -> Surgical load complete. Embeddings reset for UK English.")
    # ------------------------------

    # 7. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting FIXED Training Run...")
    trainer.fit()

if __name__ == "__main__":
    train_fresh_large()