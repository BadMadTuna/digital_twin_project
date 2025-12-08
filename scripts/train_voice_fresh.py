import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager # Needed to download base model

# --- PATHS ---
project_root = os.getcwd()
dataset_path = os.path.join(project_root, "audio_data/dataset")
output_path = os.path.join(project_root, "models/voice_model_fresh") 
cache_path = os.path.join(project_root, "audio_data/phoneme_cache")

os.makedirs(output_path, exist_ok=True)
os.makedirs(cache_path, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    # (Same formatter code as before - keeping it brief here)
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

def train_fresh():
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
        epochs=1000, 
        text_cleaner="english_cleaners",
        use_phonemes=True,
        
        # ✅ FIX 1: UK ENGLISH
        # Use "en-gb" for British pronunciation compatibility
        phoneme_language="en-gb", 
        
        phoneme_cache_path=cache_path,
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        
        # ✅ SCHEDULER (Kept from previous advice)
        lr=2e-4, 
        lr_scheduler="StepLR", 
        lr_scheduler_params={"step_size": 10, "gamma": 0.9}, 
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

    # ✅ FIX 2: DOWNLOAD & LOAD BASE MODEL (LJSpeech)
    # We load the weights, but NOT the optimizer states (strict=False).
    # This gives us a smart brain, but a fresh training start.
    print("⬇️  Downloading/Loading LJSpeech base model...")
    manager = ModelManager()
    model_path, _, _ = manager.download_model("tts_models/en/ljspeech/vits")
    
    print(f" -> Loading weights from: {model_path}")
    model.load_checkpoint(config, model_path, strict=False)

    # 6. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting FRESH Transfer Learning (British)...")
    trainer.fit()

if __name__ == "__main__":
    train_fresh()