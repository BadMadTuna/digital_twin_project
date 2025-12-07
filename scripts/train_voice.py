import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

# --- Configuration ---
dataset_path = os.path.join(os.getcwd(), "audio_data/dataset")
output_path = os.path.join(os.getcwd(), "models/voice_model")

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

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

def download_pretrained_model():
    """Downloads the standard LJSpeech VITS model to use as a base."""
    print(" -> Downloading pre-trained VITS model (LJSpeech)...")
    manager = ModelManager()
    model_name = "tts_models/en/ljspeech/vits"
    model_path, config_path, _ = manager.download_model(model_name)
    return model_path, config_path

def train_model():
    print(f"Initializing Fine-Tuning using dataset at: {dataset_path}")

    # 1. Download Base Model
    pretrained_model_path, pretrained_config_path = download_pretrained_model()
    print(f" -> Base model loaded from: {pretrained_model_path}")
    print(f" -> Base config loaded from: {pretrained_config_path}")

    # 2. Define Dataset Configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 3. Load the EXACT configuration from the pre-trained model
    # This fixes the "size mismatch" because it enables phonemes automatically
    config = load_config(pretrained_config_path)

    # 4. Override specific settings for our training
    config.output_path = output_path
    config.datasets = [dataset_config] # Point to OUR data
    config.batch_size = 8
    config.epochs = 100
    config.test_sentences = [
        "Hello, this is my digital twin speaking.",
        "I can generate new audio from text now."
    ]
    
    # 5. Initialize Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # 6. Load Data Samples (using our custom formatter)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=0.1,
        eval_split_size=0.1,
        formatter=custom_formatter 
    )

    # 7. Initialize Model & Load Pre-trained Weights
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    print(" -> Loading pre-trained weights (Transfer Learning)...")
    # strict=False is generally safer for fine-tuning, but now shapes should match perfectly
    model.load_checkpoint(config, pretrained_model_path, strict=False)

    # 8. Initialize Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    # 9. Start Training
    print("Starting Fine-Tuning...")
    trainer.fit()

if __name__ == "__main__":
    train_model()