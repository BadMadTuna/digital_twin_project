import os
import sys
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- Configuration ---
# This path points to where your metadata.csv and audio files are located
dataset_path = os.path.join(os.getcwd(), "audio_data/dataset")
output_path = os.path.join(os.getcwd(), "models/voice_model")

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Reads a CSV file and auto-detects if it uses | or , as a separator.
    Handles path joining for metadata and audio files.
    """
    items = []
    
    # FIX: Join the root_path and manifest_file to get the full path
    manifest_path = os.path.join(root_path, manifest_file)
    print(f" -> Reading metadata from: {manifest_path}")
    
    if not os.path.exists(manifest_path):
        print(f" -> Error: File not found at {manifest_path}")
        return []
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        print(" -> Error: Metadata file is empty!")
        return []

    # Check the first valid line to guess the delimiter
    first_line = lines[0].strip()
    delimiter = "|"
    if "|" not in first_line and "," in first_line:
        delimiter = ","
        print(f" -> Auto-detected delimiter: COMMA")
    else:
        print(f" -> Auto-detected delimiter: PIPE")

    # Process lines
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        
        cols = line.split(delimiter)
        
        if len(cols) >= 2:
            wav_filename = cols[0].strip()
            text = delimiter.join(cols[1:]).strip()
            
            # FIX: Robust audio path checking
            # 1. Try direct path
            wav_path = os.path.join(root_path, wav_filename)
            
            # 2. If not found, try looking in a 'wavs' subdirectory (common format)
            if not os.path.exists(wav_path):
                wav_path_subdir = os.path.join(root_path, "wavs", wav_filename)
                if os.path.exists(wav_path_subdir):
                    wav_path = wav_path_subdir
            
            # Only add if we actually found the audio file
            if os.path.exists(wav_path):
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "my_voice",
                    "root_path": root_path
                })
            else:
                 # Warn only for the first few missing files to avoid spamming logs
                 if i < 3:
                     print(f" -> Warning: Audio file not found for {wav_filename}")
                     print(f"    (Looked at: {wav_path})")

    print(f" -> Successfully loaded {len(items)} items.")
    return items

def train_model():
    print(f"Initializing training using dataset at: {dataset_path}")
    
    # 1. Define Dataset Configuration
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. Configure VITS Model
    config = VitsConfig(
        batch_size=8,
        epochs=50,
        print_step=5,
        eval_split_size=0.1,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        test_sentences=[
            "Hello, this is my digital twin speaking.",
            "I can generate new audio from text now."
        ]
    )

    # 3. Initialize Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # 4. Load Data Samples
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_size,
        eval_split_size=config.eval_split_size,
        formatter=custom_formatter 
    )

    # 5. Initialize Model
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

    # 7. Start Training
    print("Starting training... (This may take a while)")
    trainer.fit()

if __name__ == "__main__":
    train_model()