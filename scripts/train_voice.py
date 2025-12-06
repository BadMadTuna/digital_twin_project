import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# --- Configuration ---
dataset_path = os.path.join(os.getcwd(), "audio_data/dataset")
metadata_file = os.path.join(dataset_path, "metadata.csv")
output_path = os.path.join(os.getcwd(), "models/voice_model")

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Reads a CSV file with format: filename.wav|transcription
    """
    items = []
    with open(manifest_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            # We expect at least 2 columns: filename and text
            if len(cols) >= 2:
                wav_filename = cols[0].strip()
                text = cols[1].strip()
                
                # Construct full path to audio file
                wav_path = os.path.join(root_path, wav_filename)
                
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "my_voice",
                    "root_path": root_path
                })
    return items

def train_model():
    print(f"Initializing training using dataset at: {dataset_path}")
    
    # 1. Define Dataset Configuration
    # NOTE: We set formatter to "ljspeech" as a placeholder string.
    # We will pass the actual function explicitly to load_tts_samples below.
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
    # We pass 'formatter=custom_formatter' here to override the string in the config
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