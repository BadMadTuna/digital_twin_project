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

def train_model():
    print(f"Initializing training using dataset at: {dataset_path}")
    
    # 1. Define Dataset Configuration
    # We use the metadata.csv generated in the previous step
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", # The format output by our process_audio script (filename|text)
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # 2. Configure VITS Model (optimized for voice cloning)
    config = VitsConfig(
        batch_size=16, # Lower this if you get Out Of Memory (OOM) errors
        epochs=50,      # 50-100 is usually good for fine-tuning
        print_step=5,
        eval_split_size=0.1,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False
    )

    # 3. Initialize Audio Processor
    ap = AudioProcessor.init_from_config(config)

    # 4. Load Data Samples
    # The formatter handles parsing your metadata.csv
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_size,
        eval_split_size=config.eval_split_size,
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