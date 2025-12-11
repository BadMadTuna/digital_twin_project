import os
import csv
import json
import random
import torch
import types
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets.formatters import *

# -------------------------------------------------------------------------
# CONFIGURATION & PATHS
# -------------------------------------------------------------------------
RUN_NAME = "xtts_finetuned"
OUT_PATH = os.path.join(os.getcwd(), "models")
METADATA_CSV = "metadata.csv"
WAVS_DIR = "audio_data/dataset/wavs" 
LANGUAGE = "en"
SPEAKER_NAME = "my_speaker"

# Training Settings
BATCH_SIZE = 2 
EPOCHS = 10
LEARNING_RATE = 5e-6

# -------------------------------------------------------------------------
# HELPER 1: CONVERT CSV TO XTTS JSON
# -------------------------------------------------------------------------
def format_dataset(csv_file, train_json, eval_json):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"❌ Could not find {csv_file}")

    print("Converting metadata.csv to XTTS JSON format...")
    
    items = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) < 2: continue
            audio_name = row[0].strip()
            text = row[1].strip()
            audio_path = os.path.join(WAVS_DIR, audio_name)
            
            items.append({
                "text": text,
                "audio_file": audio_path,
                "audio_unique_name": audio_name, 
                "speaker_name": SPEAKER_NAME,
                "language": LANGUAGE
            })

    random.shuffle(items)
    split_idx = int(len(items) * 0.9)
    train_items = items[:split_idx]
    eval_items = items[split_idx:]

    with open(train_json, "w", encoding="utf-8") as f:
        for item in train_items: f.write(json.dumps(item) + "\n")
    with open(eval_json, "w", encoding="utf-8") as f:
        for item in eval_items: f.write(json.dumps(item) + "\n")

    print(f"✅ Created {len(train_items)} training samples.")

# -------------------------------------------------------------------------
# HELPER 2: LOAD JSON DATA
# -------------------------------------------------------------------------
def load_json_data(json_file):
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f: data.append(json.loads(line))
    return data

# -------------------------------------------------------------------------
# MAIN TRAINING ROUTINE
# -------------------------------------------------------------------------
def main():
    # 1. Prepare Data
    train_json, eval_json = "metadata_train.json", "metadata_eval.json"
    format_dataset(METADATA_CSV, train_json, eval_json)

    # 2. Locate Model
    print("⏳ Verifying model path...")
    manager = ModelManager()
    model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    model_path = model_path_tuple[0]
    CHECKPOINT_DIR = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    print(f"✅ Found model at: {CHECKPOINT_DIR}")

    # 3. Configure
    config = XttsConfig()
    config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))
    config.datasets = [
        BaseDatasetConfig(
            formatter="xtts",
            meta_file_train=train_json,
            meta_file_val=eval_json,
            path=os.getcwd(),
            language=LANGUAGE
        )
    ]
    config.batch_size = BATCH_SIZE
    config.epochs = EPOCHS
    config.lr = LEARNING_RATE
    config.output_path = OUT_PATH
    config.run_name = RUN_NAME

    # 4. Load Model
    print("⬇️  Loading XTTS v2 Base Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
    if torch.cuda.is_available(): model.cuda()

    # =========================================================================
    # 🛠️ COMPATIBILITY PATCHES
    # =========================================================================
    
    # 1. Basic Managers
    model.get_criterion = lambda: None
    if model.speaker_manager:
        model.speaker_manager.save_ids_to_file = lambda x: None
        if hasattr(type(model.speaker_manager), "name_to_id"): delattr(type(model.speaker_manager), "name_to_id")
        model.speaker_manager.name_to_id = {SPEAKER_NAME: 0}
    if model.language_manager: model.language_manager.save_ids_to_file = lambda x: None
    
    # 2. Tokenizer
    if model.tokenizer:
        model.tokenizer.use_phonemes = False
        model.tokenizer.print_logs = lambda *args, **kwargs: None
        model.tokenizer.text_to_ids = lambda t: model.tokenizer.encode(t, lang=LANGUAGE)

    # 3. Config
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False
    config.r = 1

    # 4. Audio Processor (Robust FFT settings)
    if model.ap is None:
        model.ap = AudioProcessor(
            sample_rate=22050, num_mels=80, do_trim_silence=True, 
            n_fft=1024, win_length=1024, hop_length=256
        )

    # =========================================================================
    # 🛠️ PATCH 7: CUSTOM GPT TRAINING STEP
    # The standard train_step is broken/missing. We implement the logic manually.
    # =========================================================================
    def patched_train_step(self, batch, criterion=None):
        # 1. Unpack Batch
        text_inputs = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_inputs = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]

        # 2. Compute Audio Codes (Target for GPT)
        # We use the internal VQGAN to turn Mel Spectrograms into discrete codes.
        # This is what the GPT learns to predict.
        with torch.no_grad():
            # encode returns: (quantized, loss, info). info[2] is indices.
            _, _, info = self.hifigan_decoder.vqgan.encode(mel_inputs)
            audio_codes = info[2] # [B, T]

        # 3. Compute Conditioning Latents (Speaker Style)
        # We need to tell the GPT "who" is speaking.
        # We use the speaker_encoder (if present) or hifigan_decoder to get this.
        with torch.no_grad():
            if hasattr(self, "speaker_encoder"):
                # Standard XTTS v2 speaker encoder
                cond_latents = self.speaker_encoder(mel_inputs)
            else:
                # Fallback: Try to use hifigan_decoder if speaker_encoder is merged
                # (This path depends on specific model version, trying common fallback)
                # If this fails, we might need a dummy, but let's try this first.
                cond_latents = self.hifigan_decoder(mel_inputs, return_latents=True)

        # 4. Run GPT Training Forward
        # We pass the codes and latents to the GPT. It returns the loss dictionary.
        # Note: 'forward' on the GPT submodule usually calculates loss if 'audio_codes' is provided.
        outputs = self.gpt(
            text_inputs=text_inputs,
            text_lengths=text_lengths,
            audio_codes=audio_codes,
            audio_lengths=mel_lengths,
            cond_latents=cond_latents
        )
        
        return outputs, outputs

    # Bind the new method to the instance
    model.train_step = types.MethodType(patched_train_step, model)
    # =========================================================================

    # 5. Load Data & Start
    print("⏳ Loading data samples...")
    train_samples = load_json_data(train_json)
    eval_samples = load_json_data(eval_json)

    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False),
        config, output_path=OUT_PATH, model=model, train_samples=train_samples, eval_samples=eval_samples,   
    )

    print("🚀 Starting Training...")
    trainer.fit()

if __name__ == "__main__":
    main()