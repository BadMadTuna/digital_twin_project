import os
import csv
import json
import random
import torch
import types
import inspect
import sys
from TTS.utils.manage import ModelManager
from TTS.utils.audio import AudioProcessor
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets.formatters import *

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
RUN_NAME = "xtts_finetuned"
OUT_PATH = os.path.join(os.getcwd(), "models")
METADATA_CSV = "metadata.csv"
WAVS_DIR = "audio_data/dataset/wavs" 
LANGUAGE = "en"
SPEAKER_NAME = "my_speaker"
BATCH_SIZE = 2 
EPOCHS = 10
LEARNING_RATE = 5e-6

# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def format_dataset(csv_file, train_json, eval_json):
    if not os.path.exists(csv_file): raise FileNotFoundError(f"❌ Could not find {csv_file}")
    print("Converting metadata.csv to XTTS JSON format...")
    items = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) < 2: continue
            audio_name, text = row[0].strip(), row[1].strip()
            items.append({
                "text": text,
                "audio_file": os.path.join(WAVS_DIR, audio_name),
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

def load_json_data(json_file):
    data = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f: data.append(json.loads(line))
    return data

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    train_json, eval_json = "metadata_train.json", "metadata_eval.json"
    format_dataset(METADATA_CSV, train_json, eval_json)

    manager = ModelManager()
    model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    model_path = model_path_tuple[0]
    CHECKPOINT_DIR = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
    
    config = XttsConfig()
    config.load_json(os.path.join(CHECKPOINT_DIR, "config.json"))
    config.datasets = [BaseDatasetConfig(formatter="xtts", meta_file_train=train_json, meta_file_val=eval_json, path=os.getcwd(), language=LANGUAGE)]
    config.batch_size = BATCH_SIZE
    config.epochs = EPOCHS
    config.lr = LEARNING_RATE
    config.output_path = OUT_PATH
    config.run_name = RUN_NAME

    print("⬇️  Loading XTTS v2 Base Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=CHECKPOINT_DIR, eval=True)
    if torch.cuda.is_available(): model.cuda()

    # --- PATCHES ---
    model.get_criterion = lambda: None
    if model.speaker_manager:
        model.speaker_manager.save_ids_to_file = lambda x: None
        if hasattr(type(model.speaker_manager), "name_to_id"): delattr(type(model.speaker_manager), "name_to_id")
        model.speaker_manager.name_to_id = {SPEAKER_NAME: 0}
    if model.language_manager: model.language_manager.save_ids_to_file = lambda x: None
    if model.tokenizer:
        model.tokenizer.use_phonemes = False
        model.tokenizer.print_logs = lambda *args, **kwargs: None
        model.tokenizer.text_to_ids = lambda t: model.tokenizer.encode(t, lang=LANGUAGE)
    config.model_args.use_speaker_embedding = True
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False
    config.r = 1
    if model.ap is None:
        model.ap = AudioProcessor(sample_rate=22050, num_mels=80, do_trim_silence=True, n_fft=1024, win_length=1024, hop_length=256)

    # -------------------------------------------------------------------------
    # 🔍 DIAGNOSTIC: FIND VQGAN / ENCODER
    # -------------------------------------------------------------------------
    def patched_train_step(self, batch, criterion=None):
        print("\n" + "="*50)
        print("🔍 INSPECTING HIFIGAN DECODER")
        print("="*50)
        
        # 1. Print attributes of hifigan_decoder
        if hasattr(self, "hifigan_decoder"):
            print("✅ self.hifigan_decoder EXISTS")
            print("   Attributes:")
            for attr in dir(self.hifigan_decoder):
                if not attr.startswith("__"):
                    print(f"   - {attr}")
        else:
            print("❌ self.hifigan_decoder MISSING")

        print("\n" + "="*50)
        print("🔍 INSPECTING GPT")
        print("="*50)
        
        # 2. Print attributes of gpt
        if hasattr(self, "gpt"):
            print("✅ self.gpt EXISTS")
            print("   Attributes:")
            for attr in dir(self.gpt):
                if not attr.startswith("__"):
                    print(f"   - {attr}")

        print("="*50 + "\n")
        sys.exit("⛔ Stopping for inspection.")

    model.train_step = types.MethodType(patched_train_step, model)
    # -------------------------------------------------------------------------

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