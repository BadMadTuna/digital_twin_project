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
# DATA FORMATTING
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
# 🔍 SEARCH UTILITY: FIND VQGAN
# -------------------------------------------------------------------------
def find_vqgan_in_model(model):
    """Recursively searches for the VQGAN/DVAE module."""
    print("🔍 Searching for VQGAN/DVAE in model...")
    
    # 1. Check known locations
    if hasattr(model, "hifigan_decoder") and hasattr(model.hifigan_decoder, "vqgan"):
        return model.hifigan_decoder.vqgan
    if hasattr(model, "dvae"):
        return model.dvae
        
    # 2. Recursive Search (Depth-First)
    for name, module in model.named_modules():
        # Check for common names
        if name.endswith("vqgan") or name.endswith("dvae"):
            print(f"✅ Found VQGAN at: {name}")
            return module
        # Check for signature method 'encode' and 'codebook' property
        if hasattr(module, "encode") and hasattr(module, "codebook"):
            print(f"✅ Found VQGAN (by signature) at: {name}")
            return module

    # 3. Last Resort: Check if hifigan_decoder IS the vqgan (unlikely but possible)
    if hasattr(model, "hifigan_decoder") and hasattr(model.hifigan_decoder, "encode"):
         return model.hifigan_decoder

    print("❌ VQGAN NOT FOUND. Training will likely fail.")
    return None

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

    # --- COMPATIBILITY PATCHES ---
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

    # 🛠️ FIND AND BIND VQGAN
    # We find the missing component and attach it to 'model.found_vqgan'
    model.found_vqgan = find_vqgan_in_model(model)

    # -------------------------------------------------------------------------
    # 🛠️ PATCH 7: CUSTOM GPT TRAINING STEP
    # -------------------------------------------------------------------------
    def patched_train_step(self, batch, criterion=None):
        # 1. Prepare Data Keys
        text_inputs = batch.get("text_input")
        text_lengths = batch.get("text_lengths")
        mel_inputs = batch.get("mel_input")
        mel_lengths = batch.get("mel_lengths")

        # 2. Get Audio Codes (Using our discovered VQGAN)
        if self.found_vqgan is None:
             raise RuntimeError("Cannot train: VQGAN/DVAE not found in model.")
             
        with torch.no_grad():
            # Run encoder
            # Note: The signature of encode() might vary. Usually returns (z, loss, info).
            # info[2] is the indices (codes).
            ret = self.found_vqgan.encode(mel_inputs)
            
            # Robust extraction of codes:
            if isinstance(ret, tuple) and len(ret) == 3:
                # Standard Coqui VQGAN returns (z, quant_loss, (perplexity, min_encodings, INDICES))
                audio_codes = ret[2][2] 
            elif isinstance(ret, tuple) and len(ret) > 0:
                 # Fallback guess: typically the last item or the indices tensor
                 audio_codes = ret[-1]
            else:
                 # If it just returns codes
                 audio_codes = ret
                 
        # 3. Get Conditioning Latents
        with torch.no_grad():
            if hasattr(self, "speaker_encoder"):
                cond_latents = self.speaker_encoder(mel_inputs)
            else:
                # Try hifigan decoder as fallback
                cond_latents = self.hifigan_decoder(mel_inputs, return_latents=True)

        # 4. Train GPT
        outputs = self.gpt(
            text_inputs=text_inputs,
            text_lengths=text_lengths,
            audio_codes=audio_codes,
            audio_lengths=mel_lengths,
            cond_latents=cond_latents
        )
        
        return outputs, outputs

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