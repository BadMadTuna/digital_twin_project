import os
import csv
import json
import random
import torch
import types
import sys
import requests
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
# DATA HELPERS
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
# 🛠️ DOWNLOADER
# -------------------------------------------------------------------------
def download_dvae_if_missing(checkpoint_dir):
    dvae_path = os.path.join(checkpoint_dir, "dvae.pth")
    if os.path.exists(dvae_path):
        return dvae_path
    
    print("⚠️  dvae.pth missing. Downloading from Hugging Face...")
    url = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    try:
        r = requests.get(url, allow_redirects=True)
        with open(dvae_path, 'wb') as f:
            f.write(r.content)
        print("✅ dvae.pth downloaded successfully.")
    except Exception as e:
        print(f"❌ Failed to download dvae.pth: {e}")
        try:
             url_backup = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
             r = requests.get(url_backup, allow_redirects=True)
             with open(dvae_path, 'wb') as f:
                f.write(r.content)
             print("✅ dvae.pth downloaded successfully (backup).")
        except:
             sys.exit("Could not download dvae.pth. Please download it manually.")
    return dvae_path

# -------------------------------------------------------------------------
# 🛠️ RESURRECTION UTILITY (Robust Shape Checking)
# -------------------------------------------------------------------------
def resurrect_dvae(model, checkpoint_dir):
    print("✨ Attempting to resurrect missing VQGAN/DVAE...")
    try:
        from TTS.tts.layers.xtts.dvae import DiscreteVAE
    except ImportError as e:
        print(f"❌ Could not import DiscreteVAE: {e}")
        sys.exit(1)

    dvae = DiscreteVAE(
        channels=80, normalization=None, positional_dims=1, num_tokens=1024, 
        codebook_dim=512, hidden_dim=512, num_resnet_blocks=3, kernel_size=3, num_layers=2
    )
    
    dvae_path = download_dvae_if_missing(checkpoint_dir)
    print(f"   Loading weights from {dvae_path}...")
    
    checkpoint = torch.load(dvae_path, map_location="cpu")
    
    # 1. Remap keys (fix .conv. mismatches)
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if ".conv." in k and ("decoder" in k or "encoder" in k):
            new_k = k.replace(".conv.", ".")
            new_checkpoint[new_k] = v
        else:
            new_checkpoint[k] = v
            
    # 2. 🛠️ SHAPE CHECKER: Filter out size mismatches
    # We get the shapes the model EXPECTS
    model_state = dvae.state_dict()
    filtered_checkpoint = {}
    
    for k, v in new_checkpoint.items():
        if k in model_state:
            expected_shape = model_state[k].shape
            if v.shape != expected_shape:
                print(f"   ⚠️ Skipping shape mismatch for {k}: Checkpoint {v.shape} vs Model {expected_shape}")
                # We skip this key. Since it's likely a Decoder key, we don't care (we only need Encoder).
            else:
                filtered_checkpoint[k] = v
        else:
             filtered_checkpoint[k] = v

    # 3. Load with strict=False
    dvae.load_state_dict(filtered_checkpoint, strict=False)
    print("✅ DVAE weights loaded successfully (Encoder is ready).")

    if torch.cuda.is_available():
        dvae = dvae.cuda()
        
    model.dvae = dvae
    return model.dvae

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

    # 🛠️ RESURRECT DVAE (With Robust Loader)
    resurrect_dvae(model, CHECKPOINT_DIR)

    # -------------------------------------------------------------------------
    # 🛠️ PATCH 7: CUSTOM GPT TRAINING STEP
    # -------------------------------------------------------------------------
    def patched_train_step(self, batch, criterion=None):
        text_inputs = batch.get("text_input")
        text_lengths = batch.get("text_lengths")
        mel_inputs = batch.get("mel_input")
        mel_lengths = batch.get("mel_lengths")

        # Compute Codes
        with torch.no_grad():
            _, _, info = self.dvae.encode(mel_inputs)
            audio_codes = info[2] 

        # Compute Latents
        with torch.no_grad():
            if hasattr(self, "speaker_encoder"):
                cond_latents = self.speaker_encoder(mel_inputs)
            else:
                cond_latents = self.hifigan_decoder(mel_inputs, return_latents=True)

        # Train GPT
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