import os
import csv
import json
import random
import torch
import types
import sys
import requests
import numpy as np
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
# 🛠️ CHECKPOINT FREQUENCY: Save every 500 steps
SAVE_STEP = 500 

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
        sys.exit(f"❌ Failed to download dvae.pth: {e}")
             
    return dvae_path

# -------------------------------------------------------------------------
# 🛠️ RESURRECTION UTILITY
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
    
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if ".conv." in k and ("decoder" in k or "encoder" in k):
            new_k = k.replace(".conv.", ".")
            new_checkpoint[new_k] = v
        else:
            new_checkpoint[k] = v
            
    model_state = dvae.state_dict()
    filtered_checkpoint = {}
    for k, v in new_checkpoint.items():
        if k in model_state:
            if v.shape != model_state[k].shape:
                pass 
            else:
                filtered_checkpoint[k] = v
        else:
             filtered_checkpoint[k] = v

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
    
    # 🛠️ CRITICAL FIX: Set save_step here on the config object
    config.save_step = SAVE_STEP

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

    # 🛠️ CONFIG STABILIZATION PATCH
    config.use_speaker_embedding = config.model_args.use_speaker_embedding
    config.use_language_embedding = config.model_args.use_language_embedding
    
    # 🛠️ RESURRECT DVAE
    resurrect_dvae(model, CHECKPOINT_DIR)

    # -------------------------------------------------------------------------
    # 🛠️ PRE-COMPUTE SPEAKER LATENT (The "Constant Latent" Strategy)
    # -------------------------------------------------------------------------
    print("⏳ Pre-computing speaker latent from reference audio...")
    ref_audio_path = None
    for f in os.listdir(WAVS_DIR):
        if f.endswith(".wav"):
            ref_audio_path = os.path.join(WAVS_DIR, f)
            break
    
    if not ref_audio_path:
        raise FileNotFoundError("No .wav files found in WAVS_DIR to compute speaker latent.")

    print(f"   Using reference: {ref_audio_path}")
    
    wav = model.ap.load_wav(ref_audio_path)
    wav_tensor = torch.FloatTensor(wav).unsqueeze(0).unsqueeze(0)
    
    # 1. Compute Mels
    wav_numpy = wav_tensor.squeeze().cpu().numpy() 
    mels_numpy = model.ap.melspectrogram(wav_numpy)
    mels = torch.from_numpy(mels_numpy).unsqueeze(0)
    
    if torch.cuda.is_available():
        mels = mels.cuda()
        
    # 2. Get the 512-dimensional conditioned latent
    with torch.no_grad():
        feature_map = model.gpt.get_conditioning(mels) 
        
        # Global Average Pooling: [B, C, T] -> [B, C]
        speaker_latent = feature_map.mean(dim=2, keepdim=False)
        
        # Ensure B=1 dimension is removed if B=1
        if speaker_latent.dim() == 2 and speaker_latent.shape[0] == 1:
            speaker_latent = speaker_latent.squeeze(0)
        
        print(f"✅ Speaker Latent Computed: {speaker_latent.shape}")
        
    # Store the 512-dimensional latent
    model.fixed_speaker_latent = speaker_latent

    # -------------------------------------------------------------------------
    # 🛠️ PATCH 7 & 8: TRAINING AND EVALUATION STEPS
    # -------------------------------------------------------------------------
    def patched_train_step(self, batch, criterion=None):
        text_inputs = batch.get("text_input")
        text_lengths = batch.get("text_lengths")
        mel_inputs = batch.get("mel_input")
        mel_lengths = batch.get("mel_lengths")

        # 1. TRANSPOSE: [B, T, C] -> [B, C, T]
        mel_inputs_transposed = mel_inputs.transpose(1, 2)

        # 2. Compute Codes
        with torch.no_grad():
            audio_codes = self.dvae.get_codebook_indices(mel_inputs_transposed)

        # 3. Use Pre-computed Speaker Latent
        if self.fixed_speaker_latent.dim() == 1:
            latent_2d = self.fixed_speaker_latent.unsqueeze(0)
        else:
            latent_2d = self.fixed_speaker_latent
            
        batch_size = text_inputs.shape[0]
        cond_latents_3d = latent_2d.unsqueeze(1).expand(batch_size, -1, -1)

        # 4. Train GPT (Final Call)
        outputs = self.gpt(
            text_inputs=text_inputs,
            text_lengths=text_lengths,
            audio_codes=audio_codes,
            wav_lengths=mel_lengths,
            cond_mels=cond_latents_3d,   
            cond_latents=cond_latents_3d 
        )
        
        # 5. Extract and return loss
        loss_text, loss_mel, mel_logits = outputs
        total_loss = loss_text + loss_mel
        
        return outputs, {"loss": total_loss, "loss_text": loss_text, "loss_mel": loss_mel}

    def patched_eval_step(self, batch, criterion=None):
        with torch.no_grad():
            return self.train_step(batch, criterion)

    model.train_step = types.MethodType(patched_train_step, model)
    model.eval_step = types.MethodType(patched_eval_step, model)
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