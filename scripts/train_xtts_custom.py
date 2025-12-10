import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import Xtts
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.datasets import load_tts_samples
from TTS.tts.datasets.dataset import TTSDataset

# === CONFIGURATION ===
PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset")
WAVS_FOLDER = os.path.join(DATASET_PATH, "wavs")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/xtts_finetuned")
METADATA_FILE = "metadata.csv"
LANGUAGE = "en"

BATCH_SIZE = 4        # Lowered slightly to be safe
GRAD_ACCM_STEPS = 2   # Simulate batch size 8
EPOCHS = 15
LEARNING_RATE = 5e-6  # Standard fine-tuning rate
LOG_STEPS = 50
SAVE_STEPS = 250
# =====================

os.environ["COQUI_TOS_AGREED"] = "1"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def custom_formatter(root_path, manifest_file, **kwargs):
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2: continue
            wav_file = parts[0].strip()
            text = "|".join(parts[1:]).strip()
            wav_path = os.path.join(root_path, wav_file)
            if not os.path.exists(wav_path): wav_path = os.path.join(root_path, "wavs", wav_file)
            if os.path.exists(wav_path):
                items.append({"text": text, "audio_file": wav_path, "speaker_name": "my_voice", "language": LANGUAGE, "root_path": root_path})
    return items

def main():
    # 1. SETUP CONFIG & PATHS
    print("⚙️  Initializing Configuration...")
    checkpoint_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    config = load_config(os.path.join(checkpoint_dir, "config.json"))
    
    # 2. LOAD MODEL
    print("⬇️  Loading XTTS Model...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_path=os.path.join(checkpoint_dir, "model.pth"), vocab_path=os.path.join(checkpoint_dir, "vocab.json"), eval=True, strict=False)
    
    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 3. MANUAL COMPONENT CHECK (Debug)
    print("\n🔍 Checking Model Components:")
    # Find the submodules we need for training
    dvae = getattr(model, "dvae", None) or getattr(model, "hifigan_decoder", None)
    cond_model = getattr(model, "cond_stage_model", None)
    
    if dvae: print("   ✅ DVAE/HifiGan found")
    else: print("   ❌ DVAE NOT FOUND (Critical)")
    
    if cond_model: print("   ✅ Conditioning Model found")
    else: print("   ❌ Conditioning Model NOT FOUND (Critical)")

    # 4. PREPARE DATASET
    print("\n📂 Loading Dataset...")
    dataset_config = BaseDatasetConfig(formatter="custom_formatter", meta_file_train=METADATA_FILE, path=DATASET_PATH, language=LANGUAGE)
    train_samples, _ = load_tts_samples(dataset_config, eval_split=True, formatter=custom_formatter)
    
    # Monkey patch dataset to force audio paths
    orig_load = TTSDataset.load_data
    TTSDataset.load_data = lambda self, idx: {**orig_load(self, idx), "audio_file": self.samples[idx]["audio_file"]}
    
    # Init Audio Processor
    ap = AudioProcessor(sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, num_mels=80)
    
    # Create Collate Function (The "Glue" that prepares the batch)
    def collate_fn(batch):
        # 1. Text
        text_inputs = [torch.tensor(model.tokenizer.encode(x["text_input"], lang=LANGUAGE)) for x in batch]
        text_lengths = torch.tensor([len(t) for t in text_inputs])
        text_inputs = torch.nn.utils.rnn.pad_sequence(text_inputs, batch_first=True, padding_value=0)
        
        # 2. Audio (Robust Loading)
        mels, codes, wav_lens = [], [], []
        
        for x in batch:
            # Load raw audio
            wav = ap.load_wav(x["audio_file"])
            wav = torch.tensor(wav).unsqueeze(0) # [1, T]
            wav = wav.to(device)
            
            # Compute Features on the fly
            with torch.no_grad():
                # Conditioning Mel
                mask = torch.ones(1, 1, device=device)
                cond = cond_model(wav.unsqueeze(0), mask=mask).squeeze(0) # [Embd]
                
                # Audio Codes
                code = dvae.get_codebook_indices(wav.unsqueeze(0)).squeeze(0) # [T_code]
            
            mels.append(cond)
            codes.append(code)
            wav_lens.append(code.shape[-1])

        # Stack/Pad
        mels = torch.stack(mels) # [B, Embd]
        codes = torch.nn.utils.rnn.pad_sequence(codes, batch_first=True, padding_value=0)
        wav_lens = torch.tensor(wav_lens)
        
        return {
            "text_inputs": text_inputs,
            "text_lengths": text_lengths,
            "audio_codes": codes,
            "wav_lengths": wav_lens,
            "cond_mels": mels
        }

    dataset = TTSDataset(outputs=train_samples, compute_linear_spec=False, compute_mel_spec=False, tokenizer=model.tokenizer, ap=ap)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)

    # 5. TRAINING LOOP
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"\n🚀 Starting Custom Training Loop ({EPOCHS} Epochs)")
    
    step = 0
    model.train()
    
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        for i, batch in enumerate(loader):
            # Move batch to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            outputs = model.gpt(
                text_inputs=batch["text_inputs"],
                text_lengths=batch["text_lengths"],
                audio_codes=batch["audio_codes"],
                wav_lengths=batch["wav_lengths"],
                cond_mels=batch["cond_mels"],
                return_attentions=False
            )
            
            # Loss & Backward
            loss = outputs["loss"]
            loss.backward()
            
            # Optimizer Step
            if (i + 1) % GRAD_ACCM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
            
            # Logging
            if step % LOG_STEPS == 0 and (i + 1) % GRAD_ACCM_STEPS == 0:
                print(f"   Step {step} | Loss: {loss.item():.4f}")
                
            # Saving
            if step % SAVE_STEPS == 0 and step > 0:
                save_path = os.path.join(OUTPUT_PATH, f"checkpoint_{step}.pth")
                print(f"   💾 Saving checkpoint to {save_path}")
                # Save just the GPT weights (that's all we train)
                torch.save(model.gpt.state_dict(), save_path)

    print("\n✅ Training Complete!")
    final_path = os.path.join(OUTPUT_PATH, "final_gpt_model.pth")
    torch.save(model.gpt.state_dict(), final_path)
    print(f"   Saved final model to {final_path}")

if __name__ == "__main__":
    main()