import os
import torch
import inspect
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.utils.audio import AudioProcessor

# === CONFIGURATION ===
PROJECT_ROOT = os.getcwd()
DATASET_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/xtts_finetuned")
METADATA_FILE = "metadata.csv"
LANGUAGE = "en"
# =====================

# Agree to license
os.environ["COQUI_TOS_AGREED"] = "1"

def custom_formatter(root_path, manifest_file, **kwargs):
    """
    Adapts your VITS metadata format (wav|text) to XTTS format (wav|text|speaker|lang)
    """
    items = []
    manifest_path = os.path.join(root_path, manifest_file)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2: continue
            
            wav_file = parts[0].strip()
            text = "|".join(parts[1:]).strip()
            
            # Find the full path
            wav_path = os.path.join(root_path, wav_file)
            if not os.path.exists(wav_path):
                 wav_path = os.path.join(root_path, "wavs", wav_file)

            if os.path.exists(wav_path):
                items.append({
                    "text": text,
                    "audio_file": wav_path,
                    "speaker_name": "my_voice",
                    "language": LANGUAGE,
                    "root_path": root_path
                })
    return items

def train_xtts():
    # 1. Dataset Config
    dataset_config = BaseDatasetConfig(
        formatter="custom_formatter",
        meta_file_train=METADATA_FILE,
        path=DATASET_PATH,
        language=LANGUAGE
    )

    # 2. DEFINE PATHS
    checkpoint_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    model_path = os.path.join(checkpoint_dir, "model.pth")
    config_path = os.path.join(checkpoint_dir, "config.json")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")
    speaker_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

    # 3. LOAD & UPDATE CONFIG
    print(f"⚙️  Loading Config from: {config_path}")
    config = load_config(config_path)

    # --- FIX: Inject attributes for Generic Trainer compatibility ---
    config.model_args.use_speaker_embedding = False
    config.model_args.use_d_vector_file = False
    config.model_args.use_language_embedding = False
    config.r = 1 # Legacy reduction factor (not used by XTTS, set to 1)
    
    # NEW: Force loader to return raw audio so we can compute codes
    config.return_wav = True  
    # ----------------------------------------------------------------

    # Update the loaded config with your training preferences
    config.batch_size = 8
    config.eval_batch_size = 2
    config.num_loader_workers = 2
    config.num_eval_loader_workers = 1
    config.run_eval = True
    config.test_delay_epochs = -1
    config.epochs = 15
    config.text_cleaner = "whitespace_cleaner"
    config.use_phonemes = False
    config.print_step = 50
    config.print_eval = True
    config.save_step = 500
    config.output_path = OUTPUT_PATH
    config.datasets = [dataset_config]

    # 4. Load Model
    print("⬇️  Loading XTTS v2 Base...")
    model = Xtts.init_from_config(config)
    
    print(f"📂 Loading Checkpoint from: {checkpoint_dir}")
    model.load_checkpoint(
        config, 
        checkpoint_path=model_path, 
        vocab_path=vocab_path, 
        speaker_file_path=speaker_path,
        eval=True,
        strict=False
    )

    # 5. Load Data
    print("📂 Loading Data...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_size=0.1,
        formatter=custom_formatter
    )

    # --- FINAL MONKEY PATCH BLOCK ---
    
    # A. Patch Model Methods (Trainer expects these)
    def get_criterion(): return None
    def get_auxiliary_losses(*args, **kwargs): return {}
    model.get_criterion = get_criterion
    model.get_auxiliary_losses = get_auxiliary_losses

    # B. Patch Managers (Trainer tries to save IDs)
    if hasattr(model, "speaker_manager") and model.speaker_manager:
        model.speaker_manager.save_ids_to_file = lambda path: None
    if hasattr(model, "language_manager") and model.language_manager:
        model.language_manager.save_ids_to_file = lambda path: None

    # C. Patch Tokenizer (Trainer expects specific methods)
    if hasattr(model, "tokenizer") and model.tokenizer:
        model.tokenizer.use_phonemes = False
        def tokenizer_print_logs(level=0): pass
        model.tokenizer.print_logs = tokenizer_print_logs
        
        # Generic loader calls text_to_ids, XTTS needs encode
        def text_to_ids(text):
            return model.tokenizer.encode(text, lang=LANGUAGE)
        model.tokenizer.text_to_ids = text_to_ids

    # D. Patch AudioProcessor (Loader needs this to read WAVs)
    model.ap = AudioProcessor(
        sample_rate=22050,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        num_mels=80,
        do_trim_silence=False,
        do_sound_norm=False
    )

    # 5. Patch train_step (FINAL ROBUST ADAPTER)
    def train_step_wrapper(batch, criterion=None):
        inputs = {}

        # 1. Map Text (Try 'token_ids' or 'text_input')
        if "text_input" in batch:
            inputs["token_ids"] = batch["text_input"]
        if "text_lengths" in batch:
            inputs["text_lengths"] = batch["text_lengths"]
            
        # 2. Handle Audio (Robust Fallback Logic)
        wav = None
        
        # A. Try fetching existing tensor from batch
        for k in ["audio", "waveform"]:
             if k in batch and batch[k] is not None:
                 wav = batch[k]
                 break
        
        # B. Fallback: Load from disk if tensor is missing/None
        # (This fixes the 'NoneType' error you are seeing)
        if wav is None and "audio_file" in batch:
             wavs = []
             for path in batch["audio_file"]:
                 # Load raw audio using the patched processor
                 w = model.ap.load_wav(path)
                 w = torch.tensor(w).float().to(model.device)
                 wavs.append(w)
             
             # Pad and Stack to create (Batch, 1, Time)
             max_len = max([w.shape[0] for w in wavs])
             wav = torch.zeros(len(wavs), 1, max_len, device=model.device)
             for i, w in enumerate(wavs):
                 wav[i, 0, :w.shape[0]] = w

        # 3. Compute Codes & Conditioning
        if wav is not None:
            # Ensure correct dimensions (Batch, 1, Time)
            if wav.dim() == 2: 
                wav = wav.unsqueeze(1)
            
            ref_wav = wav.to(model.device)
            
            with torch.no_grad():
                # A. Conditioning Latents
                mask = torch.ones(ref_wav.shape[0], 1, device=ref_wav.device)
                cond_mels = model.cond_stage_model(ref_wav, mask=mask)
                
                # B. Discrete Audio Codes
                # XTTS v2 uses the DVAE to get codebook indices
                audio_codes = model.dvae.get_codebook_indices(ref_wav)
                
            inputs["audio_codes"] = audio_codes
            inputs["cond_mels"] = cond_mels
            inputs["cond_refs"] = cond_mels

        # 4. Call Model
        return model(**inputs)
    
    model.train_step = train_step_wrapper
    # --------------------------------

    # 6. Trainer
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    print("🚀 Starting XTTS Fine-Tuning...")
    trainer.fit()

if __name__ == "__main__":
    train_xtts()