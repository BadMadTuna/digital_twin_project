import os
import sys 
import shutil
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager 
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# --------------------------------------------------------------------------
# --- CONFIGURATION (EXPLICIT) ---
# 1. Point to your specific training directory
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_02+59PM-bae2302"

# 2. Select the specific checkpoint you want to test
# Options from your list: checkpoint_500.pth, checkpoint_1000.pth, checkpoint_1500.pth
CHECKPOINT_NAME = "checkpoint_1500.pth"

MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

REFERENCE_WAV_PATH = "audio_data/dataset/wavs/segment_0330.wav"
OUTPUT_WAV_PATH = "output_manual_test_1500.wav"
TARGET_TEXT = "Hello Gemini, I am testing checkpoint 1500 to see how the training is progressing."
LANGUAGE = "en"

# --------------------------------------------------------------------------
# --- HELPER: SMART CHECKPOINT LOADER ---
def load_custom_checkpoint(model, checkpoint_path, device):
    print(f"🛠️  Loading and remapping weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"⚠️  Failed to load checkpoint: {e}")
        return False
    
    # Handle 'model' key nesting
    state_dict = checkpoint.get("model", checkpoint)
    
    # Get the model's actual keys
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    matched = 0
    
    for k, v in state_dict.items():
        # Direct match
        if k in model_keys:
            new_state_dict[k] = v
            matched += 1
            continue
        
        # Remap 'gpt_inference' -> 'gpt'
        if "gpt_inference" in k:
            new_k = k.replace("gpt_inference.", "")
            if new_k in model_keys:
                new_state_dict[new_k] = v
                matched += 1
                continue
    
    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weights loaded successfully ({matched} keys matched).")
    return True

# --------------------------------------------------------------------------
# --- MAIN EXECUTION ---
def main():
    # 1. Validate Paths
    if not os.path.exists(MODEL_CHECKPOINT_PATH):
        print(f"❌ Error: Checkpoint not found at {MODEL_CHECKPOINT_PATH}")
        sys.exit(1)
        
    print(f"🔍 Target Checkpoint: {CHECKPOINT_NAME}")

    # 🛡️ SAFETY COPY: Copy checkpoint to temp to avoid read/write conflicts
    temp_ckpt_path = "temp_inference_model.pth"
    print("🛡️  Creating temporary snapshot...")
    shutil.copyfile(MODEL_CHECKPOINT_PATH, temp_ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Init Model Structure (using base XTTS assets)
    manager = ModelManager()
    model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    base_model_dir = os.path.dirname(model_path_tuple[0])
    
    # Robustly find vocab file
    vocab_file = None
    for root, dirs, files in os.walk(base_model_dir):
        if "vocab.json" in files: vocab_file = os.path.join(root, "vocab.json"); break
        if "tokenizer.json" in files: vocab_file = os.path.join(root, "tokenizer.json"); break

    if not vocab_file: print("❌ Error: Could not find vocab.json"); sys.exit(1)

    print("Loading architecture...")
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    
    # Cleaning config to prevent NoneType errors
    if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
    if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')

    model = Xtts.init_from_config(config)
    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

    # 3. Load Weights from Snapshot
    success = load_custom_checkpoint(model, temp_ckpt_path, device)
    if not success:
        os.remove(temp_ckpt_path)
        sys.exit(1)

    model.to(device)
    model.eval()

    # 4. Setup Audio Processor (Hardcoded to match training)
    print("Initializing Audio Processor...")
    ap = AudioProcessor(
        sample_rate=22050, num_mels=80, do_trim_silence=True, 
        n_fft=1024, win_length=1024, hop_length=256
    )
    model.ap = ap

    # 5. Extract Speaker Latent (Manual Method)
    print("Generating speaker latent...")
    try:
        reference_wav = ap.load_wav(REFERENCE_WAV_PATH, sr=ap.sample_rate)
        wav_tensor = torch.from_numpy(reference_wav).unsqueeze(0).unsqueeze(0).to(device)
        wav_numpy = wav_tensor.squeeze().cpu().numpy()
        mels = torch.from_numpy(ap.melspectrogram(wav_numpy)).unsqueeze(0).to(device)

        with torch.no_grad():
            latents_raw = model.gpt.get_conditioning(mels)
            latents_pooled = latents_raw.mean(dim=2, keepdim=True)
            
        gpt_cond_latent = latents_pooled.transpose(1, 2)
        speaker_embedding = latents_pooled[:, :512, :]
        
    except Exception as e:
        print(f"❌ Latent generation failed: {e}")
        sys.exit(1)

    # 6. Inference
    print("🗣️  Synthesizing...")
    try:
        with torch.no_grad():
            chunks = model.inference_stream(
                TARGET_TEXT, LANGUAGE, gpt_cond_latent, speaker_embedding,
                enable_text_splitting=True, temperature=0.7, top_k=50, top_p=0.8
            )
            
            wav_chunks = []
            for chunk in chunks:
                if chunk is not None:
                    wav_chunks.append(chunk.cpu().numpy())

            if wav_chunks:
                final_wav = np.concatenate(wav_chunks)
                ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)
                print(f"✅ Audio saved: {OUTPUT_WAV_PATH}")
                print(f"   Size: {os.path.getsize(OUTPUT_WAV_PATH)} bytes")
            else:
                print("⚠️  Warning: Generated audio was empty.")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        
    finally:
        # Cleanup
        print("🧹 Cleaning up...")
        if os.path.exists(temp_ckpt_path):
            os.remove(temp_ckpt_path)
        
        # Clear VRAM so training isn't impacted
        del model
        del chunks
        torch.cuda.empty_cache()
        print("✨ GPU memory cleared.")

if __name__ == "__main__":
    main()