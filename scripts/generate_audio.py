import os
import sys 
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor

# --------------------------------------------------------------------------
# --- CONFIGURATION ---
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_01+15PM-3e81100"
MODEL_CHECKPOINT_NAME = "checkpoint_3070.pth" 
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, MODEL_CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

REFERENCE_WAV_PATH = "audio_data/dataset/wavs/segment_0330.wav"
OUTPUT_WAV_PATH = "output_fine_tuned_test.wav"
TARGET_TEXT = "Hello Gemini, this is the voice I just trained. I hope it sounds clear and natural."
LANGUAGE = "en"

# --------------------------------------------------------------------------
# --- EXECUTION ---
if not os.path.exists(MODEL_CHECKPOINT_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
    print("Please manually verify the filename and path.")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model Config and Architecture
print("Loading model architecture...")
config = XttsConfig()
config.load_json(CONFIG_PATH)

# --- FINAL FIX: Retrieve parameters robustly ---
SR = config.audio.sample_rate

# FIX 1: Safely access NFFT, falling back to fft_size if n_fft is missing.
NFFT = getattr(config.audio, 'n_fft', getattr(config.audio, 'fft_size', 1024))
WL = getattr(config.audio, 'win_length', NFFT)
# 🛠️ FINAL FIX: Safely access hop_length, falling back to frame_shift or 256.
HL = getattr(config.audio, 'hop_length', getattr(config.audio, 'frame_shift', 256))
NUM_MELS = config.audio.num_mels

# CRITICAL FIX: Delete conflicting millisecond attributes to prevent NoneType error
if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')


model = Xtts.init_from_config(config)

# 2. Load Weights (Using manual load logic)
print(f"Loading weights from {MODEL_CHECKPOINT_PATH}...")
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict, strict=False)
print("✅ Model weights loaded successfully.")

model.to(device)
model.eval()

# 3. Initialize Audio Processor (Stabilized Configuration)
print("Initializing AudioProcessor...")

# Calculate the millisecond values to inject as non-None floats
frame_length_ms = WL * 1000 / SR
frame_shift_ms = HL * 1000 / SR

# Prepare the dictionary, ensuring required values are non-null
audio_config_dict = {k: v for k, v in config.audio.to_dict().items() if v is not None}

# Inject the calculated, non-None millisecond values
audio_config_dict["frame_length_ms"] = frame_length_ms
audio_config_dict["frame_shift_ms"] = frame_shift_ms

ap = AudioProcessor(**audio_config_dict)
ap.sample_rate = config.audio.sample_rate

# 4. Generate Speaker Latent
print("Generating speaker latent...")
try:
    reference_wav = ap.load_wav(REFERENCE_WAV_PATH, sr=ap.sample_rate)
    wav_tensor = torch.from_numpy(reference_wav).unsqueeze(0).unsqueeze(0).to(device) 
    
    # Compute Mels
    wav_numpy = wav_tensor.squeeze().cpu().numpy() 
    mels_numpy = ap.melspectrogram(wav_numpy)
    mels = torch.from_numpy(mels_numpy).unsqueeze(0).to(device)
    
    with torch.no_grad():
        speaker_embedding = model.gpt.get_conditioning(mels)
        # Global Average Pooling Fix
        gpt_cond_latent = speaker_embedding.mean(dim=2, keepdim=False).squeeze(0)
    
    # Prepare tensors for generation
    gpt_cond_latent = gpt_cond_latent.unsqueeze(0)
    speaker_embedding = speaker_embedding.unsqueeze(0).to(device)

except Exception as e:
    print(f"Fatal Error during latent generation: {e}")
    sys.exit(1)

# 5. Synthesize Speech
print(f"Synthesizing speech for: '{TARGET_TEXT}'")
with torch.no_grad():
    chunks = model.inference_stream(
        TARGET_TEXT,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=True,
    )

# 6. Concatenate and Save
wav_chunks = []
for chunk in chunks:
    if chunk is not None:
        wav_chunks.append(chunk.cpu().numpy())

final_wav = np.concatenate(wav_chunks)
ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)

print(f"\n✅ Synthesis Complete! Audio saved to: {OUTPUT_WAV_PATH}")