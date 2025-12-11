import os
import sys 
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor

# --- CONFIGURATION ---
# Target the directory from your last successful run
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_01+15PM-3e81100"
# Use the checkpoint you verified exists
MODEL_CHECKPOINT_NAME = "checkpoint_3070.pth" 
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, MODEL_CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

REFERENCE_WAV_PATH = "audio_data/dataset/wavs/segment_0330.wav"
OUTPUT_WAV_PATH = "output_fine_tuned_test.wav"
TARGET_TEXT = "Hello Gemini, this is the voice I just trained. I hope it sounds clear and natural."
LANGUAGE = "en"

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
model = Xtts.init_from_config(config)

# 2. Manual Checkpoint Loading and Filtering
print(f"Loading and filtering weights from {MODEL_CHECKPOINT_PATH}...")
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)

# The generic Trainer wraps the model state in the 'model' key
state_dict = checkpoint.get("model", checkpoint)

# Filter out all 'dvae.' and 'gpt.gpt_inference' keys which are not in the instantiated Xtts object
# We keep only the core GPT, ConditioningEncoder, and HifiGAN Decoder weights.
filtered_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("dvae.") or k.startswith("gpt.gpt_inference."):
        # Ignore these keys—the DVAE is loaded from dvae.pth, and gpt_inference is not part of the core Xtts structure.
        continue
    # Rename keys to fit the instantiated Xtts model (e.g., gpt.mel_embedding -> gpt.mel_embedding)
    # The keys look correct, so we mostly keep them, but we ensure the core GPT weights are correctly mapped.
    
    # Simple direct load, relying on strict=False to ignore the keys we don't care about (DVAE is the problem)
    # Since our training script used the resurrected DVAE and separate patching, we load the full checkpoint weights
    # into the model's submodules.
    
    filtered_state_dict[k] = v

# Load the weights into the model, ignoring keys that do not match the instantiated architecture.
# We set strict=False to ignore the unexpected DVAE and GPT_Inference keys.
model.load_state_dict(filtered_state_dict, strict=False)

print("✅ Model weights loaded successfully.")
model.to(device)
model.eval()

# 3. Initialize Audio Processor
ap = AudioProcessor(**config.audio)
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
        # Get the 512-dim embedding
        speaker_embedding = model.gpt.get_conditioning(mels)
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