import os
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor

# --- CONFIGURATION ---
# 1. Update this path to point to your final checkpoint directory
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_01+15PM-3e81100"
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_3070.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
REFERENCE_WAV_PATH = "audio_data/dataset/wavs/segment_0330.wav" # Use the same reference audio
OUTPUT_WAV_PATH = "output_test_voice.wav"
TARGET_TEXT = "Hello Gemini, this is the voice I just trained. I hope it sounds clear and natural."
LANGUAGE = "en"

# --- EXECUTION ---
if not os.path.exists(MODEL_CHECKPOINT_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model Config and Model
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, MODEL_CHECKPOINT_PATH, eval=True, use_deepspeed=False)
model.to(device)

# 2. Initialize Audio Processor
ap = AudioProcessor(**config.audio)
ap.sample_rate = config.audio.sample_rate

# 3. Generate Speaker Latent from Reference Audio
print("Generating speaker latent...")
try:
    # Load raw audio
    reference_wav = ap.load_wav(REFERENCE_WAV_PATH, sr=ap.sample_rate)
    
    # Speaker Encoder expects [Batch, 1, Time]
    wav_tensor = torch.from_numpy(reference_wav).unsqueeze(0).unsqueeze(0).to(device) 
    
    # This call now handles the entire conversion chain (Raw Audio -> Mel -> Latent)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=None, # Path is used internally for saving, but we pass tensor
        audio=wav_tensor,
        gpt_cond_len=model.config.model_args.gpt_cond_len, # Usually 30
        max_autocast_cache=model.config.model_args.max_autocast_cache,
    )
except Exception as e:
    print(f"Error during latent generation (This is common if API changed): {e}")
    # FALLBACK: Use a more direct method if get_conditioning_latents fails
    # This uses the same logic we debugged, but adapted for inference
    
    # 1. Compute Mels
    wav_numpy = wav_tensor.squeeze().cpu().numpy() 
    mels_numpy = ap.melspectrogram(wav_numpy)
    mels = torch.from_numpy(mels_numpy).unsqueeze(0).to(device)
    
    # 2. Get the 512-dimensional conditioned latent
    with torch.no_grad():
        # Get 512-dim embedding from Mels
        speaker_embedding = model.gpt.get_conditioning(mels)
        # Average over time axis (dim 2) and remove singleton batch dim (dim 1)
        gpt_cond_latent = speaker_embedding.mean(dim=2, keepdim=False).squeeze(0)
    
    speaker_embedding = speaker_embedding.squeeze(0) # Prepare embedding for generation

# 4. Synthesize Speech
print(f"Synthesizing speech for: '{TARGET_TEXT}'")
chunks = model.inference_stream(
    TARGET_TEXT,
    LANGUAGE,
    gpt_cond_latent,
    speaker_embedding,
    enable_text_splitting=True,
)

# 5. Concatenate and Save
wav_chunks = []
for chunk in chunks:
    wav_chunks.append(chunk.cpu().numpy())

final_wav = np.concatenate(wav_chunks)
ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)

print(f"\n✅ Synthesis Complete! Audio saved to: {OUTPUT_WAV_PATH}")
print("You can listen to the file to hear your fine-tuned voice.")