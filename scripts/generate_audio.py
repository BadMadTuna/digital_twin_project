import os
import sys 
import torch
import numpy as np
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager 
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# --------------------------------------------------------------------------
# --- CONFIGURATION ---
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_01+15PM-3e81100"
# Use the checkpoint you verified exists
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
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Determine Base Model Path and Load Config
manager = ModelManager()
model_path_tuple = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
BASE_MODEL_DIR = os.path.dirname(model_path_tuple[0])

# Robustly find the vocab file
vocab_file = None
for root, dirs, files in os.walk(BASE_MODEL_DIR):
    if "vocab.json" in files:
        vocab_file = os.path.join(root, "vocab.json")
        break
    if "tokenizer.json" in files:
        vocab_file = os.path.join(root, "tokenizer.json")
        break

if not vocab_file:
    print("Error: Could not find vocab.json")
    sys.exit(1)

print("Loading model architecture...")
config = XttsConfig()
config.load_json(CONFIG_PATH)

# Clean conflicting millisecond attributes
if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')

model = Xtts.init_from_config(config)

# 🛠️ TOKENIZER FIX: Manually instantiate with only the vocab_file argument
print("Manually loading VoiceBpeTokenizer...")
model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

# 2. Load Weights
print(f"Loading weights from {MODEL_CHECKPOINT_PATH}...")
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)
model.load_state_dict(state_dict, strict=False)
print("✅ Model weights loaded successfully.")

model.to(device)
model.eval()

# 3. Initialize Audio Processor
print("Initializing AudioProcessor...")
SR = config.audio.sample_rate
NFFT = getattr(config.audio, 'n_fft', getattr(config.audio, 'fft_size', 1024))
WL = getattr(config.audio, 'win_length', NFFT)
HL = getattr(config.audio, 'hop_length', getattr(config.audio, 'frame_shift', 256))
NUM_MELS = getattr(config.audio, 'num_mels', 80)

# Calculate the millisecond values to inject
frame_length_ms = WL * 1000 / SR
frame_shift_ms = HL * 1000 / SR

# Prepare the dictionary
audio_config_dict = {k: v for k, v in config.audio.to_dict().items() if v is not None}
audio_config_dict["num_mels"] = NUM_MELS
audio_config_dict["frame_length_ms"] = frame_length_ms
audio_config_dict["frame_shift_ms"] = frame_shift_ms

ap = AudioProcessor(**audio_config_dict)
ap.sample_rate = config.audio.sample_rate

# 4. Generate Speaker Latent
print("Generating speaker latent...")
try:
    reference_wav = ap.load_wav(REFERENCE_WAV_PATH, sr=ap.sample_rate)
    wav_tensor = torch.from_numpy(reference_wav).unsqueeze(0).unsqueeze(0).to(device) 
    
    wav_numpy = wav_tensor.squeeze().cpu().numpy() 
    mels_numpy = ap.melspectrogram(wav_numpy)
    mels = torch.from_numpy(mels_numpy).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get raw features [1, 1024, T]
        speaker_embedding_features = model.gpt.get_conditioning(mels)
        
        # Pool to [1024] vector
        # mean over time (dim 2) -> [1, 1024], squeeze -> [1024]
        latents = speaker_embedding_features.mean(dim=2, keepdim=False).squeeze(0)
    
    # 🛠️ DIMENSION FIXES:
    # 1. GPT Cond Latent needs to be 3D: [Batch, 1, Dim] -> [1, 1, 1024]
    gpt_cond_latent = latents.unsqueeze(0).unsqueeze(0).to(device)
    
    # 2. Speaker Embedding (for Vocoder) needs to be 2D: [Batch, Dim] -> [1, 1024]
    speaker_embedding = latents.unsqueeze(0).to(device)

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

    if not wav_chunks:
        print("Error: No audio chunks generated.")
    else:
        final_wav = np.concatenate(wav_chunks)
        ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)
        print(f"\n✅ Synthesis Complete! Audio saved to: {OUTPUT_WAV_PATH}")