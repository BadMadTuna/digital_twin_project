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

# 1. Determine Base Model Path (for Tokenizer)
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

# Manually instantiate tokenizer
print(f"Loading Tokenizer from {vocab_file}...")
model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

# 2. Load Weights
print(f"Loading weights from {MODEL_CHECKPOINT_PATH}...")
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=device)
state_dict = checkpoint.get("model", checkpoint)

# Filter keys just in case, though usually Xtts.init handles this if structure matches
model.load_state_dict(state_dict, strict=False)
print("✅ Model weights loaded successfully.")

model.to(device)
model.eval()

# 3. Initialize Audio Processor (HARDCODED TO MATCH TRAINING)
print("Initializing AudioProcessor (Hardcoded to match training)...")
# These are the exact parameters used in 'train_xtts_real.py'
ap = AudioProcessor(
    sample_rate=22050, 
    num_mels=80, 
    do_trim_silence=True, 
    n_fft=1024, 
    win_length=1024, 
    hop_length=256
)

# 4. Generate Speaker Latent
print("Generating speaker latent...")
try:
    reference_wav = ap.load_wav(REFERENCE_WAV_PATH, sr=ap.sample_rate)
    wav_tensor = torch.from_numpy(reference_wav).unsqueeze(0).unsqueeze(0).to(device) 
    
    # Debug: Check audio stats
    print(f"   Audio Tensor Stats: Min={wav_tensor.min():.4f}, Max={wav_tensor.max():.4f}, Mean={wav_tensor.mean():.4f}")

    # Compute Mels using the FIXED AudioProcessor
    wav_numpy = wav_tensor.squeeze().cpu().numpy() 
    mels_numpy = ap.melspectrogram(wav_numpy)
    mels = torch.from_numpy(mels_numpy).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Get raw features [1, 1024, T]
        latents_raw = model.gpt.get_conditioning(mels) 
        
        # Pool over time dimension (dim=2) to get a global vector [1, 1024, 1]
        latents_pooled = latents_raw.mean(dim=2, keepdim=True)
        
    # Prepare inputs
    gpt_cond_latent = latents_pooled.transpose(1, 2) # [1, 1, 1024]
    speaker_embedding = latents_pooled[:, :512, :]   # [1, 512, 1]

    # Debug: Check latent stats
    print(f"   Latent Stats: Mean={gpt_cond_latent.mean():.4f}, Std={gpt_cond_latent.std():.4f}")
    if torch.isnan(gpt_cond_latent).any():
        print("❌ CRITICAL: Latent contains NaNs!")
        sys.exit(1)

    gpt_cond_latent = gpt_cond_latent.to(device)
    speaker_embedding = speaker_embedding.to(device)

except Exception as e:
    print(f"Fatal Error during latent generation: {e}")
    sys.exit(1)

# 5. Debug Tokenization
print(f"Tokenizing text: '{TARGET_TEXT}'")
tokens = model.tokenizer.encode(TARGET_TEXT, lang=LANGUAGE)
print(f"   Tokens: {tokens}")
print(f"   Token Count: {len(tokens)}")

# 6. Synthesize Speech
print("Synthesizing speech...")
with torch.no_grad():
    chunks = model.inference_stream(
        TARGET_TEXT,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=True,
        # Generation Parameters to force output
        temperature=0.7, 
        length_penalty=1.0,
        repetition_penalty=10.0, # High penalty to prevent loops, encouraging new tokens
        top_k=50,
        top_p=0.85,
    )

    wav_chunks = []
    for i, chunk in enumerate(chunks):
        if chunk is not None:
            # print(f"   Received chunk {i}, shape: {chunk.shape}") # Uncomment to see chunk flow
            wav_chunks.append(chunk.cpu().numpy())

    if not wav_chunks:
        print("Error: No audio chunks generated (Silent output).")
    else:
        final_wav = np.concatenate(wav_chunks)
        ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)
        print(f"\n✅ Synthesis Complete! Audio saved to: {OUTPUT_WAV_PATH}")
        print(f"   Filesize: {os.path.getsize(OUTPUT_WAV_PATH)} bytes")