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
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/xtts_finetuned-December-11-2025_03+45PM-482c5df"
# Use the checkpoint you verified exists
MODEL_CHECKPOINT_NAME = "checkpoint_8000.pth" 
MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, MODEL_CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

REFERENCE_WAV_PATH = "audio_data/dataset/wavs/segment_0328.wav"
OUTPUT_WAV_PATH = "output_fine_tuned_test.wav"
TARGET_TEXT = "Have you heard about Deep Resolve, our lord and savior? It is an absolute game changer for magnetic resonance imaging."
LANGUAGE = "en"

# --------------------------------------------------------------------------
# --- HELPER: SMART CHECKPOINT LOADER ---
def load_custom_checkpoint(model, checkpoint_path, device):
    print(f"🛠️ Loading and remapping weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle 'model' key nesting
    state_dict = checkpoint.get("model", checkpoint)
    
    # Get the model's actual keys
    model_keys = set(model.state_dict().keys())
    
    new_state_dict = {}
    matched_keys = 0
    ignored_keys = 0
    
    for k, v in state_dict.items():
        # 1. Direct match
        if k in model_keys:
            new_state_dict[k] = v
            matched_keys += 1
            continue
            
        # 2. Fix 'gpt_inference' prefix mismatch
        # Checkpoint: gpt.gpt_inference.transformer...
        # Model:      gpt.transformer...
        if "gpt_inference" in k:
            new_k = k.replace("gpt_inference.", "")
            if new_k in model_keys:
                new_state_dict[new_k] = v
                matched_keys += 1
                continue
                
        # 3. Handle 'dvae' keys (Ignore them, inference doesn't use DVAE directly)
        if k.startswith("dvae"):
            ignored_keys += 1
            continue

    # Load the remapped dictionary
    model.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Loaded {matched_keys} keys. Ignored {ignored_keys} keys (DVAE/Other).")

# --------------------------------------------------------------------------
# --- EXECUTION ---
if not os.path.exists(MODEL_CHECKPOINT_PATH):
    print(f"Error: Model checkpoint not found at {MODEL_CHECKPOINT_PATH}")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Determine Base Model Path
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
print("Manually loading VoiceBpeTokenizer...")
model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

# 2. LOAD WEIGHTS WITH REMAPPING
load_custom_checkpoint(model, MODEL_CHECKPOINT_PATH, device)

model.to(device)
model.eval()

# 3. Initialize Audio Processor (Hardcoded to match training)
print("Initializing AudioProcessor (Hardcoded to match training)...")
ap = AudioProcessor(
    sample_rate=22050, 
    num_mels=80, 
    do_trim_silence=True, 
    n_fft=1024, 
    win_length=1024, 
    hop_length=256
)
# Assign AP to model so internal methods work
model.ap = ap

# 4. Generate Speaker Latent
print("Generating speaker latent...")
try:
    # Use the standard method which handles slicing/projection correctly
    # Removed the unsupported 'max_autocast_cache' argument
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[REFERENCE_WAV_PATH],
        gpt_cond_len=30
    )
    
except Exception as e:
    print(f"Error using standard latent method: {e}")
    sys.exit(1)

# Debug: Check tokenization
print(f"Tokenizing text: '{TARGET_TEXT}'")
tokens = model.tokenizer.encode(TARGET_TEXT, lang=LANGUAGE)
print(f"Tokens found: {len(tokens)} IDs")

# 5. Synthesize Speech
print("Synthesizing speech...")
with torch.no_grad():
    chunks = model.inference_stream(
        TARGET_TEXT,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=True,
        temperature=0.7, 
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.8,
    )

    wav_chunks = []
    for chunk in chunks:
        if chunk is not None:
            wav_chunks.append(chunk.cpu().numpy())

    if not wav_chunks:
        print("Error: No audio chunks generated (Silent output).")
    else:
        final_wav = np.concatenate(wav_chunks)
        ap.save_wav(final_wav, OUTPUT_WAV_PATH, sr=ap.sample_rate)
        print(f"\n✅ Synthesis Complete! Audio saved to: {OUTPUT_WAV_PATH}")
        print(f"   Filesize: {os.path.getsize(OUTPUT_WAV_PATH)} bytes")