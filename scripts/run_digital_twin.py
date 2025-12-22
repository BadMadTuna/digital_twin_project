import os
import sys
import torch
import subprocess
import time
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.audio import AudioProcessor
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================

# --- PATHS (Dynamic based on your 'ls' output) ---
PROJECT_ROOT = os.getcwd() # Assumes you run this from ~/digital_twin_project
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 1. XTTS SETTINGS
# Automatically find the latest training run if you don't want to hardcode
TRAIN_RUN_NAME = "xtts_finetuned-December-11-2025_02+59PM-bae2302" # Update if you prefer a different one
CHECKPOINT_NAME = "checkpoint_1500.pth" 

MODEL_DIR = os.path.join(MODELS_DIR, TRAIN_RUN_NAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Reference Audio (The one used for cloning)
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset/wavs/segment_0330.wav") 
OUTPUT_AUDIO_PATH = os.path.join(PROJECT_ROOT, "temp_speech.wav")
LANGUAGE = "en"

# 2. VIDEO SETTINGS (LivePortrait in venv_video)
SOURCE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets/avatar.jpg") # ⚠️ Make sure you put a jpg here!
OUTPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "final_digital_twin.mp4")

# Pointing to the VENV you already created
PYTHON_VIDEO_EXEC = os.path.join(PROJECT_ROOT, "venv_video", "bin", "python")
LIVEPORTRAIT_DIR = os.path.join(PROJECT_ROOT, "LivePortrait")
LIVEPORTRAIT_SCRIPT = os.path.join(LIVEPORTRAIT_DIR, "inference.py") 
# Note: Standard inference.py needs a driving video. 
# If using an audio-fork, change this to "inference_audio.py"

# =========================================================================
# 🔊 XTTS ENGINE
# =========================================================================

def load_xtts_model():
    print(f"⏳ Loading XTTS Model from {CHECKPOINT_NAME}...")
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    
    # Clean config quirks
    if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
    if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')

    model = Xtts.init_from_config(config)
    
    # Tokenizer Setup
    base_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    vocab_file = os.path.join(base_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        # Fallback
        import TTS.utils.manage as manage
        mgr = manage.ModelManager()
        path = mgr.download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
        vocab_file = os.path.join(os.path.dirname(path), "vocab.json")
    
    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

    # Load & Remap Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint)
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in model_keys: 
            new_state_dict[k] = v
        elif "gpt_inference" in k:
            new_k = k.replace("gpt_inference.", "")
            if new_k in model_keys: 
                new_state_dict[new_k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.cuda()
    model.eval()
    
    model.ap = AudioProcessor(sample_rate=22050, num_mels=80, do_trim_silence=True, n_fft=1024, win_length=1024, hop_length=256)
    return model

def generate_audio(model, text):
    print(f"🎤 Generating Voice: '{text}'")
    t0 = time.time()
    
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[REF_AUDIO_PATH],
        gpt_cond_len=30
    )
    
    out = model.inference(
        text,
        LANGUAGE,
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.8
    )
    
    model.ap.save_wav(out["wav"], OUTPUT_AUDIO_PATH)
    print(f"✅ Audio saved ({time.time()-t0:.2f}s) -> {OUTPUT_AUDIO_PATH}")
    return OUTPUT_AUDIO_PATH

# =========================================================================
# 🎥 VIDEO ENGINE (Calling venv_video)
# =========================================================================

def generate_video(audio_path):
    print("🎬 Starting LivePortrait...")
    t0 = time.time()
    
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"❌ Avatar image missing at: {SOURCE_IMAGE_PATH}")
        print("   Please upload a photo of yourself to 'assets/avatar.jpg' first.")
        return

    # Check if we have the video environment python
    if not os.path.exists(PYTHON_VIDEO_EXEC):
        print(f"❌ Video VEnv Python not found: {PYTHON_VIDEO_EXEC}")
        return

    # COMMAND CONSTRUCTION
    # This assumes you have an inference script that accepts --driving audio
    # If using standard LivePortrait, you might need to adapt this to use a bridge script
    cmd = [
        PYTHON_VIDEO_EXEC, 
        LIVEPORTRAIT_SCRIPT,
        "--source", SOURCE_IMAGE_PATH,
        "--driving", audio_path, 
        "--output", OUTPUT_VIDEO_PATH
    ]
    
    try:
        # We run this in the LivePortrait directory so it finds its weights/configs
        subprocess.run(cmd, check=True, cwd=LIVEPORTRAIT_DIR)
        print(f"✅ Video Ready ({time.time()-t0:.2f}s) -> {OUTPUT_VIDEO_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Video Generation Failed. Check if '{LIVEPORTRAIT_SCRIPT}' supports audio input.")
        print(f"   Error: {e}")

# =========================================================================
# 🚀 MAIN LOOP
# =========================================================================
def main():
    # 1. Load Audio Model (Once)
    model = load_xtts_model()
    
    print("\n✨ Digital Twin Interface ✨")
    print(f"   Audio Engine: {TRAIN_RUN_NAME} | Checkpoint: {CHECKPOINT_NAME}")
    print(f"   Video Engine: LivePortrait (via venv_video)")
    
    while True:
        text = input("\n📝 Enter text (or 'exit'): ")
        if text.lower() in ["exit", "quit"]: break
        if not text.strip(): continue
            
        # 1. Speak
        audio_file = generate_audio(model, text)
        
        # 2. Animate
        generate_video(audio_file)

if __name__ == "__main__":
    main()