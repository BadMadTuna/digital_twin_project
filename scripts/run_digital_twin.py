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

PROJECT_ROOT = os.getcwd() 

# 1. XTTS SETTINGS (Your Voice)
TRAIN_RUN_NAME = "xtts_finetuned-December-11-2025_02+59PM-bae2302"
CHECKPOINT_NAME = "checkpoint_5000.pth"  # <--- UPDATED to 5000

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", TRAIN_RUN_NAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

# Audio Config
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset/wavs/segment_0330.wav") 
OUTPUT_AUDIO_PATH = os.path.join(PROJECT_ROOT, "temp_speech.wav")
LANGUAGE = "en"

# 2. VIDEO SETTINGS (Ditto Engine)
# Make sure your avatar image is here
SOURCE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets/avatar.jpg")
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "outputs")
if not os.path.exists(OUTPUT_VIDEO_DIR): os.makedirs(OUTPUT_VIDEO_DIR)

# Ditto Environment & Script Paths
DITTO_DIR = os.path.join(PROJECT_ROOT, "Ditto")
PYTHON_DITTO_EXEC = os.path.join(PROJECT_ROOT, "venv_ditto", "bin", "python")
DITTO_SCRIPT = os.path.join(DITTO_DIR, "inference.py")

# =========================================================================
# 🔊 XTTS ENGINE
# =========================================================================

def load_xtts_model():
    print(f"⏳ Loading XTTS Model from {CHECKPOINT_NAME}...")
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    
    # Cleanup legacy config attributes if present
    if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
    if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')

    model = Xtts.init_from_config(config)
    
    # Robust Tokenizer Loading
    base_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    vocab_file = os.path.join(base_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        import TTS.utils.manage as manage
        mgr = manage.ModelManager()
        path = mgr.download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
        vocab_file = os.path.join(os.path.dirname(path), "vocab.json")
    
    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

    # Load Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    state_dict = checkpoint.get("model", checkpoint)
    model_keys = set(model.state_dict().keys())
    new_state_dict = {}
    
    for k, v in state_dict.items():
        if k in model_keys: new_state_dict[k] = v
        elif "gpt_inference" in k:
            new_k = k.replace("gpt_inference.", "")
            if new_k in model_keys: new_state_dict[new_k] = v
            
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
# 🎥 VIDEO ENGINE (Ditto)
# =========================================================================

def generate_video(audio_path):
    print("🎬 Starting Ditto (Motion-Space Diffusion)...")
    t0 = time.time()
    
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"❌ Avatar image missing at: {SOURCE_IMAGE_PATH}")
        print("   Please upload a photo of yourself to 'assets/avatar.jpg'")
        return

    # Standard Ditto Inference Call
    # Note: If your Ditto version uses a config file (e.g. --config path/to/config), add it here.
    cmd = [
        PYTHON_DITTO_EXEC, 
        DITTO_SCRIPT,
        "--source_image", SOURCE_IMAGE_PATH,
        "--audio_path", audio_path,
        "--output_path", os.path.join(OUTPUT_VIDEO_DIR, "ditto_output.mp4"),
        "--device", "cuda"
    ]
    
    try:
        # We run inside the Ditto directory so it finds relative paths (checkpoints/configs)
        subprocess.run(cmd, check=True, cwd=DITTO_DIR)
        print(f"✅ Video Finished ({time.time()-t0:.2f}s)")
        print(f"   Saved to: {os.path.join(OUTPUT_VIDEO_DIR, 'ditto_output.mp4')}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Video Generation Failed.")
        print(f"   Command used: {' '.join(cmd)}")
        print(f"   Error code: {e}")

# =========================================================================
# 🚀 MAIN LOOP
# =========================================================================
def main():
    # Ensure we are running from project root
    if os.path.basename(os.getcwd()) == "scripts":
        os.chdir("..")

    model = load_xtts_model()
    
    print("\n✨ Digital Twin Interface (Ditto Edition) ✨")
    print(f"   Audio: {TRAIN_RUN_NAME} (Steps: 5000)")
    print(f"   Video: AntGroup/Ditto")
    
    while True:
        text = input("\n📝 Enter text (or 'exit'): ")
        if text.lower() in ["exit", "quit"]: break
        if not text.strip(): continue
            
        audio_file = generate_audio(model, text)
        generate_video(audio_file)

if __name__ == "__main__":
    main()