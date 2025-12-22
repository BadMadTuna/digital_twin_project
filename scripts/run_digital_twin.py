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

# 1. XTTS SETTINGS
TRAIN_RUN_NAME = "xtts_finetuned-December-11-2025_02+59PM-bae2302"
CHECKPOINT_NAME = "checkpoint_5000.pth" 

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", TRAIN_RUN_NAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")

REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset/wavs/segment_0330.wav") 
OUTPUT_AUDIO_PATH = os.path.join(PROJECT_ROOT, "temp_speech.wav")
LANGUAGE = "en"

# 2. VIDEO SETTINGS (Specific to Hekenye/LivePortrait-AudioDriven)
SOURCE_IMAGE_PATH = os.path.join(PROJECT_ROOT, "assets/avatar.jpg")
OUTPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "outputs")
if not os.path.exists(OUTPUT_VIDEO_DIR): os.makedirs(OUTPUT_VIDEO_DIR)

PYTHON_VIDEO_EXEC = os.path.join(PROJECT_ROOT, "venv_video", "bin", "python")
LIVEPORTRAIT_DIR = os.path.join(PROJECT_ROOT, "LivePortrait")

# 🔍 AUTO-DETECT KEY FILES
# 1. Inference Script
INFERENCE_SCRIPT_NAME = "inference_with_audio.py"
LIVEPORTRAIT_SCRIPT = os.path.join(LIVEPORTRAIT_DIR, INFERENCE_SCRIPT_NAME)

# 2. Statistic File (Recursive Search)
STATISTIC_PATH = None
print("🔍 Searching for statistic.pt...")
for root, dirs, files in os.walk(LIVEPORTRAIT_DIR):
    if "statistic.pt" in files:
        STATISTIC_PATH = os.path.join(root, "statistic.pt")
        print(f"   Found: {STATISTIC_PATH}")
        break

# 3. Pretrained Model Path (Recursive Search for main checkpoint)
# Usually named something like 'landmark_model.pth' or just the folder. 
# Based on the user prompt, it expects a path. We'll default to the 'pretrained_weights' dir 
# or a specific file if we can guess it.
PRETRAINED_MODEL_PATH = os.path.join(LIVEPORTRAIT_DIR, "pretrained_weights")
# If there is a specific 'liveportrait.pth' or similar, we might need to point to it.
# For now, pointing to the directory is the safest bet for these scripts unless it asks for a specific .pth

# =========================================================================
# 🔊 XTTS ENGINE
# =========================================================================

def load_xtts_model():
    print(f"⏳ Loading XTTS Model from {CHECKPOINT_NAME}...")
    if not os.path.exists(CHECKPOINT_PATH):
        sys.exit(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")

    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    
    if hasattr(config.audio, 'frame_length_ms'): delattr(config.audio, 'frame_length_ms')
    if hasattr(config.audio, 'frame_shift_ms'): delattr(config.audio, 'frame_shift_ms')

    model = Xtts.init_from_config(config)
    
    base_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    vocab_file = os.path.join(base_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        import TTS.utils.manage as manage
        mgr = manage.ModelManager()
        path = mgr.download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
        vocab_file = os.path.join(os.path.dirname(path), "vocab.json")
    
    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)

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
# 🎥 VIDEO ENGINE (Specific for Hekenye Fork)
# =========================================================================

def generate_video(audio_path):
    print(f"🎬 Starting LivePortrait ({INFERENCE_SCRIPT_NAME})...")
    t0 = time.time()
    
    if not os.path.exists(SOURCE_IMAGE_PATH):
        print(f"❌ Avatar image missing at: {SOURCE_IMAGE_PATH}")
        return

    if not os.path.exists(LIVEPORTRAIT_SCRIPT):
        print(f"❌ Script not found: {LIVEPORTRAIT_SCRIPT}")
        print("   Please check the folder structure of LivePortrait.")
        return

    # Check for statistic file
    stat_arg = STATISTIC_PATH
    if not stat_arg:
        print("⚠️  Warning: 'statistic.pt' not found automatically.")
        # Fallback assumption if the user hasn't downloaded it yet
        # We might need to assume a default path or fail gracefully
        stat_arg = os.path.join(LIVEPORTRAIT_DIR, "processed_MEAD", "statistic.pt")
        print(f"   Using default/guessed path: {stat_arg}")

    # COMMAND CONSTRUCTION
    cmd = [
        PYTHON_VIDEO_EXEC, 
        LIVEPORTRAIT_SCRIPT,
        "-s", SOURCE_IMAGE_PATH,
        "-d", audio_path, 
        "-o", OUTPUT_VIDEO_DIR,
        "--statistic_path", stat_arg,
        "--pretrained_model_path", PRETRAINED_MODEL_PATH
    ]
    
    try:
        # Run inside LivePortrait dir
        subprocess.run(cmd, check=True, cwd=LIVEPORTRAIT_DIR)
        
        print(f"✅ Video Finished ({time.time()-t0:.2f}s)")
        print(f"   Check the '{OUTPUT_VIDEO_DIR}' folder.")
        
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
    
    print("\n✨ Digital Twin Interface ✨")
    print(f"   Audio Engine: {TRAIN_RUN_NAME}")
    print(f"   Video Engine: LivePortrait (Audio Fork)")
    
    while True:
        text = input("\n📝 Enter text (or 'exit'): ")
        if text.lower() in ["exit", "quit"]: break
        if not text.strip(): continue
            
        audio_file = generate_audio(model, text)
        generate_video(audio_file)

if __name__ == "__main__":
    main()