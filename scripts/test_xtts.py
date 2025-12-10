import os
import random
import torch
from TTS.api import TTS

# === CONFIGURATION ===
DATASET_DIR = "audio_data/dataset/wavs"
OUTPUT_FILE = "xtts_test_result.wav"
TEXT_TO_SPEAK = "The ocean is a place of wonder, where giant creatures roam the deep in absolute silence."
# =====================

def run_test():
    # 1. Agree to Coqui License (Required for XTTS)
    os.environ["COQUI_TOS_AGREED"] = "1"

    # 2. Pick a random reference file from your dataset
    wavs = [f for f in os.listdir(DATASET_DIR) if f.endswith(".wav")]
    if not wavs:
        print("Error: No wavs found.")
        return
    
    # Pick a file between 5-10 seconds for best cloning
    ref_wav_name = random.choice(wavs)
    ref_wav_path = os.path.join(DATASET_DIR, ref_wav_name)
    
    print(f"🎤 Reference Audio: {ref_wav_name}")
    print("⏳ Loading XTTS v2 (this downloads ~2GB on first run)...")

    # 3. Load Model
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Generate
    print("🗣️  Generating speech...")
    tts.tts_to_file(
        text=TEXT_TO_SPEAK,
        file_path=OUTPUT_FILE,
        speaker_wav=ref_wav_path,
        language="en"
    )

    print(f"✅ Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_test()