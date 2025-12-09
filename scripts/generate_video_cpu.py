import os
import shutil
import torch
from TTS.utils.synthesizer import Synthesizer

# --- CONFIGURATION ---
# Point to your active LARGE dataset run
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/voice_model/run-December-09-2025_03+49PM-8d1e3b8"

# Checkpoint paths
#LIVE_CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth") 
LIVE_CHECKPOINT = os.path.join(MODEL_DIR, "checkpoint_10000.pth") 
LIVE_CONFIG     = os.path.join(MODEL_DIR, "config.json")

# --- STANDARD VITS DEFAULTS ---
LENGTH_SCALE = 1.0   # Speed (1.0 = Normal)
NOISE_SCALE = 0.667  # Randomness/Emotion (Standard)
NOISE_SCALE_W = 0.8  # Pronunciation Variance (Standard)

# Paths
project_root = os.getcwd()
output_folder = os.path.join(project_root, "outputs")
os.makedirs(output_folder, exist_ok=True)

def generate_digital_twin(text_prompt):
    print(f"--- 1. Safe-Loading Model ---")
    
    # Copy model to avoid lock conflicts with the running trainer
    temp_model_path = os.path.join(output_folder, "temp_inference_model.pth")
    
    if not os.path.exists(LIVE_CHECKPOINT):
        print(f"❌ Error: {LIVE_CHECKPOINT} not found.")
        print("   Did the trainer overwrite it? Check for 'best_model_XXXX.pth'")
        return
        
    shutil.copy(LIVE_CHECKPOINT, temp_model_path)

    print("-> Loading Synthesizer on CPU...")
    synthesizer = Synthesizer(
        tts_checkpoint=temp_model_path,
        tts_config_path=LIVE_CONFIG,
        use_cuda=False  # Protects your GPU memory
    )

    print(f"--- 2. Generating Audio ---")
    output_audio_path = os.path.join(output_folder, "generated_speech_test.wav")
    
    # Generate raw wav
    wav = synthesizer.tts(
        text_prompt,
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W
    )
    
    # Save directly (No trimming)
    synthesizer.save_wav(wav, output_audio_path)
    print(f"✅ Audio saved: {output_audio_path}")
    print(f"   (Settings: Speed={LENGTH_SCALE}, Noise={NOISE_SCALE}, NoiseW={NOISE_SCALE_W})")
    
    # Cleanup temp file
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

if __name__ == "__main__":
    text = input("\n🗣️  Enter text to test: ")
    generate_digital_twin(text)