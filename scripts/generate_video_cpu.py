import os
import shutil
import torch
from TTS.utils.synthesizer import Synthesizer

# --- CONFIGURATION ---
# 1. Point to the NEW 'fresh' run
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/voice_model_fresh/run-December-08-2025_09+48AM-83efe94"
LIVE_CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth")
LIVE_CONFIG     = os.path.join(MODEL_DIR, "config.json")

# 2. Tuning Knobs
LENGTH_SCALE = 1.1 
NOISE_SCALE = 0.667 
NOISE_SCALE_W = 0.8

# 3. Output
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
os.makedirs(output_folder, exist_ok=True)

def generate_digital_twin(text_prompt):
    print(f"--- 1. Safe-Loading Model ---")
    
    # SAFETY STEP 1: Copy the model to a temp file
    # This prevents us from reading the file while the trainer tries to write to it.
    temp_model_path = os.path.join(output_folder, "temp_inference_model.pth")
    shutil.copy(LIVE_CHECKPOINT, temp_model_path)
    print(f"-> Copied checkpoint to temp file (Safe Mode)")

    # SAFETY STEP 2: Force CPU
    # We set use_cuda=False so we don't crash the training run with Out-Of-Memory errors.
    print("-> Loading Synthesizer on CPU (to protect training VRAM)...")
    synthesizer = Synthesizer(
        tts_checkpoint=temp_model_path,
        tts_config_path=LIVE_CONFIG,
        use_cuda=False  # <--- CRITICAL
    )

    print(f"--- 2. Generating Audio ---")
    output_audio_path = os.path.join(output_folder, "generated_speech_test.wav")
    
    wav = synthesizer.tts(
        text_prompt,
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W
    )
    
    synthesizer.save_wav(wav, output_audio_path)
    print(f"✅ Audio saved to: {output_audio_path}")
    
    # Cleanup temp file
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

if __name__ == "__main__":
    text = input("\n🗣️  Enter text to test the new British Brain: ")
    generate_digital_twin(text)