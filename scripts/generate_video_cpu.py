import os
import shutil
import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer

# --- CONFIGURATION ---
# UPDATE THIS to your best run folder
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/voice_model_large/run-December-08-2025_02+11PM-de2b618"

# Try to find the specific best checkpoint if possible, otherwise standard best
LIVE_CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth") 
LIVE_CONFIG     = os.path.join(MODEL_DIR, "config.json")

# Tuning
LENGTH_SCALE = 0.95 
NOISE_SCALE = 0.8 
NOISE_SCALE_W = 0.9

# Paths
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
os.makedirs(output_folder, exist_ok=True)

def generate_digital_twin(text_prompt):
    print(f"--- 1. Safe-Loading Model ---")
    
    # Copy model to avoid lock conflicts
    temp_model_path = os.path.join(output_folder, "temp_inference_model.pth")
    # If the file is missing (training deleted it), warn the user
    if not os.path.exists(LIVE_CHECKPOINT):
        print(f"❌ Error: {LIVE_CHECKPOINT} not found.")
        print("   Did the trainer overwrite it? Check your folder for 'best_model_XXXX.pth'")
        return
        
    shutil.copy(LIVE_CHECKPOINT, temp_model_path)

    print("-> Loading Synthesizer on CPU...")
    synthesizer = Synthesizer(
        tts_checkpoint=temp_model_path,
        tts_config_path=LIVE_CONFIG,
        use_cuda=False 
    )

    print(f"--- 2. Generating Audio ---")
    output_audio_path = os.path.join(output_folder, "generated_speech_test.wav")
    
    # Generate raw wav (Returns a list of floats)
    wav = synthesizer.tts(
        text_prompt,
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W
    )
    
    # --- ✂️ THE SAFE FIX ---
    # Convert list to numpy for slicing
    # We trim based on sample rate (e.g., 22050 samples = 1 second)
    sr = synthesizer.output_sample_rate
    trim_seconds = 0.15
    trim_samples = int(sr * trim_seconds)
    
    # Slice the list/array
    if len(wav) > trim_samples:
        wav = wav[:-trim_samples]
    
    # Use Coqui's internal saver (Handles headers/speed correctly)
    synthesizer.save_wav(wav, output_audio_path)
    print(f"✅ Audio saved (Correctly Encoded): {output_audio_path}")
    
    # Cleanup
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

if __name__ == "__main__":
    text = input("\n🗣️  Enter text to test: ")
    generate_digital_twin(text)