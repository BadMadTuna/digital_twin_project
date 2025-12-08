import os
import shutil
import torch
import numpy as np
from scipy.io.wavfile import write
from TTS.utils.synthesizer import Synthesizer

# --- CONFIGURATION ---
# Point to your active fresh run
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/voice_model_fresh/run-December-08-2025_09+48AM-83efe94"
LIVE_CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth")
LIVE_CONFIG     = os.path.join(MODEL_DIR, "config.json")

# Tuning
LENGTH_SCALE = 1.1 
NOISE_SCALE = 0.667 
NOISE_SCALE_W = 0.8

# Paths
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
os.makedirs(output_folder, exist_ok=True)

def trim_end_artifact(audio_array, sample_rate, trim_duration_sec=0.15):
    """
    Cuts the last X seconds to remove the TTS 'end-of-sequence' breath/click.
    """
    # Calculate how many samples to cut
    samples_to_cut = int(sample_rate * trim_duration_sec)
    
    if len(audio_array) > samples_to_cut:
        return audio_array[:-samples_to_cut]
    return audio_array

def generate_digital_twin(text_prompt):
    print(f"--- 1. Safe-Loading Model ---")
    
    # Copy model to avoid lock conflicts
    temp_model_path = os.path.join(output_folder, "temp_inference_model.pth")
    shutil.copy(LIVE_CHECKPOINT, temp_model_path)

    print("-> Loading Synthesizer on CPU...")
    synthesizer = Synthesizer(
        tts_checkpoint=temp_model_path,
        tts_config_path=LIVE_CONFIG,
        use_cuda=False 
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
    
    # --- ✂️ THE FIX: TRIM THE ARTIFACT ---
    # Convert to numpy if needed (TTS usually returns list or numpy)
    wav_np = np.array(wav)
    
    # Trim the last 0.15 seconds (adjust if the 'ugh' is longer)
    wav_clean = trim_end_artifact(wav_np, synthesizer.output_sample_rate, trim_duration_sec=0.15)
    
    # Save cleaned audio
    synthesizer.save_wav(wav_clean, output_audio_path)
    print(f"✅ Audio saved (Trimmed): {output_audio_path}")
    
    # Cleanup
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)

if __name__ == "__main__":
    text = input("\n🗣️  Enter text to test: ")
    generate_digital_twin(text)