import os
import glob
import torch
from TTS.utils.synthesizer import Synthesizer

# ==========================================
# 🎛️ INFERENCE TUNING KNOBS (Fix the Robot)
# ==========================================
# Speed: 1.0 is normal. 1.1 is slightly slower (often clearer).
LENGTH_SCALE = 1.2  

# Emotion/Randomness: 0.667 is standard. 
# If too flat/robotic, try 0.8. If too static/buzzy, try 0.5.
NOISE_SCALE = 0.5 

# Pronunciation Variance: 0.8 is standard.
NOISE_SCALE_W = 0.8
# ==========================================

# --- Configuration ---
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
voice_model_dir = os.path.join(project_root, "models/voice_model")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def get_best_model_path():
    """
    Automatically finds the latest run and the best_model.pth within it.
    """
    # 1. Find all run folders
    run_folders = glob.glob(os.path.join(voice_model_dir, "*run*"))
    if not run_folders:
        print("❌ Error: No training runs found in models/voice_model/")
        return None, None
    
    # 2. Sort runs by time (Newest first)
    run_folders.sort(key=os.path.getmtime, reverse=True)
    latest_run = run_folders[0]
    
    # 3. Define paths
    model_path = os.path.join(latest_run, "best_model.pth")
    config_path = os.path.join(latest_run, "config.json")

    # 4. Fallback if best_model doesn't exist yet (e.g. run just started)
    if not os.path.exists(model_path):
        print(f"⚠️  'best_model.pth' not found in {os.path.basename(latest_run)}.")
        print("   Looking for latest checkpoint...")
        checkpoints = glob.glob(os.path.join(latest_run, "checkpoint_*.pth"))
        if checkpoints:
            model_path = max(checkpoints, key=os.path.getmtime)
        else:
            print("❌ Error: No checkpoints found.")
            return None, None

    print(f"✅ Selected Model: {os.path.basename(latest_run)}/{os.path.basename(model_path)}")
    return model_path, config_path

def generate_digital_twin(text_prompt):
    print(f"\n--- 1. Processing Input Image ---")
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find image at {image_path}")
        return
    print(f"Using face image: {os.path.basename(image_path)}")

    print(f"--- 2. Generating Audio ---")
    model_path, config_path = get_best_model_path()
    
    if not model_path:
        return

    print("Loading Synthesizer... (this may take a moment)")
    # Initialize the Coqui Synthesizer
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available()
    )

    output_audio_path = os.path.join(output_folder, "generated_speech.wav")
    
    # Generate speech with Custom Knobs
    # We pass the VITS parameters as kwargs
    wav = synthesizer.tts(
        text_prompt,
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W
    )
    
    # Save the file
    synthesizer.save_wav(wav, output_audio_path)
    print(f"✅ Audio saved to: {output_audio_path}")
    print(f"   (Settings: Speed={LENGTH_SCALE}, Noise={NOISE_SCALE})")

    print(f"\n--- 3. Lip Syncing (Animation) ---")
    output_video_path = os.path.join(output_folder, "final_result.mp4")
    
    # Added --resize_factor 1 to improve face quality
    wav2lip_cmd = (
        f"python3 Wav2Lip/inference.py "
        f"--checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth "
        f"--face {image_path} "
        f"--audio {output_audio_path} "
        f"--outfile {output_video_path} "
        f"--resize_factor 1"
    )
    
    print("To animate the face, copy/paste this command into your terminal:")
    print("-" * 60)
    print(wav2lip_cmd)
    print("-" * 60)

if __name__ == "__main__":
    text = input("\n🗣️  Enter text for your Digital Twin: ")
    generate_digital_twin(text)