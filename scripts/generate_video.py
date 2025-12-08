import os
import glob
import torch
from TTS.utils.synthesizer import Synthesizer

# ==========================================
# 🔒 MANUAL MODEL OVERRIDE
# Points to: run-December-07-2025_01+12PM-88a911f
# ==========================================
MODEL_DIR = "/home/ubuntu/digital_twin_project/models/voice_model/run-December-07-2025_01+12PM-88a911f"
MANUAL_CHECKPOINT = os.path.join(MODEL_DIR, "best_model.pth")
MANUAL_CONFIG     = os.path.join(MODEL_DIR, "config.json")

# ==========================================
# 🎛️ INFERENCE TUNING KNOBS
# ==========================================
# Speed: 1.0 is normal. 1.1 is slower (often clearer).
LENGTH_SCALE = 1.1 

# Emotion: 0.667 is standard. 0.5 is more stable.
NOISE_SCALE = 0.667 

# Pronunciation: 0.8 is standard.
NOISE_SCALE_W = 0.8
# ==========================================

# --- Configuration ---
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def get_best_model_path():
    """
    Returns the manually defined model path.
    """
    if os.path.exists(MANUAL_CHECKPOINT) and os.path.exists(MANUAL_CONFIG):
        print(f"✅ Using locked model: {os.path.basename(MODEL_DIR)}")
        return MANUAL_CHECKPOINT, MANUAL_CONFIG
    else:
        print(f"❌ Error: Could not find files in {MODEL_DIR}")
        print("   Please check if 'best_model.pth' and 'config.json' exist there.")
        return None, None

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
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available()
    )

    output_audio_path = os.path.join(output_folder, "generated_speech.wav")
    
    # Generate speech
    wav = synthesizer.tts(
        text_prompt,
        length_scale=LENGTH_SCALE,
        noise_scale=NOISE_SCALE,
        noise_scale_w=NOISE_SCALE_W
    )
    
    # Save the file
    synthesizer.save_wav(wav, output_audio_path)
    print(f"✅ Audio saved to: {output_audio_path}")

    print(f"\n--- 3. Lip Syncing (Animation) ---")
    output_video_path = os.path.join(output_folder, "final_result.mp4")
    
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