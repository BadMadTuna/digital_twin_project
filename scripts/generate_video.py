import os
import glob
import torch
from TTS.utils.synthesizer import Synthesizer

# --- Configuration ---
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
voice_model_dir = os.path.join(project_root, "models/voice_model")

# ==============================================================================
# OPTIONAL: Force a specific model if the "latest" one sounds bad.
# Example: "/home/ubuntu/digital_twin_project/models/voice_model/run-X/checkpoint_96.pth"
# Leave as None to auto-detect the latest best model.
# ==============================================================================
MANUAL_CHECKPOINT_PATH = None
MANUAL_CONFIG_PATH = None 
# ==============================================================================

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def get_best_model_path():
    """
    Finds the best checkpoint in the voice_model folder.
    Prioritizes:
    1. Manual paths set above.
    2. 'best_model.pth' in the most recent run folder.
    3. The highest numbered 'checkpoint_X.pth' in the most recent run folder.
    """
    
    # 1. Check Manual Override
    if MANUAL_CHECKPOINT_PATH and MANUAL_CONFIG_PATH:
        if os.path.exists(MANUAL_CHECKPOINT_PATH) and os.path.exists(MANUAL_CONFIG_PATH):
            print(f"-> Using manually defined model: {os.path.basename(MANUAL_CHECKPOINT_PATH)}")
            return MANUAL_CHECKPOINT_PATH, MANUAL_CONFIG_PATH
        else:
            print(f"Warning: Manual paths defined but not found. Falling back to auto-search.")

    # 2. Find all run folders
    run_folders = glob.glob(os.path.join(voice_model_dir, "*run*"))
    if not run_folders:
        print("No training run found. Have you run scripts/train_voice.py?")
        return None, None
    
    # Sort runs by time (Newest first)
    run_folders.sort(key=os.path.getmtime, reverse=True)

    # 3. Iterate through runs to find a valid model
    for run in run_folders:
        config = os.path.join(run, "config.json")
        if not os.path.exists(config):
            continue
            
        # A. Look for 'best_model.pth' (Created by Trainer if eval improves)
        best_model = os.path.join(run, "best_model.pth")
        if os.path.exists(best_model):
            print(f"-> Found best_model.pth in {os.path.basename(run)}")
            return best_model, config
        
        # B. Fallback: Find highest numbered checkpoint (e.g. checkpoint_100.pth)
        checkpoints = glob.glob(os.path.join(run, "checkpoint_*.pth"))
        if checkpoints:
            # Sort by the number in the filename
            try:
                latest_ckpt = max(checkpoints, key=lambda p: int(p.split('_')[-1].split('.')[0]))
                print(f"-> Found latest checkpoint {os.path.basename(latest_ckpt)} in {os.path.basename(run)}")
                return latest_ckpt, config
            except ValueError:
                continue

    print("Error: Could not find any valid .pth checkpoints in recent runs.")
    return None, None

def generate_digital_twin(text_prompt):
    print(f"--- 1. Processing Input Image ---")
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    print(f"Using face image: {image_path}")

    print(f"--- 2. Generating Audio ---")
    model_path, config_path = get_best_model_path()
    
    if not model_path:
        print("Error: Could not find trained model.")
        return

    print(f"Loading voice model from: {model_path}")
    
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available()
    )

    output_audio_path = os.path.join(output_folder, "generated_speech.wav")
    
    # Generate speech
    wav = synthesizer.tts(text_prompt)
    synthesizer.save_wav(wav, output_audio_path)
    print(f"Audio saved to: {output_audio_path}")

    print(f"--- 3. Lip Syncing (Animation) ---")
    output_video_path = os.path.join(output_folder, "final_result.mp4")
    
    wav2lip_cmd = (
        f"python3 Wav2Lip/inference.py "
        f"--checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth "
        f"--face {image_path} "
        f"--audio {output_audio_path} "
        f"--outfile {output_video_path} "
        f"--resize_factor 1" 
    )
    
    print("\nTo animate the face, run this command in your 'venv_video' terminal:")
    print("-" * 50)
    print(wav2lip_cmd)
    print("-" * 50)

if __name__ == "__main__":
    text = input("Enter the text for your digital twin to speak: ")
    generate_digital_twin(text)