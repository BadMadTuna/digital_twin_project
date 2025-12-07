import os
import torch
import glob
from TTS.api import TTS

# --- Configuration ---
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
voice_model_dir = os.path.join(project_root, "models/voice_model")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def get_best_model_path():
    """Finds the best checkpoint in the voice_model folder."""
    # Look for the folder created by the trainer (usually date-stamped)
    run_folders = glob.glob(os.path.join(voice_model_dir, "*run*"))
    if not run_folders:
        print("No training run found. Have you run scripts/train_voice.py?")
        return None, None
    
    # Get the latest run
    latest_run = max(run_folders, key=os.path.getmtime)
    
    # Get the best_model.pth inside that run
    checkpoint = os.path.join(latest_run, "best_model.pth")
    config = os.path.join(latest_run, "config.json")
    
    return checkpoint, config

def generate_digital_twin(text_prompt):
    print(f"--- 1. Processing Input Image ---")
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
    print(f"Using face image: {image_path}")

    print(f"--- 2. Generating Audio ---")
    model_path, config_path = get_best_model_path()
    
    if not model_path:
        print("Using default TTS (Training not complete).")
        # Fallback to generic TTS if training isn't done
        tts = TTS("tts_models/en/ljspeech/vits")
    else:
        print(f"Loading custom voice from: {model_path}")
        tts = TTS(model_path=model_path, config_path=config_path, progress_bar=True, gpu=True)

    output_audio_path = os.path.join(output_folder, "generated_speech.wav")
    
    # Generate speech
    tts.tts_to_file(text=text_prompt, file_path=output_audio_path)
    print(f"Audio saved to: {output_audio_path}")

    print(f"--- 3. Lip Syncing (Animation) ---")
    output_video_path = os.path.join(output_folder, "final_twin.mp4")
    
    # NOTE: This assumes a Wav2Lip inference script is available.
    # If running locally, you typically call the inference.py from the Wav2Lip repo.
    # We will construct the command for you to run, or run it if configured.
    
    wav2lip_cmd = (
        f"python3 Wav2Lip/inference.py "
        f"--checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth "
        f"--face {image_path} "
        f"--audio {output_audio_path} "
        f"--outfile {output_video_path}"
    )
    
    print("\nTo animate the face, ensure you have Wav2Lip cloned and checkpoints downloaded.")
    print("Run the following command:\n")
    print(wav2lip_cmd)
    
    # Optional: Automatically run it if Wav2Lip is present
    if os.path.exists("Wav2Lip/inference.py"):
        print("\nWav2Lip found! Executing lip sync...")
        os.system(wav2lip_cmd)
        print(f"\nVideo generated at: {output_video_path}")

if __name__ == "__main__":
    text = input("Enter the text for your digital twin to speak: ")
    generate_digital_twin(text)