import os
import glob
import torch
from TTS.utils.synthesizer import Synthesizer

# --- Configuration ---
project_root = os.getcwd()
image_path = os.path.join(project_root, "image_data", "my_face2.jpg")
output_folder = os.path.join(project_root, "outputs")
voice_model_dir = os.path.join(project_root, "models/voice_model")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

def get_best_model_path():
    """Finds the best checkpoint in the voice_model folder."""
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
        print("Error: Could not find trained model.")
        return

    print(f"Loading custom voice from: {model_path}")
    
    # FIX: Use Synthesizer instead of TTS() wrapper
    # This bypasses the attribute errors for custom models
    synthesizer = Synthesizer(
        tts_checkpoint=model_path,
        tts_config_path=config_path,
        use_cuda=torch.cuda.is_available()
    )

    output_audio_path = os.path.join(output_folder, "generated_speech.wav")
    
    # Generate speech
    # The synthesizer returns a list of audio values
    wav = synthesizer.tts(text_prompt)
    
    # Save the file
    synthesizer.save_wav(wav, output_audio_path)
    print(f"Audio saved to: {output_audio_path}")

    print(f"--- 3. Lip Syncing (Animation) ---")
    output_video_path = os.path.join(output_folder, "final_result.mp4")
    
    wav2lip_cmd = (
        f"python3 Wav2Lip/inference.py "
        f"--checkpoint_path Wav2Lip/checkpoints/wav2lip_gan.pth "
        f"--face {image_path} "
        f"--audio {output_audio_path} "
        f"--outfile {output_video_path}"
    )
    
    print("\nTo animate the face (Lip Sync), run the following command manually")
    print("inside your 'venv_video' environment:\n")
    print(wav2lip_cmd)

if __name__ == "__main__":
    text = input("Enter the text for your digital twin to speak: ")
    generate_digital_twin(text)