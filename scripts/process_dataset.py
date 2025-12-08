import os
import shutil
import subprocess
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm
import pandas as pd

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
INPUT_FILE = os.path.join(PROJECT_ROOT, "audio_data", "attenborough_cut.mp4")
DATASET_DIR = os.path.join(PROJECT_ROOT, "audio_data", "dataset")
WAVS_DIR = os.path.join(DATASET_DIR, "wavs")
METADATA_PATH = os.path.join(DATASET_DIR, "metadata.csv")

# Audio Settings
TARGET_SR = 22050
MIN_SILENCE_LEN = 500   # ms (0.5 seconds pause triggers a cut)
SILENCE_THRESH = -40    # dB (Adjust if it cuts mid-word or doesn't cut enough)
KEEP_SILENCE = 150      # ms (Add a tiny bit of silence back to ends for natural flow)

def extract_audio_from_video(video_path, output_wav):
    """Converts MP4 to a mono WAV file."""
    print(f"🔄 Extracting audio from {os.path.basename(video_path)}...")
    if os.path.exists(output_wav):
        os.remove(output_wav)
        
    command = [
        "ffmpeg",
        "-i", video_path,
        "-ac", "1",             # Mono
        "-ar", str(TARGET_SR),  # 22050 Hz
        "-vn",                  # No Video
        output_wav
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✅ Audio extracted.")

def process_data():
    # 1. Setup Folders (Clean Start)
    if os.path.exists(DATASET_DIR):
        print("⚠️  Cleaning up old dataset...")
        shutil.rmtree(DATASET_DIR)
    os.makedirs(WAVS_DIR, exist_ok=True)

    # 2. Extract Raw Audio
    raw_audio_path = os.path.join(PROJECT_ROOT, "audio_data", "temp_full_audio.wav")
    extract_audio_from_video(INPUT_FILE, raw_audio_path)

    # 3. Load and Split Audio
    print("✂️  Loading and splitting audio (this may take a minute)...")
    full_audio = AudioSegment.from_wav(raw_audio_path)
    
    # Split based on silence
    chunks = split_on_silence(
        full_audio,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    
    print(f"-> Found {len(chunks)} segments.")

    # 4. Save Chunks & Transcribe
    print("📝 Transcribing with OpenAI Whisper (Medium)...")
    # Load Whisper (use 'medium' for best accuracy, 'base' if GPU is weak)
    model = whisper.load_model("medium") 
    
    metadata_rows = []
    
    # Iterate through chunks
    for i, chunk in enumerate(tqdm(chunks)):
        # Skip chunks that are too short (garbage/noise)
        if len(chunk) < 1000: # Less than 1 second
            continue
            
        filename = f"segment_{i:04d}.wav"
        filepath = os.path.join(WAVS_DIR, filename)
        
        # Export chunk
        chunk.export(filepath, format="wav")
        
        # Transcribe
        result = model.transcribe(filepath)
        text = result["text"].strip()
        
        # Validate text (Skip empty or hallucinated short text)
        if len(text) < 2:
            os.remove(filepath)
            continue
            
        # Clean text (optional: remove newlines)
        text = text.replace("\n", " ")
        
        # Format for metadata.csv: filename|text
        metadata_rows.append(f"{filename}|{text}")

    # 5. Write Metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_rows))
    
    # Cleanup temp file
    if os.path.exists(raw_audio_path):
        os.remove(raw_audio_path)

    print(f"🎉 Dataset ready! {len(metadata_rows)} valid samples created.")
    print(f"-> Location: {DATASET_DIR}")

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find input file at {INPUT_FILE}")
    else:
        process_data()