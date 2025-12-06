import os
import whisper
import pandas as pd
from pydub import AudioSegment
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_FILE = "audio_data/attenborough_ref.mp3"  # Your input file
OUTPUT_DIR = "audio_data/dataset"                 # Where slices go
METADATA_FILE = "metadata.csv"
SAMPLE_RATE = 22050                               # Standard for XTTS

def format_filename(text, index):
    # Create a safe filename from text (optional, or just use number)
    safe_text = "".join([c for c in text if c.isalnum() or c in (' ', '_')]).rstrip()
    safe_text = safe_text.replace(" ", "_").lower()[:20]
    return f"wavs/seq_{index:04d}_{safe_text}.wav"

def main():
    # 1. Setup paths
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / SOURCE_FILE
    output_dir = base_dir / OUTPUT_DIR / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model... (Using GPU if available)")
    # 'medium' is a good balance. Use 'large-v3' if you want perfection (slower)
    model = whisper.load_model("medium") 
    
    print(f"Transcribing {input_path.name}...")
    # This does the magic: finding segments and timestamps
    result = model.transcribe(str(input_path))
    segments = result["segments"]
    
    print(f"Loading audio for slicing...")
    original_audio = AudioSegment.from_file(input_path)
    
    metadata = []
    
    print(f"Slicing {len(segments)} segments...")
    for i, seg in enumerate(segments):
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        text = seg["text"].strip()
        
        # Skip very short segments (< 1 second)
        if end_ms - start_ms < 1000:
            continue
            
        # Cut and Resample
        chunk = original_audio[start_ms:end_ms]
        chunk = chunk.set_frame_rate(SAMPLE_RATE).set_channels(1)
        
        # Save File
        filename = f"seq_{i:04d}.wav"
        save_path = output_dir / filename
        chunk.export(save_path, format="wav")
        
        # Add to metadata (Format: filename|text)
        # Note: metadata.csv usually expects path relative to its location or just filename
        metadata.append(f"{filename}|{text}")
        
    # Save Metadata CSV
    csv_path = base_dir / OUTPUT_DIR / METADATA_FILE
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
        
    print(f"Done! Created {len(metadata)} slices in {OUTPUT_DIR}")
    print(f"Metadata saved to {csv_path}")

if __name__ == "__main__":
    main()