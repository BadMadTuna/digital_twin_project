import os
import re
import csv
import soundfile as sf
from num2words import num2words
from tqdm import tqdm

# === CONFIGURATION ===
INPUT_CSV = "metadata.csv"
OUTPUT_CSV = "metadata_cleaned.csv"
WAVS_DIR = "audio_data/dataset/wavs" # Adjust if your wavs are elsewhere
MIN_DURATION = 1.0  # Seconds
MAX_DURATION = 10.0 # Seconds (VITS typically struggles > 10s)
# =====================

def clean_text(text):
    """
    Normalizes raw text into 'spoken' text.
    """
    # 1. Expand basic symbols
    text = text.replace("%", " percent")
    text = text.replace("&", " and ")
    text = text.replace("+", " plus ")
    text = text.replace("@", " at ")
    
    # 2. Convert currency (basic)
    text = text.replace("$", " dollars ")
    text = text.replace("£", " pounds ")

    # 3. Regex to find numbers and convert them
    # This finds integers and floats (e.g., 1937, 2.3)
    def replace_num(match):
        num_str = match.group(0)
        try:
            # Check if it's a year (4 digits, no decimal) - Optional Logic
            # if len(num_str) == 4 and num_str.isdigit() and 1900 < int(num_str) < 2030:
            #    return num2words(int(num_str), to='year')
            
            # Default: treat as cardinal number
            return num2words(num_str)
        except:
            return num_str

    text = re.sub(r'\d+(\.\d+)?', replace_num, text)

    # 4. Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_dataset():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Processing {INPUT_CSV}...")
    print(f"Filtering Audio: {MIN_DURATION}s - {MAX_DURATION}s")

    valid_rows = []
    dropped_short = 0
    dropped_long = 0
    total_files = 0

    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        # Detect delimiter
        line = f.readline()
        f.seek(0)
        delimiter = "|" if "|" in line else ","
        
        reader = csv.reader(f, delimiter=delimiter)
        
        for row in tqdm(reader):
            if len(row) < 2: continue
            
            filename = row[0].strip()
            raw_text = delimiter.join(row[1:]).strip()
            total_files += 1

            # 1. Check Audio Duration
            wav_path = os.path.join(WAVS_DIR, filename)
            # Try finding it in root if not in wavs subdir
            if not os.path.exists(wav_path):
                 wav_path = os.path.join("audio_data/dataset", filename)
            
            if os.path.exists(wav_path):
                try:
                    info = sf.info(wav_path)
                    duration = info.duration
                    
                    if duration < MIN_DURATION:
                        dropped_short += 1
                        continue
                    if duration > MAX_DURATION:
                        dropped_long += 1
                        continue
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
            else:
                print(f"Warning: Audio file not found: {filename}")
                continue

            # 2. Clean Text
            cleaned_text = clean_text(raw_text)
            valid_rows.append(f"{filename}|{cleaned_text}")

    # Save to new file
    with open(OUTPUT_CSV, 'w', encoding='utf-8') as f:
        f.write("\n".join(valid_rows))

    print("-" * 30)
    print(f"Original Count: {total_files}")
    print(f"Optimized Count: {len(valid_rows)}")
    print(f"Dropped (Too Short < {MIN_DURATION}s): {dropped_short}")
    print(f"Dropped (Too Long > {MAX_DURATION}s):  {dropped_long}")
    print(f"Saved to: {OUTPUT_CSV}")
    print("-" * 30)
    print("Example Changes:")
    for i in range(min(3, len(valid_rows))):
        print(f" > {valid_rows[i]}")

if __name__ == "__main__":
    process_dataset()