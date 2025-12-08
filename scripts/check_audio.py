import os
import glob
import json
import numpy as np
import soundfile as sf
import sys
import argparse

# === AUTO-DETECTED PATH FROM LOGS ===
# Based on your trainer_0_log.txt
DEFAULT_CONFIG = "/home/ubuntu/digital_twin_project/models/voice_model_large/run-December-08-2025_04+29PM-557d261/config.json"
# ====================================

def load_config(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Error loading config: {e}")
        return None

def analyze_dataset(data_path, config_path):
    print(f"\n--- AUDIO DIAGNOSTIC REPORT ---")
    print(f"Dataset Path: {data_path}")
    print(f"Config Path:  {config_path}")
    print("-" * 30)

    if not os.path.exists(data_path):
        print(f"[ERROR] The dataset path does not exist: {data_path}")
        return

    # Recursive search for wavs
    files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    
    # If no wavs, try searching for flac or mp3 just in case
    if not files:
        files = glob.glob(os.path.join(data_path, "**", "*.flac"), recursive=True)
    
    if not files:
        print("[!] No audio files (.wav/.flac) found.")
        print("    Did you provide the correct folder?")
        return

    # Load config
    config = load_config(config_path)
    if config:
        print(f"[INFO] Config loaded successfully.")
    else:
        print(f"[WARN] Config file not found at {config_path}.")
        print("       (Proceeding without cross-checking config settings...)")

    # Statistics
    global_max = -float('inf')
    global_min = float('inf')
    sample_rates = set()
    dtypes = set()
    clipped_count = 0
    quiet_count = 0
    nan_count = 0
    
    # Limit check to 2000 files to be fast
    files_to_check = files[:2000]
    total_files = len(files_to_check)

    print(f"\nScanning {total_files} files...")

    for i, fpath in enumerate(files_to_check):
        try:
            data, sr = sf.read(fpath)
            
            # Mix to mono if needed
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            abs_data = np.abs(data)
            current_max = np.max(abs_data)
            
            global_max = max(global_max, current_max)
            global_min = min(global_min, np.min(data))
            
            sample_rates.add(sr)
            dtypes.add(str(data.dtype))

            if np.isnan(data).any() or np.isinf(data).any():
                nan_count += 1
            
            # Clipping check
            if 'float' in str(data.dtype) and current_max > 1.0:
                clipped_count += 1
            elif 'int' in str(data.dtype) and current_max >= 32767:
                clipped_count += 1
            
            # Quiet check
            if current_max < 0.05: 
                quiet_count += 1

            # Progress
            if i % 100 == 0:
                sys.stdout.write(f"\rProgress: {i}/{total_files}")
                sys.stdout.flush()

        except Exception as e:
            print(f"\n[!] Error reading {os.path.basename(fpath)}: {e}")

    print(f"\rProgress: {total_files}/{total_files} - Done.\n")
    
    # === RESULTS ===
    print("=" * 30)
    print("1. DATA STATS")
    print(f"   - Sample Rates: {sorted(list(sample_rates))} Hz")
    print(f"   - Data Types:   {', '.join(dtypes)}")
    print(f"   - Max Amplitude: {global_max:.4f}")
    
    if len(sample_rates) > 1:
        print("   [!] WARNING: Multiple sample rates found! This ruins training.")

    # 2. CONFIG CHECK
    if config:
        print("\n2. CONFIG CHECK")
        c_max_val = config.get('max_wav_value', 'N/A')
        c_sr = config.get('sampling_rate', 'N/A')
        
        print(f"   - Config Sample Rate: {c_sr}")
        print(f"   - Config Max Value:   {c_max_val}")

        # Logic Checks
        data_is_float = any('float' in d for d in dtypes)
        
        if isinstance(c_sr, int) and c_sr not in sample_rates:
             print(f"   [!!!] MISMATCH: Config expects {c_sr}Hz, but data is {sample_rates}.")
        else:
             print(f"   [OK] Sample rates match.")

        if data_is_float and c_max_val != 1.0:
             print(f"   [!!!] CRITICAL: Data is FLOAT (0.0-1.0), but config 'max_wav_value' is {c_max_val}.")
             print("         This is likely the cause of your high loss (~17.0).")
        elif data_is_float and c_max_val == 1.0:
             print(f"   [OK] Data format matches config expectations.")

    # 3. VOLUME ISSUES
    print("\n3. VOLUME ISSUES")
    print(f"   - Clipped Files: {clipped_count}")
    print(f"   - Quiet Files:   {quiet_count}")
    
    if quiet_count > total_files * 0.5:
        print("   [!] WARNING: More than 50% of your dataset is very quiet. Normalize it.")

    print("=" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required: The folder with your wavs
    parser.add_argument("--wavs", type=str, required=True, help="Path to the folder containing .wav files")
    # Optional: Config path (defaults to the one found in logs)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config.json")
    
    args = parser.parse_args()
    
    analyze_dataset(args.wavs, args.config)