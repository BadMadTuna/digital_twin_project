import re
import csv
import os

# --- CONFIGURATION ---
INPUT_LOG = "models/voice_model/run-December-09-2025_03+49PM-8d1e3b8/trainer_0_log.txt"
OUTPUT_CSV = "training_metrics.csv"

def strip_ansi(text):
    """Removes ANSI color codes from log text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def extract_metrics():
    # Helper storage
    data_map = {} # Key: epoch, Value: {mel: x, kl: y}
    current_epoch = -1

    # Regex patterns
    # Matches: " > EPOCH: 5/1000"
    epoch_pattern = re.compile(r">\s*EPOCH:\s*(\d+)/")
    # Matches: "| > avg_loss_mel: 17.55..."
    mel_pattern = re.compile(r"\|\s*>\s*avg_loss_mel:\s*([\d\.]+)")
    # Matches: "| > avg_loss_kl: 2.33..."
    kl_pattern = re.compile(r"\|\s*>\s*avg_loss_kl:\s*([\d\.]+)")

    if not os.path.exists(INPUT_LOG):
        print(f"❌ Error: Could not find {INPUT_LOG}")
        return

    print(f"🔄 Reading {INPUT_LOG}...")
    
    with open(INPUT_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = strip_ansi(line)

            # 1. Detect Epoch Change
            ep_match = epoch_pattern.search(clean_line)
            if ep_match:
                current_epoch = int(ep_match.group(1))
                if current_epoch not in data_map:
                    data_map[current_epoch] = {}
                continue

            # 2. Extract Metrics (Only if we have a valid epoch)
            if current_epoch != -1:
                # Check for Mel
                mel_match = mel_pattern.search(clean_line)
                if mel_match:
                    data_map[current_epoch]['loss_mel'] = mel_match.group(1)

                # Check for KL
                kl_match = kl_pattern.search(clean_line)
                if kl_match:
                    data_map[current_epoch]['loss_kl'] = kl_match.group(1)

    # 3. Write to CSV
    print(f"📝 Writing to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'loss_kl', 'loss_mel']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        # Sort by epoch to ensure order
        for epoch in sorted(data_map.keys()):
            metrics = data_map[epoch]
            # Only write rows that have both metrics (avoids partial/broken logs)
            if 'loss_mel' in metrics and 'loss_kl' in metrics:
                writer.writerow({
                    'epoch': epoch,
                    'loss_kl': metrics['loss_kl'],
                    'loss_mel': metrics['loss_mel']
                })

    print(f"✅ Done! Extracted {len(data_map)} epochs.")

if __name__ == "__main__":
    extract_metrics()