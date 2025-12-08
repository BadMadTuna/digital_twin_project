import matplotlib.pyplot as plt
import re
import os

# Configuration
LOG_FILE = "trainer_0_log.txt"
OUTPUT_IMAGE = "loss_plot.jpg"

def parse_log_file(filepath):
    """
    Parses the log file to extract epoch numbers and average losses
    from the EVAL PERFORMANCE sections.
    """
    epochs = []
    loss_mel = []
    loss_kl = []
    
    # Regex to find the Epoch number
    # Example: > EPOCH: 0/1000
    epoch_pattern = re.compile(r">\s*EPOCH:\s*(\d+)/")
    
    # Regex to find average losses in the EVAL PERFORMANCE block
    # Handles ANSI color codes (e.g., [92m) and whitespace
    # Example: | > avg_loss_kl: [92m 2.4584643500191823  [0m(+0)
    mel_pattern = re.compile(r"avg_loss_mel:\s*(?:\[\d+m\s*)?([\d\.]+)")
    kl_pattern = re.compile(r"avg_loss_kl:\s*(?:\[\d+m\s*)?([\d\.]+)")

    current_epoch = -1
    
    # Temporary holders for the current block
    temp_mel = None
    temp_kl = None

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None, None, None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Check for new Epoch
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                # If we have data from the previous epoch, save it
                if current_epoch != -1 and temp_mel is not None and temp_kl is not None:
                    epochs.append(current_epoch)
                    loss_mel.append(temp_mel)
                    loss_kl.append(temp_kl)
                    
                    # Reset for next
                    temp_mel = None
                    temp_kl = None
                
                current_epoch = int(epoch_match.group(1))
                continue

            # Check for Mel Loss
            mel_match = mel_pattern.search(line)
            if mel_match:
                temp_mel = float(mel_match.group(1))

            # Check for KL Loss
            kl_match = kl_pattern.search(line)
            if kl_match:
                temp_kl = float(kl_match.group(1))

        # Append the final epoch if data exists
        if current_epoch != -1 and temp_mel is not None and temp_kl is not None:
            epochs.append(current_epoch)
            loss_mel.append(temp_mel)
            loss_kl.append(temp_kl)

    return epochs, loss_mel, loss_kl

def plot_losses(epochs, loss_mel, loss_kl):
    """Generates a plot with two subplots for Mel and KL loss."""
    if not epochs:
        print("No data found to plot. Check log file format.")
        return

    plt.figure(figsize=(12, 8))

    # Plot Mel Loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_mel, marker='o', linestyle='-', color='b', label='avg_loss_mel')
    plt.title("Training Loss over Epochs")
    plt.ylabel("Mel Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot KL Loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss_kl, marker='o', linestyle='-', color='r', label='avg_loss_kl')
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, format='jpg', dpi=150)
    print(f"Plot saved successfully to {OUTPUT_IMAGE}")

if __name__ == "__main__":
    print(f"Reading {LOG_FILE}...")
    ep, mel, kl = parse_log_file(LOG_FILE)
    
    if ep:
        print(f"Found {len(ep)} epochs. Generating plot...")
        plot_losses(ep, mel, kl)
    else:
        print("Could not parse data.")