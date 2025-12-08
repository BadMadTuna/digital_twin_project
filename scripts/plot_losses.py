import re
import matplotlib.pyplot as plt

# File path to your uploaded log
log_file_path = 'trainer_0_log.txt'

def parse_log_file(filepath):
    # Regex to strip ANSI color codes (e.g. [1m, [0m)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    # Store data: { 'loss_name': { step: value, ... } }
    data = {}
    current_step = None
    
    with open(filepath, 'r') as f:
        for line in f:
            # Clean line of special formatting
            clean_line = ansi_escape.sub('', line).strip()
            
            # 1. Capture GLOBAL_STEP
            # Format: --> TIME: ... -- STEP: 10/35 -- GLOBAL_STEP: 2775
            if "GLOBAL_STEP:" in clean_line:
                parts = clean_line.split("GLOBAL_STEP:")
                if len(parts) > 1:
                    try:
                        # Extract the number immediately following GLOBAL_STEP:
                        step_str = parts[1].split()[0]
                        current_step = int(step_str)
                    except ValueError:
                        continue

            # 2. Capture Losses (only if we have a valid current_step)
            # Format: | > loss_disc: 2.8341... (2.8593...)
            if current_step is not None and clean_line.startswith("| >"):
                # Remove "| > " prefix
                content = clean_line.replace("| > ", "")
                # Split key and value by ":"
                if ":" in content:
                    key_part, val_part = content.split(":", 1)
                    key = key_part.strip()
                    
                    # Extract the first number (current batch loss)
                    # We ignore the value in parenthesis (running avg) for raw plotting
                    val_str = val_part.strip().split()[0]
                    
                    try:
                        val = float(val_str)
                        
                        if key not in data:
                            data[key] = {'steps': [], 'values': []}
                        
                        data[key]['steps'].append(current_step)
                        data[key]['values'].append(val)
                    except ValueError:
                        continue
                        
    return data

def plot_losses(data):
    if not data:
        print("No valid data found in log file.")
        return

    # metrics_to_plot = [k for k in data.keys() if "loss" in k]
    # Filter for main losses to keep plot clean (ignoring individual disc_real_x for summary)
    # We plot everything that doesn't end in a digit (like real_0, real_1) to avoid clutter,
    # or just plot specific important ones.
    
    main_metrics = [k for k in data.keys() if not re.search(r'_\d+$', k)]
    
    num_plots = len(main_metrics)
    cols = 2
    rows = (num_plots + 1) // cols
    
    plt.figure(figsize=(15, 4 * rows))
    
    for i, metric in enumerate(sorted(main_metrics)):
        plt.subplot(rows, cols, i + 1)
        steps = data[metric]['steps']
        values = data[metric]['values']
        
        plt.plot(steps, values, label=metric, alpha=0.7, linewidth=1.5)
        plt.title(metric.upper().replace("_", " "))
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
    plt.tight_layout()
    plt.show()

# --- Execution ---
parsed_data = parse_log_file(log_file_path)
plot_losses(parsed_data)