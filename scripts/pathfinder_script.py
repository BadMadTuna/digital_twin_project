import os
import site

def find_dvae_file():
    # 1. Get the site-packages directory where TTS is installed
    site_packages = site.getsitepackages()[0]
    tts_path = os.path.join(site_packages, "TTS")
    
    print(f"📂 Scanning for 'dvae.py' in: {tts_path}")
    
    found_paths = []
    
    # 2. Walk through the directory tree
    for root, dirs, files in os.walk(tts_path):
        for file in files:
            # We are looking for any file that looks like it defines the DVAE
            if "dvae" in file.lower() or "tortoise" in file.lower():
                full_path = os.path.join(root, file)
                print(f"   found: {full_path}")
                found_paths.append(full_path)

    print("-" * 50)
    
    # 3. Analyze results to guess the import path
    if not found_paths:
        print("❌ No DVAE files found. Your installation might be corrupted.")
    else:
        print("✅ Analysis:")
        for path in found_paths:
            # Convert file path to python import path
            # e.g. .../TTS/tts/models/xtts/dvae.py -> TTS.tts.models.xtts.dvae
            rel_path = os.path.relpath(path, site_packages)
            import_path = rel_path.replace(os.path.sep, ".").replace(".py", "")
            print(f"   Try import: {import_path}")

if __name__ == "__main__":
    find_dvae_file()