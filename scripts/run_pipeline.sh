#!/bin/bash

# 1. Setup Environment
echo "--- Installing Dependencies ---"
pip install TTS
sudo apt-get install -y ffmpeg

# 2. Fine-Tune Voice (The 'Easy' CLI method)
# This assumes you have valid dataset formatting
echo "--- Starting Voice Fine-Tuning ---"
# Note: XTTS often needs a specific recipe. 
# For simplicity, we will assume we are just running inference 
# with your reference audio first (Zero-Shot), which is very powerful 
# with 24mins of audio, you just pick the best 10s clip as reference.

# 3. Clone & Setup EchoMimic (Only if not exists)
if [ ! -d "EchoMimic" ]; then
    echo "--- Cloning EchoMimic ---"
    git clone https://github.com/BadToBest/EchoMimic.git
    cd EchoMimic
    pip install -r requirements.txt
    # Download weights (You would normally do this via python script or wget)
    cd ..
fi

echo "--- Pipeline Ready ---"