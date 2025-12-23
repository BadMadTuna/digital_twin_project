# Save this inside the Ditto folder!
import argparse
import sys
import os
import cv2
import soundfile as sf
import numpy as np
from stream_pipeline_offline import StreamSDK

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_pkl", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # Initialize SDK
    SDK = StreamSDK(args.cfg_pkl, args.data_root)
    
    # Run Generation
    SDK.reset_avatar(args.source_path)
    SDK.process_audio(args.audio_path)
    
    # Save Video
    # We use a custom loop here to ensure it saves correctly without GUI
    import imageio
    writer = imageio.get_writer(args.output_path, fps=25, codec='libx264', audio_path=args.audio_path)
    
    while True:
        frame = SDK.get_next_frame()
        if frame is None:
            break
        # Convert BGR (OpenCV) to RGB (ImageIO)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame_rgb)
        
    writer.close()

if __name__ == "__main__":
    main()