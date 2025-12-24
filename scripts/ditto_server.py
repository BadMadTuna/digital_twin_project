# scripts/ditto_server.py
import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# Add Ditto to path so we can import modules
sys.path.append(os.path.expanduser("~/digital_twin_project/Ditto"))
from stream_pipeline_offline import StreamSDK

app = FastAPI()

# Global variable to hold the model
SDK = None
AVATAR_PATH = os.path.expanduser("~/digital_twin_project/assets/avatar.jpg")

class VideoRequest(BaseModel):
    audio_path: str
    output_path: str

@app.on_event("startup")
def load_model():
    global SDK
    print("🚀 [Server] Loading Ditto Model... (This takes ~20s)")
    
    # HARDCODED PATHS (Adjust if needed)
    ditto_root = os.path.expanduser("~/digital_twin_project/Ditto")
    cfg_pkl = os.path.join(ditto_root, "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
    data_root = os.path.join(ditto_root, "checkpoints/ditto_trt_Ampere_Plus")
    
    # Initialize the SDK ONCE
    SDK = StreamSDK(cfg_pkl, data_root)
    SDK.reset_avatar(AVATAR_PATH) # Pre-load the avatar
    print("✅ [Server] Model Loaded & Ready!")

@app.post("/generate")
async def generate_video(req: VideoRequest):
    if not SDK:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    print(f"⚡ Processing: {req.audio_path}")
    
    try:
        # Run generation
        SDK.process_audio(req.audio_path)
        
        # Save video manually (reusing logic from inference loop)
        import cv2
        import imageio
        
        # Create a writer
        writer = imageio.get_writer(req.output_path, fps=25, codec='libx264', audio_path=req.audio_path)
        
        while True:
            frame = SDK.get_next_frame()
            if frame is None:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(frame_rgb)
            
        writer.close()
        
        return {"status": "success", "video_path": req.output_path}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # We run on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)