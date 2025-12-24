import sys
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Setup Paths so we can import Ditto modules
DITTO_ROOT = os.path.expanduser("~/digital_twin_project/Ditto")
sys.path.append(DITTO_ROOT)

# 2. Import the Ditto SDK and the official 'run' logic
from stream_pipeline_offline import StreamSDK
# We import 'run' from inference.py to reuse its logic
from inference import run as run_ditto_inference

app = FastAPI()

# Global variables
SDK = None
# We define the avatar path here, but we pass it during generation
AVATAR_PATH = os.path.expanduser("~/digital_twin_project/assets/avatar.jpg")
CFG_PATH = os.path.join(DITTO_ROOT, "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
DATA_ROOT = os.path.join(DITTO_ROOT, "checkpoints/ditto_trt_Ampere_Plus")

class VideoRequest(BaseModel):
    audio_path: str
    output_path: str

@app.on_event("startup")
def load_model():
    """
    Loads the heavy TensorRT engine into GPU memory ONCE at startup.
    """
    global SDK
    print("🚀 [Server] Loading Ditto Model... (This takes ~20s)")
    try:
        # Initialize the heavy engine
        SDK = StreamSDK(CFG_PATH, DATA_ROOT)
        print("✅ [Server] Model Loaded & Ready!")
    except Exception as e:
        print(f"❌ [Server] Failed to load model: {e}")
        sys.exit(1)

@app.post("/generate")
async def generate_video(req: VideoRequest):
    """
    Receives an audio path, generates video using the pre-loaded SDK.
    """
    if not SDK:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    print(f"⚡ Processing Audio: {req.audio_path}")
    
    try:
        # We reuse the official 'run' function from inference.py
        # It handles librosa loading, frame calculations, and the loop.
        run_ditto_inference(
            SDK, 
            req.audio_path, 
            AVATAR_PATH, 
            req.output_path
        )
        
        if os.path.exists(req.output_path):
            return {"status": "success", "video_path": req.output_path}
        else:
            raise HTTPException(status_code=500, detail="Video file was not created.")
            
    except Exception as e:
        print(f"❌ Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)