import os
import time
import requests
import json
import re
import torch
import gradio as gr
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================
# The Brain (Ollama)
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1" 

# The Voice (XTTS Paths)
PROJECT_ROOT = os.getcwd()
TRAIN_RUN_NAME = "xtts_finetuned-December-11-2025_02+59PM-bae2302"
CHECKPOINT_NAME = "checkpoint_5000.pth"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", TRAIN_RUN_NAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset/wavs/segment_0330.wav")
LANGUAGE = "en"

# =========================================================================
# 🔊 LOAD ENGINE
# =========================================================================
def load_xtts():
    print("⏳ Loading XTTS Model...")
    config = XttsConfig()
    config.load_json(CONFIG_PATH)
    model = Xtts.init_from_config(config)
    
    # Tokenizer setup (fixes common path issues)
    base_dir = os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2")
    vocab_file = os.path.join(base_dir, "vocab.json")
    if not os.path.exists(vocab_file):
        import TTS.utils.manage as manage
        try:
            path = manage.ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
            vocab_file = os.path.join(os.path.dirname(path), "vocab.json")
        except:
            print("⚠️ Could not verify vocab file. Assuming standard path.")

    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)
    
    # Load the specific fine-tuned checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)
    model.cuda()
    model.eval()
    return model

# Load model once at startup
tts_model = load_xtts()

# =========================================================================
# 🧠 & 🗣️ PIPELINE (Paced & Synced)
# =========================================================================

def synthesize_audio(text):
    """
    Generates audio for a text chunk.
    Returns: (file_path, duration_in_seconds)
    """
    if not text.strip(): return None, 0
    
    # Conditioning latents (Clone the voice from the reference file)
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path=[REF_AUDIO_PATH], gpt_cond_len=30, max_ref_length=60
    )
    
    # Run Inference
    out = tts_model.inference(
        text, LANGUAGE, gpt_cond_latent, speaker_embedding,
        temperature=0.7, length_penalty=1.0, repetition_penalty=2.0,
        top_k=50, top_p=0.8
    )
    
    # Calculate Duration (Wav Array Length / Sample Rate)
    # We save at 22050Hz, so we divide by that to get seconds
    sample_rate = 22050
    duration = len(out["wav"]) / sample_rate
    
    filename = f"temp_tts_{int(time.time()*1000)}.wav"
    sf.write(filename, out["wav"], sample_rate)
    
    return filename, duration

def chat_pipeline(user_input, history):
    """
    1. Sends prompt to Ollama.
    2. Buffers stream into sentences.
    3. Synthesizes sentence.
    4. Yields to UI -> Waits for audio to finish -> Repeats.
    """
    if history is None: history = []
    
    # Add new turn to history: [User, Bot (Empty)]
    history.append([user_input, ""])
    
    payload = {
        "model": LLM_MODEL,
        "prompt": user_input,
        "system": "You are a concise voice assistant. Keep answers short (1-2 sentences).",
        "stream": True
    }
    
    sentence_buffer = ""
    # Regex splits by . ! ? followed by space/newline
    sentence_endings = re.compile(r'(?<=[.!?