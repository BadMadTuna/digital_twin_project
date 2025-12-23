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
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1" 

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
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)
    model.cuda()
    model.eval()
    return model

# Global load
tts_model = load_xtts()

# =========================================================================
# 🧠 & 🗣️ PIPELINE (Legacy Format: [[User, Bot], ...])
# =========================================================================

def synthesize_audio(text):
    """Generates audio and returns path."""
    if not text.strip(): return None
    
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(
        audio_path=[REF_AUDIO_PATH], gpt_cond_len=30, max_ref_length=60
    )
    
    out = tts_model.inference(
        text, LANGUAGE, gpt_cond_latent, speaker_embedding,
        temperature=0.7, length_penalty=1.0, repetition_penalty=2.0,
        top_k=50, top_p=0.8
    )
    
    filename = f"temp_tts_{int(time.time()*1000)}.wav"
    sf.write(filename, out["wav"], 22050)
    return filename

def chat_pipeline(user_input, history):
    """
    Revised pipeline using classic List of Lists format:
    [[ "Hi", "Hello" ], [ "How are you?", "Good" ]]
    """
    if history is None: history = []
    
    # 1. Append new turn: [User Input, Empty Bot Response]
    history.append([user_input, ""])
    
    payload = {
        "model": LLM_MODEL,
        "prompt": user_input,
        "system": "You are a concise voice assistant. Keep answers short (1-2 sentences).",
        "stream": True
    }
    
    sentence_buffer = ""
    sentence_endings = re.compile(r'(?<=[.!?])\s+')

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if line:
                    body = json.loads(line)
                    if "response" in body:
                        token = body["response"]
                        
                        # Update the bot's response in the last tuple
                        history[-1][1] += token
                        sentence_buffer += token
                        
                        # Check for full sentence to speak
                        if sentence_endings.search(sentence_buffer):
                            parts = sentence_endings.split(sentence_buffer)
                            
                            for part in parts[:-1]:
                                if part.strip():
                                    audio_path = synthesize_audio(part.strip())
                                    yield history, audio_path
                            
                            sentence_buffer = parts[-1]
                        else:
                            # Yield text update only
                            yield history, None

            # Process remainder
            if sentence_buffer.strip():
                audio_path = synthesize_audio(sentence_buffer.strip())
                yield history, audio_path

    except Exception as e:
        history[-1][1] += f"\n[Error: {str(e)}]"
        yield history, None

# =========================================================================
# 🖥️ GRADIO UI (Version 3.x Compatible)
# =========================================================================

with gr.Blocks(title="Digital Twin Voice Chat") as demo:
    gr.Markdown("## 🤖 Digital Twin Interface")
    
    # REMOVED type="messages" to support older Gradio
    chatbot = gr.Chatbot(label="Conversation")
    
    with gr.Row():
        msg = gr.Textbox(label="Type your message...", scale=4)
        submit = gr.Button("Send", scale=1)
    
    audio_out = gr.Audio(label="Voice Output", autoplay=True, visible=True)

    def clear_msg(): return ""

    msg.submit(chat_pipeline, [msg, chatbot], [chatbot, audio_out]) \
       .then(clear_msg, None, msg)
       
    submit.click(chat_pipeline, [msg, chatbot], [chatbot, audio_out]) \
          .then(clear_msg, None, msg)

if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)