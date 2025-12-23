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
    sentence_endings = re.compile(r'(?<=[.!?])\s+')

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
            r.raise_for_status()
            
            for line in r.iter_lines():
                if line:
                    body = json.loads(line)
                    if "response" in body:
                        token = body["response"]
                        sentence_buffer += token
                        
                        # Check for full sentence completion
                        if sentence_endings.search(sentence_buffer):
                            parts = sentence_endings.split(sentence_buffer)
                            
                            # Process every complete sentence found
                            for part in parts[:-1]:
                                if part.strip():
                                    # 1. Generate Voice
                                    audio_path, duration = synthesize_audio(part.strip())
                                    
                                    # 2. Update UI Text
                                    history[-1][1] += part + " "
                                    
                                    # 3. Send Text & Audio to Browser
                                    yield history, audio_path
                                    
                                    # 4. PAUSE Loop (Audio Pacing)
                                    # Wait for the audio to finish playing before generating next chunk
                                    if duration > 0:
                                        time.sleep(duration + 0.2) 
                            
                            # Keep incomplete remainder in buffer
                            sentence_buffer = parts[-1]

            # Process any remaining text at the end of the stream
            if sentence_buffer.strip():
                history[-1][1] += sentence_buffer
                audio_path, duration = synthesize_audio(sentence_buffer.strip())
                yield history, audio_path
                # Final sleep not strictly necessary, but good for cleanup
                time.sleep(duration + 0.2)

    except Exception as e:
        history[-1][1] += f"\n[Error: {str(e)}]"
        yield history, None

# =========================================================================
# 🖥️ GRADIO UI (Version 3.x Compatible)
# =========================================================================

with gr.Blocks(title="Digital Twin Voice Chat") as demo:
    gr.Markdown("## 🤖 Digital Twin Interface")
    
    # Classic Gradio Chatbot (No type='messages')
    chatbot = gr.Chatbot(label="Conversation")
    
    with gr.Row():
        msg = gr.Textbox(label="Type your message...", scale=4)
        submit = gr.Button("Send", scale=1)
    
    # Autoplay is TRUE so you hear it immediately
    audio_out = gr.Audio(label="Voice Output", autoplay=True, visible=True)

    def clear_msg(): return ""

    # Wire up the events
    msg.submit(chat_pipeline, [msg, chatbot], [chatbot, audio_out]) \
       .then(clear_msg, None, msg)
       
    submit.click(chat_pipeline, [msg, chatbot], [chatbot, audio_out]) \
          .then(clear_msg, None, msg)

if __name__ == "__main__":
    # Launch with public link enabled
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)