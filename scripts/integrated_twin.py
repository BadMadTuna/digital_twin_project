import os
import time
import requests
import json
import re
import torch
import subprocess
import gradio as gr
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer

# =========================================================================
# ⚙️ CONFIGURATION
# =========================================================================
# 1. BRAIN & VOICE
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1"
PROJECT_ROOT = os.getcwd()
TRAIN_RUN_NAME = "xtts_finetuned-December-11-2025_02+59PM-bae2302"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", TRAIN_RUN_NAME)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint_5000.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")
REF_AUDIO_PATH = os.path.join(PROJECT_ROOT, "audio_data/dataset/wavs/segment_0330.wav")
LANGUAGE = "en"

# 2. VIDEO (DITTO) CONFIGURATION
# We use absolute paths to ensure the subprocess finds everything
DITTO_PYTHON = os.path.expanduser("~/digital_twin_project/venv_ditto/bin/python")
# CHANGED: Pointing to the official inference script
DITTO_SCRIPT = os.path.expanduser("~/digital_twin_project/Ditto/inference.py")
DITTO_ROOT = os.path.expanduser("~/digital_twin_project/Ditto")
DITTO_CFG = os.path.join(DITTO_ROOT, "checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl")
DITTO_DATA = os.path.join(DITTO_ROOT, "checkpoints/ditto_trt_Ampere_Plus")
AVATAR_IMAGE = os.path.expanduser("~/digital_twin_project/assets/avatar.jpg")

# =========================================================================
# 🔊 LOAD VOICE ENGINE
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
        except: pass

    model.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_file)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint, strict=False)
    model.cuda()
    model.eval()
    return model

tts_model = load_xtts()

# =========================================================================
# 🎥 VIDEO GENERATION BRIDGE
# =========================================================================
def generate_ditto_video(audio_path):
    """
    Sends a request to the always-on Ditto Server (Port 8000).
    """
    output_video = audio_path.replace(".wav", ".mp4")
    
    # The API Payload
    payload = {
        "audio_path": audio_path,
        "output_path": output_video
    }
    
    try:
        # Hit the local server
        response = requests.post("http://localhost:8000/generate", json=payload)
        response.raise_for_status() # Check for errors
        return output_video
    except Exception as e:
        print(f"❌ Video Server Error: Is 'scripts/ditto_server.py' running?\n{e}")
        return None
    
# =========================================================================
# 🧠 MAIN PIPELINE
# =========================================================================
def synthesize_audio(text):
    if not text.strip(): return None, 0
    gpt_cond_latent, speaker_embedding = tts_model.get_conditioning_latents(audio_path=[REF_AUDIO_PATH], gpt_cond_len=30, max_ref_length=60)
    out = tts_model.inference(text, LANGUAGE, gpt_cond_latent, speaker_embedding, temperature=0.7, length_penalty=1.0, repetition_penalty=2.0, top_k=50, top_p=0.8)
    
    # SAVE AS ABSOLUTE PATH to avoid subprocess finding errors
    filename = f"temp_{int(time.time()*1000)}.wav"
    abs_path = os.path.abspath(filename)
    
    sf.write(abs_path, out["wav"], 22050)
    duration = len(out["wav"]) / 22050
    return abs_path, duration

def chat_pipeline(user_input, mode, history):
    if history is None: history = []
    history.append([user_input, ""])
    
    payload = {"model": LLM_MODEL, "prompt": user_input, "system": "You are a helpful assistant. Keep answers short (1 sentence).", "stream": True}
    
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
                        sentence_buffer += token
                        
                        if sentence_endings.search(sentence_buffer):
                            parts = sentence_endings.split(sentence_buffer)
                            for part in parts[:-1]:
                                if part.strip():
                                    # 1. Generate Audio
                                    audio_path, duration = synthesize_audio(part.strip())
                                    
                                    # 2. Generate Video (If Mode is ON)
                                    final_output = audio_path
                                    video_result = None
                                    
                                    if mode == "Video (Ditto)":
                                        video_result = generate_ditto_video(audio_path)
                                        if video_result: final_output = video_result
                                    
                                    # 3. Update UI
                                    history[-1][1] += part + " "
                                    
                                    # Return appropriate output
                                    if mode == "Voice Only":
                                        yield history, final_output, None
                                    else:
                                        if video_result:
                                            yield history, None, final_output
                                        else:
                                            # Fallback to audio if video fails
                                            yield history, audio_path, None
                                    
                                    # Pacing
                                    if duration > 0: time.sleep(duration + 0.2)

                            sentence_buffer = parts[-1]
            
            # Final Chunk
            if sentence_buffer.strip():
                history[-1][1] += sentence_buffer
                audio_path, duration = synthesize_audio(sentence_buffer.strip())
                
                final_output = audio_path
                video_result = None

                if mode == "Video (Ditto)":
                    video_result = generate_ditto_video(audio_path)
                    if video_result: final_output = video_result

                if mode == "Voice Only":
                    yield history, final_output, None
                else:
                     if video_result:
                        yield history, None, final_output
                     else:
                        yield history, audio_path, None

    except Exception as e:
        history[-1][1] += f"\n[Error: {str(e)}]"
        yield history, None, None

# =========================================================================
# 🖥️ GRADIO UI (Layout Updated)
# =========================================================================
with gr.Blocks(title="Integrated Digital Twin", theme=gr.themes.Soft()) as demo:
    # Header
    gr.Markdown("# 🧬 Integrated Digital Twin Interface")
    gr.Markdown("Chat with your AI avatar. Switch modes to enable real-time video generation.")
    
    # Main Layout container
    with gr.Row():
        # ----- LEFT COLUMN (MAIN CHAT) -----
        with gr.Column(scale=4):
            # Mode selection near the chat
            mode_select = gr.Radio(
                ["Voice Only", "Video (Ditto)"], 
                label="Output Mode", 
                value="Voice Only", 
                info="Video mode will take several seconds to process each response."
            )
            
            # The main chat window
            chatbot = gr.Chatbot(label="Conversation", height=500)
            
            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    label="Type message...", 
                    placeholder="Ask me anything...",
                    scale=4,
                    autofocus=True
                )
                submit = gr.Button("Send", scale=1, variant="primary")

        # ----- RIGHT COLUMN (AVATAR VIEW - Top Right) -----
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Avatar View")
            
            # Video Player (Top Right)
            # Setting interactive=False prevents the user from uploading their own video
            video_player = gr.Video(
                label="Visual Output", 
                autoplay=True, 
                visible=True, 
                interactive=False,
                height=300 # Fixed height to keep it compact
            )
            
            # Audio Player (Below Video)
            audio_player = gr.Audio(
                label="Voice Output", 
                autoplay=True, 
                visible=True, 
                interactive=False
            )

    # Helper to clear text box
    def clear_msg(): return ""

    # Event Wiring
    # Note: We update both players. The pipeline decides which one gets data.
    msg.submit(chat_pipeline, [msg, mode_select, chatbot], [chatbot, audio_player, video_player]) \
       .then(clear_msg, None, msg)
       
    submit.click(chat_pipeline, [msg, mode_select, chatbot], [chatbot, audio_player, video_player]) \
          .then(clear_msg, None, msg)

if __name__ == "__main__":
    # Launch with public link
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860, allowed_paths=[PROJECT_ROOT])