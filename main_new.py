
import os
import platform

# Add the current directory to the DLL search path on Windows
if platform.system() == "Windows":
    os.add_dll_directory(os.path.dirname(os.path.abspath(__file__)))

import torch
import fastapi
import uvicorn
import base64
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import WebSocket, WebSocketDisconnect, Request
from fastapi.concurrency import run_in_threadpool
from concurrent.futures import ProcessPoolExecutor
import asyncio
import json
import logging
import nltk
import warnings
import traceback
from contextlib import asynccontextmanager
import httpx
from llama_cpp import Llama
from faster_whisper import WhisperModel

import io
import time
import numpy as np
import soundfile as sf
import re
from num2words import num2words

import webrtcvad
import psutil

from dotenv import load_dotenv
from kokoro import KPipeline
from kittentts import KittenTTS

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")


# --- Setup ---
ROOT = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)

def get_system_specs():
    """Gets system hardware specifications."""
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    cpu_cores = psutil.cpu_count(logical=False)
    cpu_threads = psutil.cpu_count(logical=True)
    
    gpu_info = "N/A"
    if torch.cuda.is_available():
        gpu_info = torch.cuda.get_device_name(0)

    return {
        "ram_gb": f"{ram_gb:.2f} GB",
        "cpu_info": f"{cpu_cores} Cores, {cpu_threads} Threads",
        "gpu_info": gpu_info
    }

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    print("--- Loading all AI models ---")
    load_dotenv()
    app.state.main_event_loop = asyncio.get_running_loop()

    # --- Hardware Detection & State Initialization ---
    app.state.system_specs = get_system_specs()
    app.state.model_info = {"stt": "", "llm": "", "tts": ""}
    print(f"System Specs: {app.state.system_specs}")

    # --- LLM Initialization ---
    llm_model_name = os.getenv("LLM_MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
    llm_model_path = os.path.join(ROOT, "models", llm_model_name)
    # print(f"Loading LLM from: {llm_model_path}")
    app.state.llm = Llama(model_path=llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    app.state.model_info["llm"] = llm_model_name
    # print("LLM initialized.")

    tts_engine_choice = os.getenv("TTS_ENGINE", "kokoro").lower().strip()
    app.state.tts_engine = tts_engine_choice
    app.state.model_info["tts"] = tts_engine_choice.capitalize()
    print(f"Selected TTS backend: {tts_engine_choice}")

    # Read Kitten TTS voice if engine is kitten
    if app.state.tts_engine == "kitten":
        kitten_tts_voice = os.getenv("KITTEN_TTS_VOICE", "expr-voice-2-f") # Default to expr-voice-2-f
        app.state.kitten_tts_voice = kitten_tts_voice
        print(f"Selected Kitten TTS voice: {kitten_tts_voice}")

    # --- TTS Engine Initialization ---
    if app.state.tts_engine == "kokoro":
        app.state.kokoro_tts = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        print("Kokoro TTS engine initialized.")
        kokoro_tts_voice = os.getenv("KOKORO_TTS_VOICE", "Bella") # Default to Bella
        app.state.kokoro_tts_voice = kokoro_tts_voice
        print(f"Selected Kokoro TTS voice: {kokoro_tts_voice}")
    elif app.state.tts_engine == "kitten":
        app.state.kittentts_model = KittenTTS()
        print("KittenTTS engine initialized.")
    else:
        raise ValueError(f"Unknown TTS_ENGINE: {app.state.tts_engine}. Choose 'kokoro' or 'kitten'.")

    stt_model_name = os.getenv("WHISPER_MODEL_NAME", "tiny.en") # Default to tiny.en
    app.state.whisper_model = WhisperModel(stt_model_name, device="cpu", compute_type="int8")
    app.state.model_info["stt"] = f"Faster-Whisper {stt_model_name}"
    print("Faster Whisper STT model initialized.")
    print("--- All models loaded successfully ---")

    # Initialize ProcessPoolExecutor for TTS synthesis
    app.state.tts_executor = ProcessPoolExecutor(max_workers=2) # Use 2 cores for TTS synthesis

    nltk.download('punkt')
    yield
    print("--- Application shutting down ---")
    app.state.tts_executor.shutdown(wait=True) # Shutdown the TTS executor


app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

async def get_llm_response(app_state, text: str, history: list):
    full_prompt = ""
    for user_msg, assistant_resp in history:
        full_prompt += f'''<|user|>
{user_msg}<|end|>
<|assistant|>
{assistant_resp}<|end|>
'''
    full_prompt += f'''<|user|>
{text}<|end|>
<|assistant|>'''

    # Run the LLM inference in a background thread
    llm_stream = await run_in_threadpool(
        app_state.llm,
        prompt=full_prompt,
        max_tokens=150,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.1,
        stream=True
    )

    # Process the streaming response
    response_text = ""
    for output in llm_stream:
        response_text += output['choices'][0]['text']
        
    return response_text.strip()

def normalize_text(text):
    def decimal_to_words(match):
        number_str = match.group(0)

        if '.' in number_str:
            parts = number_str.split('.')
            integer_part_words = num2words(int(parts[0]))
            decimal_part_words = ' '.join(num2words(int(digit)) for digit in parts[1])
            return f"{integer_part_words} point {decimal_part_words}"
        else:
            return num2words(int(number_str))
    text = re.sub(r'\d+(\.\d+)?', decimal_to_words, text)
    return text

# Global/static variable to hold the TTS model in each worker process
_tts_model_cache = {}

def _perform_tts_synthesis_sync(
    sentence_text: str,
    tts_engine_choice: str,
    voice: str # This is the voice parameter (e.g., "Bella", "expr-voice-2-f")
) -> bytes:
    all_pcm_audio = bytearray()
    try:
        # Initialize model if not already in cache for this process
        if tts_engine_choice not in _tts_model_cache:
            if tts_engine_choice == "kokoro":
                _tts_model_cache[tts_engine_choice] = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
            elif tts_engine_choice == "kitten":
                _tts_model_cache[tts_engine_choice] = KittenTTS()
            else:
                raise ValueError(f"Unknown TTS_ENGINE: {tts_engine_choice}. Choose 'kokoro' or 'kitten'.")

        active_tts_model_instance = _tts_model_cache[tts_engine_choice]

        if tts_engine_choice == "kokoro":
            all_generated_audio_chunks = []
            for i, (gs, ps, audio) in enumerate(active_tts_model_instance(sentence_text, voice=voice)):
                if audio is not None and audio.numel() > 0:
                    all_generated_audio_chunks.append(audio)
            if all_generated_audio_chunks:
                generated_audio = torch.cat(all_generated_audio_chunks)
            else:
                generated_audio = None
            if generated_audio is not None:
                if isinstance(generated_audio, torch.Tensor):
                    audio_np = generated_audio.detach().cpu().numpy()
                    scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
                    all_pcm_audio.extend(scaled_audio_np.tobytes())
        
        elif tts_engine_choice == "kitten":
            audio_np = active_tts_model_instance.generate(sentence_text, voice=voice)
            audio_np = audio_np.detach() # Ensure it's detached
            scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
            all_pcm_audio.extend(scaled_audio_np.tobytes())
        print(f"Synthesized audio size: {len(all_pcm_audio)} bytes")
        return bytes(all_pcm_audio)
    except Exception as e:
        print(f"Error in _perform_tts_synthesis_sync: {e}")
        traceback.print_exc()
        return b""

async def stream_tts_and_synthesize(websocket: WebSocket, text: str):
    print(f"stream_tts_and_synthesize called with text: '{text}'")
    sample_rate = 24000
    channels = 1
    ffmpeg_command = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "-", "-c:a", "libopus", "-map", "0:a", "-f", "webm", "-"]
    print(f"Starting ffmpeg process: {' '.join(ffmpeg_command)}")
    process = await asyncio.create_subprocess_exec(*ffmpeg_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    async def feed_ffmpeg():
        try:
            sentence_buffer = ""
            processing_buffer = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
            sentences = re.split(r'(?<=[.?!,])\s*', processing_buffer)
            
            tts_futures = []
            sentences_to_process = []

            if len(sentences) > 1:
                sentences_to_process.extend(sentences[:-1])
                sentence_buffer = sentences[-1]
            else:
                sentences_to_process.extend(sentences)

            for sentence in sentences_to_process:
                if not sentence.strip():
                    continue
                sentence_with_decimals = sentence.replace('<DECIMAL>', '.')
                normalized_sentence = normalize_text(sentence_with_decimals)
                processed_sentence = re.sub(r'<\|user\|>|<\|end\|>|<\|assistant\|>|[\*#`_]', '', normalized_sentence).lstrip()
                
                # Determine which TTS model and voice to pass
                current_tts_voice = None
                if websocket.app.state.tts_engine == "kokoro":
                    current_tts_voice = websocket.app.state.kokoro_tts_voice
                elif websocket.app.state.tts_engine == "kitten":
                    current_tts_voice = websocket.app.state.kitten_tts_voice
                
                # Submit to ProcessPoolExecutor
                future = websocket.app.state.tts_executor.submit(
                    _perform_tts_synthesis_sync,
                    processed_sentence,
                    websocket.app.state.tts_engine,
                    current_tts_voice
                )
                tts_futures.append(future)

            # If there's a remaining sentence, add it to futures
            if sentence_buffer.strip():
                sentence_with_decimals = sentence_buffer.replace('<DECIMAL>', '.')
                normalized_sentence = normalize_text(sentence_with_decimals)
                processed_sentence = re.sub(r'[*#`_]', '', normalized_sentence).lstrip()
                
                # Determine which TTS model and voice to pass
                current_tts_voice = None
                if websocket.app.state.tts_engine == "kokoro":
                    current_tts_voice = websocket.app.state.kokoro_tts_voice
                elif websocket.app.state.tts_engine == "kitten":
                    current_tts_voice = websocket.app.state.kitten_tts_voice

                future = websocket.app.state.tts_executor.submit(
                    _perform_tts_synthesis_sync,
                    processed_sentence,
                    websocket.app.state.tts_engine,
                    current_tts_voice
                )
                tts_futures.append(future)

            # Process synthesized audio in order as it becomes available
            for future in tts_futures:
                sentence_audio = await asyncio.wrap_future(future) # Await result from process pool
                if sentence_audio:
                    if process.stdin and not process.stdin.is_closing():
                        process.stdin.write(sentence_audio)
                        await process.stdin.drain()
                    else:
                        break

        except Exception as e:
            print(f"Error in feed_ffmpeg: {e}")
            traceback.print_exc()
        finally:
            if process.stdin and not process.stdin.is_closing():
                print("Closing ffmpeg stdin.")
                process.stdin.close()

    async def stream_to_client():
        while True:
            try:
                chunk = await process.stdout.read(4096)
                if not chunk:
                    break
                await websocket.send_bytes(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            except Exception as e:
                print(f"--- UNEXPECTED ERROR IN STREAM_TO_CLIENT ---")
                print(f"Error: {e}")
                traceback.print_exc()
                break
        print("Finished streaming audio to client.")

    feeder_task = asyncio.create_task(feed_ffmpeg())
    streamer_task = asyncio.create_task(stream_to_client())
    await asyncio.gather(feeder_task, streamer_task)
    print("Feeder and streamer tasks completed.")

    await process.wait()
    stdout, stderr = await process.communicate()
    if stderr:
        pass

    try:
        await websocket.send_text("EOS")
        print("EOS sent.")
    except Exception as e:
        print(f"Could not send EOS: {e}")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")

    conversation_history = [] # Initialize conversation history

    await websocket.send_json({
        "type": "session_info",
        "data": {
            "system_specs": websocket.app.state.system_specs,
            "model_info": websocket.app.state.model_info,
        }
    })

    vad = webrtcvad.Vad(1)
    VAD_SAMPLE_RATE = 16000
    VAD_FRAME_MS = 30
    VAD_FRAME_SAMPLES = int(VAD_SAMPLE_RATE * (VAD_FRAME_MS / 1000.0))
    raw_audio_buffer = bytearray()
    vad_audio_buffer = bytearray()
    speech_buffer = bytearray()
    is_speaking = False
    silence_frames_count = 0
    SILENCE_THRESHOLD_MS = 750
    silence_threshold_frames = int(SILENCE_THRESHOLD_MS / VAD_FRAME_MS)

    try:
        while True:
            message = await websocket.receive()
            if 'bytes' in message:
                raw_audio_buffer.extend(message["bytes"])
                audio_float32 = np.frombuffer(raw_audio_buffer, dtype=np.float32)
                raw_audio_buffer = bytearray()
                audio_int16 = (audio_float32 * 32767).astype(np.int16)
                vad_audio_buffer.extend(audio_int16.tobytes())

                while len(vad_audio_buffer) >= VAD_FRAME_SAMPLES * 2:
                    frame_bytes = vad_audio_buffer[:VAD_FRAME_SAMPLES * 2]
                    del vad_audio_buffer[:VAD_FRAME_SAMPLES * 2]
                    is_speech = vad.is_speech(frame_bytes, VAD_SAMPLE_RATE)
                    if is_speech:
                        if not is_speaking:
                            print("Speech started.")
                            is_speaking = True
                            speech_buffer = bytearray() # Clear buffer at the start of a new speech segment
                        speech_buffer.extend(frame_bytes)
                        silence_frames_count = 0
                    elif is_speaking:
                        silence_frames_count += 1
                        if silence_frames_count > silence_threshold_frames:
                            print("Speech ended due to silence.")
                            is_speaking = False
                            if speech_buffer:
                                speech_np_int16 = np.frombuffer(speech_buffer, dtype=np.int16)
                                speech_buffer = bytearray() # Clear buffer after processing
                                rms_energy = np.sqrt(np.mean(speech_np_int16.astype(np.float64)**2))
                                ENERGY_THRESHOLD = 999
                                print(f"Utterance RMS energy: {rms_energy:.2f}")
                                if rms_energy > ENERGY_THRESHOLD:
                                    speech_np_float32 = speech_np_int16.astype(np.float32) / 32768.0
                                    start_time = time.time()
                                    segments, _ = await run_in_threadpool(
                                        websocket.app.state.whisper_model.transcribe,
                                        speech_np_float32,
                                        language="en",
                                        beam_size=3
                                    )
                                    end_time = time.time()
                                    logging.info(f"STT Transcription Time: {end_time - start_time:.4f} seconds")
                                    transcription = "".join(s.text for s in segments)
                                    transcription = transcription.strip()
                                    if transcription and not transcription.lower().startswith(("[blank audio]", "[blank_audio]")):
                                        print(f"Final Transcription: '{transcription}'")
                                        await websocket.send_json({"type": "transcription", "text": transcription})
                                        response_text = await get_llm_response(websocket.app.state, transcription, conversation_history)
                                        conversation_history.append((transcription, response_text))
                                        asyncio.create_task(stream_tts_and_synthesize(websocket, response_text))
                                    else:
                                        print("Transcription was blank or discarded.")
                                else:
                                    print(f"Utterance discarded, energy below threshold.")
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception as send_e:
            print(f"Failed to send error message to client: {send_e}")

# --- Main Entry Point ---
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8002, ws_ping_interval=30, ws_ping_timeout=120)
