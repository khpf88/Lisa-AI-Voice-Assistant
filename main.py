
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
from abc import ABC, abstractmethod
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
    app.state.llms = {}
    # General model for conversation
    general_llm_model_name = os.getenv("GENERAL_LLM_MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
    general_llm_model_path = os.path.join(ROOT, "models", general_llm_model_name)
    app.state.llms["general"] = Llama(model_path=general_llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print(f"Loaded general LLM: {general_llm_model_name}")

    # Coding model for code-related prompts
    coding_llm_model_name = os.getenv("CODING_LLM_MODEL_FILENAME", "qwen2-0_5b-instruct-q8_0.gguf")
    coding_llm_model_path = os.path.join(ROOT, "models", coding_llm_model_name)
    app.state.llms["coding"] = Llama(model_path=coding_llm_model_path, n_ctx=2048, n_gpu_layers=-1, verbose=False)
    print(f"Loaded coding LLM: {coding_llm_model_name}")

    app.state.model_info["llm"] = f"General: {general_llm_model_name}, Coding: {coding_llm_model_name}"

    # --- TTS Configuration (models are loaded in worker processes) ---
    tts_engine_choice = os.getenv("TTS_ENGINE", "kokoro").lower().strip()
    if tts_engine_choice not in ["kokoro", "kitten"]:
        raise ValueError(f"Unknown TTS_ENGINE: {tts_engine_choice}. Choose 'kokoro' or 'kitten'.")
    app.state.tts_engine = tts_engine_choice
    app.state.model_info["tts"] = tts_engine_choice.capitalize()
    app.state.kokoro_tts_voice = os.getenv("KOKORO_TTS_VOICE", "Bella")
    app.state.kitten_tts_voice = os.getenv("KITTEN_TTS_VOICE", "expr-voice-2-f")
    print(f"Selected TTS backend: {app.state.tts_engine}")
    print(f" - Kokoro voice: {app.state.kokoro_tts_voice}")
    print(f" - Kitten voice: {app.state.kitten_tts_voice}")

    stt_model_name = os.getenv("WHISPER_MODEL_NAME", "tiny.en") # Default to tiny.en
    app.state.whisper_model = WhisperModel(stt_model_name, device="cpu", compute_type="int8")
    app.state.model_info["stt"] = f"Faster-Whisper {stt_model_name}"
    print("Faster Whisper STT model initialized.")
    print("--- All models loaded successfully ---")

    # --- Initialize ProcessPoolExecutor with a declarative, resource-aware policy ---
    engine_class = None
    if app.state.tts_engine == "kokoro":
        engine_class = KokoroEngine
    elif app.state.tts_engine == "kitten":
        engine_class = KittenEngine

    if not engine_class:
        raise ValueError(f"Could not find a profile for TTS engine: {app.state.tts_engine}")

    # Read the declared requirements from the engine's profile.
    est_ram_per_worker, max_cores_for_engine = engine_class.RESOURCE_PROFILE

    # Get actual system resources.
    cpu_cores_total = os.cpu_count() or 1
    available_ram_gb = psutil.virtual_memory().available / (1024**3)

    # --- Calculate optimal workers based on multiple constraints ---
    # 1. Constraint by available RAM
    # (Add a small 0.25GB buffer to be safe)
    ram_workers = int((available_ram_gb - 0.25) // est_ram_per_worker)

    # 2. Constraint by total CPU cores (default policy is half)
    cpu_workers = cpu_cores_total // 2

    # 3. The final number of workers is the minimum of all constraints.
    tts_workers = max(1, min(ram_workers, cpu_workers, max_cores_for_engine))
    if tts_workers == 1:
        tts_workers = 2

    print(f"INFO: Engine Profile ({app.state.tts_engine}): Needs ~{est_ram_per_worker}GB RAM, can use up to {max_cores_for_engine} cores.")
    print(f"INFO: System State: {available_ram_gb:.2f}GB RAM available, {cpu_cores_total} CPU cores total.")
    print(f"INFO: Calculated worker constraints: RAM limit={ram_workers}, CPU limit={cpu_workers}, Engine limit={max_cores_for_engine}.")
    print(f"INFO: Final worker count set to {tts_workers}.")

    app.state.tts_executor = ProcessPoolExecutor(
        max_workers=tts_workers,
        initializer=init_worker,
        initargs=(app.state.tts_engine,)
    )

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

def route_prompt(prompt: str) -> str:
    """Routes a prompt to the appropriate LLM."""
    coding_keywords = ["code", "python", "javascript", "java", "c++", "rust", "go", "typescript"]
    if any(keyword in prompt.lower() for keyword in coding_keywords):
        return "coding"
    return "general"

async def summarize_text(app_state, text: str) -> str:
    """Summarizes a long text using a smaller LLM."""
    summarization_prompt = f"Strictly summarize the following text in a conversational way, using no more than 50 words. Do not exceed this word count: {text}"
    selected_llm = app_state.llms["coding"] # Use the smaller model for summarization
    print("Summarizing long response...")

    llm_stream = await run_in_threadpool(
        selected_llm,
        prompt=summarization_prompt,
        max_tokens=500,
        temperature=0.5,
        stream=False # No streaming for summarization
    )

    summary = llm_stream['choices'][0]['text'].strip()
    return summary

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

    # Route the prompt to the appropriate LLM
    model_choice = route_prompt(text)
    selected_llm = app_state.llms[model_choice]
    print(f"Routing prompt to '{model_choice}' LLM.")

    # Run the LLM inference in a background thread
    llm_stream = await run_in_threadpool(
        selected_llm,
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
        
    response_text = response_text.strip()
    summarized = False
    if len(response_text) > 500:
        response_text = await summarize_text(app_state, response_text)
        summarized = True

    return response_text, model_choice, summarized

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

# --- TTS Engine Abstraction ---

class TTSEngine(ABC):
    """Abstract base class for a TTS engine."""

    # --- Resource Profile Declaration ---
    # Each engine should override this profile to declare its requirements.
    # Tuple format: (estimated_ram_gb_per_worker, max_cpu_cores_to_use)
    RESOURCE_PROFILE = (1.0, 4) # A sensible default for a generic engine.

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> bytes:
        """
        Synthesizes text to raw PCM audio bytes (s16le, 24000Hz, mono).
        Returns empty bytes if synthesis fails.
        """
        pass

class KokoroEngine(TTSEngine):
    """Wrapper for the memory-intensive Kokoro TTS engine."""
    # Kokoro is heavy: Needs ~2GB RAM, shouldn't use more than 2 cores total to be safe.
    RESOURCE_PROFILE = (2.0, 2)

    def __init__(self):
        print("Initializing Kokoro TTS engine in worker process...")
        self.model = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        print("Kokoro TTS engine initialized.")

    def synthesize(self, text: str, voice: str) -> bytes:
        all_pcm_audio = bytearray()
        try:
            all_generated_audio_chunks = []
            for i, (gs, ps, audio) in enumerate(self.model(text, voice=voice)):
                if audio is not None and audio.numel() > 0:
                    all_generated_audio_chunks.append(audio)

            if not all_generated_audio_chunks:
                print(f"Warning: Kokoro TTS produced no audio for text: '{text}' with voice: '{voice}'")
                return b""

            generated_audio = torch.cat(all_generated_audio_chunks)
            if generated_audio is not None and isinstance(generated_audio, torch.Tensor):
                audio_np = generated_audio.detach().cpu().numpy()
                scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
                all_pcm_audio.extend(scaled_audio_np.tobytes())

            return bytes(all_pcm_audio)
        except Exception as e:
            print(f"Error during Kokoro TTS synthesis: {e}")
            traceback.print_exc()
            return b""

class KittenEngine(TTSEngine):
    """Wrapper for the lightweight KittenTTS engine."""
    # Kitten is light: Needs ~0.5GB RAM, can scale up to 8 cores if available.
    RESOURCE_PROFILE = (0.5, 8)

    def __init__(self):
        print("Initializing KittenTTS engine in worker process...")
        self.model = KittenTTS()
        print("KittenTTS engine initialized.")

    def synthesize(self, text: str, voice: str) -> bytes:
        all_pcm_audio = bytearray()
        try:
            audio_np = self.model.generate(text, voice=voice)
            scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
            all_pcm_audio.extend(scaled_audio_np.tobytes())
            return bytes(all_pcm_audio)
        except Exception as e:
            print(f"Error during KittenTTS synthesis: {e}")
            traceback.print_exc()
            return b""

# --- Process Pool Worker Initialization ---

# Global variable to hold the loaded TTS engine in a worker process
_worker_tts_engine = None

def init_worker(engine_choice: str):
    """
    Initializer for each worker process. Loads the specified TTS model.
    This function runs once when a worker process is created.
    """
    global _worker_tts_engine
    if _worker_tts_engine is None:
        if engine_choice == "kokoro":
            _worker_tts_engine = KokoroEngine()
        elif engine_choice == "kitten":
            _worker_tts_engine = KittenEngine()

def _perform_tts_synthesis_sync(sentence_text: str, voice: str) -> bytes:
    """
    Uses the pre-loaded TTS engine in the worker to synthesize audio.
    """
    global _worker_tts_engine
    try:
        if _worker_tts_engine is None:
            print("Error: TTS engine not initialized in worker. This should not happen.")
            return b""

        pcm_audio = _worker_tts_engine.synthesize(sentence_text, voice)
        print(f"Synthesized audio size: {len(pcm_audio)} bytes")
        return pcm_audio
    except Exception as e:
        print(f"Fatal error in _perform_tts_synthesis_sync: {e}")
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

    await websocket.send_json({"type": "calibration_start"})

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

    # Dynamic VAD Thresholding
    is_calibrating = True
    calibration_frames = 0
    CALIBRATION_DURATION_MS = 2000  # 2 seconds
    calibration_frame_count = int(CALIBRATION_DURATION_MS / VAD_FRAME_MS)
    ambient_energy_values = []
    ENERGY_THRESHOLD = 999  # Default value

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

                    if is_calibrating:
                        frame_int16 = np.frombuffer(frame_bytes, dtype=np.int16)
                        rms_energy = np.sqrt(np.mean(frame_int16.astype(np.float64)**2))
                        ambient_energy_values.append(rms_energy)
                        calibration_frames += 1
                        if calibration_frames >= calibration_frame_count:
                            if ambient_energy_values:
                                avg_ambient_energy = np.mean(ambient_energy_values)
                                ENERGY_THRESHOLD = avg_ambient_energy * 5  # Set threshold to 5x ambient noise
                                print(f"Ambient energy calculated: {avg_ambient_energy:.2f}. New energy threshold: {ENERGY_THRESHOLD:.2f}")
                            else:
                                print("No audio received during calibration. Using default energy threshold.")
                            is_calibrating = False
                            await websocket.send_json({"type": "calibration_complete"})
                        continue

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
                                        response_text, model_choice, summarized = await get_llm_response(websocket.app.state, transcription, conversation_history)
                                        conversation_history.append((transcription, response_text))
                                        await websocket.send_json({"type": "model_update", "model": model_choice})
                                        if summarized:
                                            await websocket.send_json({"type": "summarization_info"})
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
