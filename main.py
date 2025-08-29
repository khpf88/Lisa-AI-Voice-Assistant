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
import asyncio
import json
import logging
import nltk
import warnings
import traceback
from contextlib import asynccontextmanager

from faster_whisper import WhisperModel
import io
import time
import numpy as np
import soundfile as sf
import re
from num2words import num2words

import webrtcvad
import psutil

import google.generativeai as genai
from dotenv import load_dotenv
from kokoro import KPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from llama_cpp import Llama
from threading import Thread
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
    app.state.accumulated_tokens = 0
    app.state.model_info = {"stt": "", "llm": "", "tts": ""}
    print(f"System Specs: {app.state.system_specs}")


    # --- Model Selection Logic ---
    ram_gb_str = app.state.system_specs['ram_gb'].replace(' GB', '')
    ram_gb = float(ram_gb_str)
    ram_threshold_high = float(os.getenv("LLM_RAM_THRESHOLD_HIGH", "10"))
    ram_threshold_low = float(os.getenv("LLM_RAM_THRESHOLD_LOW", "4"))

    if ram_gb > ram_threshold_high:
        llm_choice = "LlamaCpp"
    elif ram_threshold_low < ram_gb <= ram_threshold_high:
        llm_choice = "Transformers"
    else:
        llm_choice = "Gemini"
    app.state.llm_choice = llm_choice
    print(f"Selected LLM backend: {llm_choice}")

    tts_engine_choice = os.getenv("TTS_ENGINE", "kokoro").lower().strip()
    app.state.tts_engine = tts_engine_choice
    app.state.model_info["tts"] = tts_engine_choice.capitalize()
    print(f"Selected TTS backend: {tts_engine_choice}")


    # --- LLM Loading ---
    app.state.llm_client = None
    system_prompt = "You are Lisa, a conversational AI. Your responses must be concise, containing only one or two sentences. Do not use markdown, just plain text."

    if app.state.llm_choice == "Gemini":
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            app.state.llm_client = genai.GenerativeModel('models/gemini-1.5-flash-latest', system_instruction=system_prompt)
            app.state.model_info["llm"] = "Gemini (gemini-1.5-flash-latest)"
            print("Gemini model loaded.")
        else:
            print("GEMINI_API_KEY not found. Gemini client not initialized.")

    elif app.state.llm_choice == "LlamaCpp":
        quantization = os.getenv("LLAMA_QUANTIZATION", "Q4_K_M")
        model_filename = f"tinyllama-1.1b-chat-v1.0.{quantization}.gguf"
        LLAMA_CPP_MODEL_PATH = os.path.join(ROOT, "models", model_filename)
        if os.path.exists(LLAMA_CPP_MODEL_PATH):
            try:
                app.state.llm_client = Llama(model_path=LLAMA_CPP_MODEL_PATH, chat_format="zephyr", n_ctx=2048, n_gpu_layers=-1 if torch.cuda.is_available() else 0)
                app.state.model_info["llm"] = f"Llama.cpp ({quantization})"
                print(f"Llama.cpp model loaded from {LLAMA_CPP_MODEL_PATH}")
            except Exception as e:
                print(f"Error loading Llama.cpp model: {e}")
        else:
            print(f"Llama.cpp model not found at {LLAMA_CPP_MODEL_PATH}.")

    elif app.state.llm_choice == "Transformers":
        transformers_model_size = os.getenv("TRANSFORMERS_MODEL_SIZE", "350") # Default to 350m
        transformers_model_name = f"lfm2-{transformers_model_size}m-chat-hf"
        TRANSFORMERS_MODEL_PATH = os.path.join(ROOT, "models", transformers_model_name)
        if os.path.exists(TRANSFORMERS_MODEL_PATH):
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_PATH)
                model = AutoModelForCausalLM.from_pretrained(TRANSFORMERS_MODEL_PATH).to(device)
                app.state.llm_client = {"tokenizer": tokenizer, "model": model, "device": device}
                app.state.model_info["llm"] = f"Transformers ({transformers_model_name})"
                print(f"Transformers model loaded from {TRANSFORMERS_MODEL_PATH} on {device}.")
            except Exception as e:
                print(f"Error loading Transformers model: {e}")
        else:
            print(f"Transformers model not found at {TRANSFORMERS_MODEL_PATH}.")

    if not app.state.llm_client:
        print("Warning: No LLM client was successfully initialized. Falling back to Gemini if possible.")
        if not app.state.llm_choice == "Gemini":
            GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
            if GEMINI_API_KEY:
                genai.configure(api_key=GEMINI_API_KEY)
                app.state.llm_client = genai.GenerativeModel('models/gemini-1.5-flash-latest', system_instruction=system_prompt)
                app.state.llm_choice = "Gemini"
                app.state.model_info["llm"] = "Gemini (gemini-1.5-flash-latest)"
                print("Fell back to and loaded Gemini model.")
            else:
                raise ValueError("No LLM clients could be initialized. Please provide at least one API key or a local model.")

    # --- TTS Engine Initialization ---
    if app.state.tts_engine == "kokoro":
        app.state.kokoro_tts = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        print("Kokoro TTS engine initialized.")
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

    nltk.download('punkt')
    yield
    print("--- Application shutting down ---")


app = fastapi.FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

class TTSRequest(BaseModel):
    text: str

async def generate(text: str, request: Request):
    print(f"generate called with text: '{text}' using {request.app.state.llm_choice}")
    
    system_prompt = "You are Lisa, a conversational AI. Your responses must be concise, containing only one or two sentences. Do not use markdown, just plain text."
    llm_client = request.app.state.llm_client
    llm_choice = request.app.state.llm_choice
    full_response_text = ""

    if not llm_client:
        print("No LLM client available to generate response.")
        yield "I'm sorry, I don't have an active language model to respond."
        return

    try:
        if llm_choice == "Gemini":
            prompt_tokens = await llm_client.count_tokens_async(text)
            request.app.state.accumulated_tokens += prompt_tokens.total_tokens
            start_time = time.time()
            response_stream = await llm_client.generate_content_async(
                text,
                stream=True,
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            first_chunk = True
            async for chunk in response_stream:
                if first_chunk:
                    end_time = time.time()
                    logging.info(f"Gemini LLM Time to First Chunk: {end_time - start_time:.4f} seconds")
                    first_chunk = False
                if chunk.text:
                    full_response_text += chunk.text
                    yield chunk.text
            response_tokens = await llm_client.count_tokens_async(full_response_text)
            request.app.state.accumulated_tokens += response_tokens.total_tokens

        elif llm_choice == "LlamaCpp":
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
            prompt_tokens = llm_client.tokenize(text.encode('utf-8'))
            request.app.state.accumulated_tokens += len(prompt_tokens)
            start_time = time.time()
            response_stream = llm_client.create_chat_completion(messages=messages, stream=True)
            first_chunk = True
            for chunk in response_stream:
                if first_chunk:
                    end_time = time.time()
                    logging.info(f"LlamaCpp LLM Time to First Chunk: {end_time - start_time:.4f} seconds")
                    first_chunk = False
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    content = delta['content']
                    full_response_text += content
                    yield content
            response_tokens = llm_client.tokenize(full_response_text.encode('utf-8'))
            request.app.state.accumulated_tokens += len(response_tokens)

        elif llm_choice == "Transformers":
            tokenizer = llm_client["tokenizer"]
            model = llm_client["model"]
            device = llm_client["device"]
            prompt_tokens = tokenizer.encode(text)
            request.app.state.accumulated_tokens += len(prompt_tokens)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            generation_kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            start_time = time.time()
            first_chunk = True
            for new_text in streamer:
                if first_chunk:
                    end_time = time.time()
                    logging.info(f"Transformers LLM Time to First Chunk: {end_time - start_time:.4f} seconds")
                    first_chunk = False
                full_response_text += new_text
                yield new_text
            thread.join()
            response_tokens = tokenizer.encode(full_response_text)
            request.app.state.accumulated_tokens += len(response_tokens)

    except Exception as e:
        print(f"Error generating content from {llm_choice} API: {e}")
        traceback.print_exc()
        yield "I'm sorry, I couldn't generate a response at this time."

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

async def stream_llm_and_synthesize(websocket: WebSocket, text: str):
    print(f"stream_llm_and_synthesize called with text: '{text}'")
    sample_rate = 24000
    channels = 1
    ffmpeg_command = ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "-i", "-", "-c:a", "libopus", "-map", "0:a", "-f", "webm", "-"]
    print(f"Starting ffmpeg process: {' '.join(ffmpeg_command)}")
    process = await asyncio.create_subprocess_exec(*ffmpeg_command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

    async def feed_ffmpeg():
        def synthesize_sentence_sync(sentence_text: str) -> bytes:
            all_pcm_audio = bytearray()
            try:
                tts_engine_choice = websocket.app.state.tts_engine
                if tts_engine_choice == "kokoro":
                    all_generated_audio_chunks = []
                    for i, (gs, ps, audio) in enumerate(websocket.app.state.kokoro_tts(sentence_text, voice="af_heart")):
                        if audio is not None and audio.numel() > 0:
                            all_generated_audio_chunks.append(audio)
                    if all_generated_audio_chunks:
                        generated_audio = torch.cat(all_generated_audio_chunks)
                    else:
                        generated_audio = None
                    if generated_audio is not None:
                        if isinstance(generated_audio, torch.Tensor):
                            audio_np = generated_audio.cpu().numpy()
                            scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
                            all_pcm_audio.extend(scaled_audio_np.tobytes())
                
                elif tts_engine_choice == "kitten":
                    kittentts_model = websocket.app.state.kittentts_model
                    # Using a default voice for now. KittenTTS supports multiple voices.
                    audio_np = kittentts_model.generate(sentence_text, voice='expr-voice-2-f')
                    # KittenTTS returns audio as a NumPy array (float32). Convert to int16 bytes.
                    scaled_audio_np = (audio_np * 32767.0).astype(np.int16)
                    all_pcm_audio.extend(scaled_audio_np.tobytes())
                print(f"Synthesized audio size: {len(all_pcm_audio)} bytes")
                return bytes(all_pcm_audio)
            except Exception as e:
                print(f"Error in synthesize_sentence_sync: {e}")
                traceback.print_exc()
                return b""

        try:
            sentence_buffer = ""
            async for text_chunk in generate(text, websocket):
                sentence_buffer += text_chunk
                processing_buffer = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', sentence_buffer)
                sentences = re.split(r'(?<=[.?!,])\s*', processing_buffer)
                if len(sentences) > 1:
                    for sentence in sentences[:-1]:
                        if not sentence.strip():
                            continue
                        sentence_with_decimals = sentence.replace('<DECIMAL>', '.')
                        normalized_sentence = normalize_text(sentence_with_decimals)
                        processed_sentence = re.sub(r'[\*#`_]', '', normalized_sentence).lstrip()
                        print(f"Synthesizing sentence: '{processed_sentence}'")
                        sentence_audio = await run_in_threadpool(synthesize_sentence_sync, processed_sentence)
                        if sentence_audio:
                            if process.stdin and not process.stdin.is_closing():
                                process.stdin.write(sentence_audio)
                                await process.stdin.drain()
                            else:
                                break
                    sentence_buffer = sentences[-1]
            if sentence_buffer.strip():
                sentence_with_decimals = sentence_buffer.replace('<DECIMAL>', '.')
                normalized_sentence = normalize_text(sentence_with_decimals)
                processed_sentence = re.sub(r'[\*#`_]', '', normalized_sentence).lstrip()
                print(f"Synthesizing remaining sentence: '{processed_sentence}'")
                sentence_audio = await run_in_threadpool(synthesize_sentence_sync, processed_sentence)
                if sentence_audio and process.stdin and not process.stdin.is_closing():
                    process.stdin.write(sentence_audio)
                    await process.stdin.drain()
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
        await websocket.send_json({
            "type": "token_update",
            "data": {"accumulated_tokens": websocket.app.state.accumulated_tokens}
        })
        print(f"Token update sent: {websocket.app.state.accumulated_tokens}")
    except Exception as e:
        print(f"Could not send EOS/token update: {e}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")

    await websocket.send_json({
        "type": "session_info",
        "data": {
            "system_specs": websocket.app.state.system_specs,
            "model_info": websocket.app.state.model_info,
            "accumulated_tokens": websocket.app.state.accumulated_tokens
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
                        speech_buffer.extend(frame_bytes)
                        silence_frames_count = 0
                    elif is_speaking:
                        silence_frames_count += 1
                        if silence_frames_count > silence_threshold_frames:
                            print("Speech ended due to silence.")
                            is_speaking = False
                            if speech_buffer:
                                speech_np_int16 = np.frombuffer(speech_buffer, dtype=np.int16)
                                speech_buffer = bytearray()
                                rms_energy = np.sqrt(np.mean(speech_np_int16.astype(np.float64)**2))
                                ENERGY_THRESHOLD = 300
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
                                        await stream_llm_and_synthesize(websocket, transcription)
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
    uvicorn.run(app, host="0.0.0.0", port=8000, ws_ping_interval=30, ws_ping_timeout=120)