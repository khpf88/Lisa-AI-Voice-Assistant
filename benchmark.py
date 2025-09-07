import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from llama_cpp import Llama
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import os

# --- Configuration (should match main.py's model loading) ---
# IMPORTANT: Update these paths and names to match your actual models
# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the models directory
MODELS_DIR = os.path.join(BASE_DIR, 'models')

LLM_MODEL_NAME = os.getenv("LLM_MODEL_FILENAME", "tinyllama-1.1b-chat-v1.0.Q3_K_S.gguf")
LLM_MODEL_PATH = os.path.join(MODELS_DIR, LLM_MODEL_NAME)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "tiny") # or "base", "small"
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# --- Load Models (once for the benchmark script) ---
# These would be passed to the benchmark functions
print("Loading LLM model for benchmark...")
llm_instance = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
print("Loading Whisper model for benchmark...")
whisper_model_instance = WhisperModel(WHISPER_MODEL_NAME, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

# --- Sample Data ---
SAMPLE_LLM_PROMPT = "Tell me a short story about a brave knight."
# Create a dummy audio file for STT (e.g., 5 seconds of silence or noise)
# In a real benchmark, use a representative audio recording.
# For a real benchmark, replace this with actual audio data from a file:
# sample_audio_data, _ = sf.read("path/to/your/sample_audio.wav", dtype='float32')
SAMPLE_AUDIO_DURATION_SECONDS = 5
SAMPLE_AUDIO_SAMPLE_RATE = 16000
SAMPLE_AUDIO_DATA = np.zeros(SAMPLE_AUDIO_SAMPLE_RATE * SAMPLE_AUDIO_DURATION_SECONDS, dtype=np.float32) # 5 seconds of silence

# --- Core Functions (extracted from main.py logic) ---
def _sync_llm_inference(llm_instance: Llama, prompt: str):
    # Simplified for benchmark, actual logic from get_llm_response
    messages = [{"role": "user", "content": prompt}]
    response = llm_instance.create_chat_completion(messages=messages, max_tokens=50, stream=False)
    return response["choices"][0]["message"]["content"]

def _sync_stt_transcription(whisper_model_instance: WhisperModel, audio_data: np.ndarray):
    # Simplified for benchmark, actual logic from websocket_endpoint
    segments, _ = whisper_model_instance.transcribe(audio_data, language="en", beam_size=3)
    return "".join(s.text for s in segments)

# --- Benchmark Runner ---
async def run_benchmark(executor_type: str, num_runs: int, concurrent_tasks: int):
    executor = None
    if executor_type == "thread":
        executor = ThreadPoolExecutor(max_workers=concurrent_tasks)
    elif executor_type == "process":
        executor = ProcessPoolExecutor(max_workers=concurrent_tasks)
    else:
        raise ValueError("Invalid executor_type")

    print(f"\n--- Benchmarking with {executor_type} executor ({concurrent_tasks} concurrent tasks) ---")

    # LLM Benchmark
    llm_latencies = []
    tasks = []
    for _ in range(num_runs):
        tasks.append(asyncio.get_event_loop().run_in_executor(
            executor, _sync_llm_inference, llm_instance, SAMPLE_LLM_PROMPT
        ))
    start_time = time.perf_counter()
    await asyncio.gather(*tasks) # Wait for all tasks to complete
    end_time = time.perf_counter()
    
    total_llm_time = end_time - start_time
    avg_llm_latency = total_llm_time / num_runs
    llm_throughput = num_runs / total_llm_time
    print(f"LLM Average Latency: {avg_llm_latency:.4f} seconds")
    print(f"LLM Throughput: {llm_throughput:.2f} ops/sec")

    # STT Benchmark
    stt_latencies = []
    tasks = []
    for _ in range(num_runs):
        tasks.append(asyncio.get_event_loop().run_in_executor(
            executor, _sync_stt_transcription, whisper_model_instance, SAMPLE_AUDIO_DATA
        ))
    start_time = time.perf_counter()
    await asyncio.gather(*tasks) # Wait for all tasks to complete
    end_time = time.perf_counter()
    
    total_stt_time = end_time - start_time
    avg_stt_latency = total_stt_time / num_runs
    stt_throughput = num_runs / total_stt_time
    print(f"STT Average Latency: {avg_stt_latency:.4f} seconds")
    print(f"STT Throughput: {stt_throughput:.2f} ops/sec")

    executor.shutdown(wait=True)

async def main():
    # Single request latency
    await run_benchmark("thread", num_runs=1, concurrent_tasks=1)
    await run_benchmark("process", num_runs=1, concurrent_tasks=1)

    # Concurrent requests (e.g., 4 concurrent tasks, 10 total runs)
    await run_benchmark("thread", num_runs=10, concurrent_tasks=4)
    await run_benchmark("process", num_runs=10, concurrent_tasks=4)

if __name__ == "__main__":
    asyncio.run(main())
