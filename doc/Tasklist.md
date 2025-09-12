# Tasklist: Lisa AI Voice Assistant

## Phase 1: Core Functionality (Completed)

*   [x] Create basic project structure (`index.html`, `style.css`, `script.js`, `main.py`).
*   [x] Implement modern, minimalist, and responsive web UI.
*   [x] Set up FastAPI backend for web server and WebSocket communication.
*   [x] Integrate Google Gemini API for Large Language Model (LLM) responses.
*   [x] Integrate Text-to-Speech (TTS) via external Kokoro FastAPI service.
*   [x] Integrate **Faster Whisper** for local, CPU-optimized Speech-to-Text (STT).
*   [x] Implement interruption functionality via an on-screen button.

## Phase 2: Streaming & Stability Refactor (Completed)

*   [x] **Implement Server-Side VAD:**
    *   [x] Remove client-side VAD logic from `script.js`.
    *   [x] Implement continuous audio streaming from client to server.
    *   [x] Use `webrtcvad` on the server to detect speech and silence.
*   [x] **Implement Sentence-by-Sentence Streaming:**
    *   [x] Refactor LLM integration to support streaming responses token-by-token.
    *   [x] Refactor TTS pipeline to synthesize sentences as they arrive from the LLM stream.
*   [x] **Ensure Non-Blocking Operation:**
    *   [x] Move all CPU-intensive operations (VAD, STT, TTS) to background threads.
    *   [x] Resolve WebSocket disconnection issues caused by blocking the main event loop.
    *   [x] Adjusted WebSocket `ws_ping_timeout` for improved connection stability.
    *   [x] Fine-tuned client-side `initialBuffer` for smoother audio playback.
    *   [x] Migrated client-side audio processing from `ScriptProcessorNode` to `AudioWorklet` for better performance and modern browser compatibility.

## Phase 2.5: Debugging & Refinement (Completed)
*   [x] Resolve `IndentationError` and `SyntaxError` issues in `main.py`.

*   [x] Lower Whisper `LOGPROB_THRESHOLD` to improve transcription confidence.
*   [x] Temporarily remove FFmpeg stderr filtering for detailed debugging.
*   [x] Confirm server-side audio pipeline functionality (TTS, FFmpeg, WebSocket send).
*   [x] Debug client-side audio playback issues using browser developer tools and `debugger;` statements.
*   [x] **Fix TTS Number Pronunciation:** Added text normalization to correctly pronounce decimal numbers (e.g., "1.4" as "one point four").
*   [x] **Fix TinyLlama Prompting:** Changed `Llama.cpp` chat format to `zephyr` to ensure concise response instructions are followed.
*   [x] **Project Cleanup:** Removed unused files, models, and dependencies to prepare for containerization.

## Phase 3: UI and Session Tracking (Completed)
*   [x] **Implement Dynamic Session Info Panel:**
    *   [x] Display machine specs (CPU, GPU, RAM).
    *   [x] Display the specific STT, LLM, and TTS models being used.
    *   [x] Implement and display a session-wide token counter.
    *   [x] Refactor backend to send all session info to the client on connection.
    *   [x] Refactor frontend to parse and display the session information.

## Phase 4: Performance Optimization (Completed)

*   [x] **Audio Input & VAD:**
    *   [x] **Offload Resampling to Client:** Move audio resampling from the server to the client-side JavaScript to reduce server CPU load and network traffic.
*   [x] **Speech-to-Text (STT):**
    *   [x] **Tune Whisper Beam Size:** Reduce the `beam_size` parameter in `faster-whisper` (e.g., from 5 to 3) to significantly speed up transcription.
*   [x] **Language Model (LLM):**
    *   [x] **Enable Configurable Quantization:** Allow selecting LLM quantization level (e.g., `Q3_K_S`, `Q4_K_M`) via an environment variable (`LLAMA_QUANTIZATION`) for benchmarking.
*   [x] **Text-to-Speech (TTS):**
    *   [x] **Aggressive Sentence Splitting:** Implemented logic to start TTS synthesis on smaller text chunks (e.g., after commas) to reduce time-to-first-sound.
    *   [x] **Client-Side Buffering:** Increased the client-side audio buffer to mitigate choppiness caused by aggressive splitting.
    *   [x] **Implement RealtimeTTS:** (Now handled by external Kokoro FastAPI service. Further evaluation of smaller footprint TTS options is ongoing for alternative backends.)
    *   [x] **TTS Multi-core Optimization:** Implemented `ProcessPoolExecutor` for parallel TTS synthesis.
    *   [x] **Serialization Error Resolution:** Fixed `requires_grad` serialization issues when passing PyTorch tensors across process boundaries.

## Phase 5: Scalability and Flexibility (Completed)

*   [x] **Integrate `llama-cpp-python`:**
    *   [x] Add `psutil` to `requirements.txt` for hardware detection.
    *   [x] Pin a stable version of `llama-cpp-python` in `requirements.txt`.
    *   [x] Move the `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` model to a dedicated `models` directory.
    *   [x] Implement logic in `main.py` to load and use the GGUF model with `llama-cpp-python`.
*   [x] **Implement Automatic & Configurable Model Switching:**
    *   [x] **LLM Switching:**
        *   [x] Implement hardware detection logic (based on RAM) to automatically select between `llama-cpp-python`, `transformers`, and `Gemini`.
        *   [x] Make RAM thresholds configurable via environment variables.
        *   [x] Refactor the LLM integration in `main.py` to support seamless switching between the different backends.
    *   [x] **TTS Switching:**
        *   [x] Implement TTS engine selection (`Kokoro`, `KittenTTS`) via an environment variable (`TTS_ENGINE`).
        *   [x] Refactor the TTS integration to support this selection.

## Phase 5.5: Recent Bug Fixes and Enhancements

*   [x] **Implement Kitten TTS Voice Selection:** Allow users to select specific voices for Kitten TTS via environment variables.
*   [x] **Implement Kokoro TTS Voice Selection:** Allow users to select specific voices for Kokoro TTS via environment variables.
*   [x] **Fix LLM Conversation History:** Implement conversation history management to prevent the LLM from repeating previous responses.
*   [x] **Fix Audio Player Quota Exceeded Error:** Resolve the `QuotaExceededError` in the client-side audio player by ensuring proper cleanup and re-initialization of MediaSource and SourceBuffer objects.
*   [x] **Filter LLM Prompt Tags from TTS Output:** Remove unwanted LLM prompt tags (e.g., `<|user|>`, `<|end|>`) from the text sent to the TTS engine to ensure clean speech responses.
*   [x] **Fix KittenTTS Crash:** Resolved an `AttributeError` during KittenTTS synthesis caused by an incorrect method call, ensuring stable audio generation.
*   [x] **Implement Declarative Resource Management:** Refactored the TTS system to be fully agnostic. Each TTS engine class now declares its own resource profile (RAM and CPU requirements). The application reads this profile at startup and calculates the optimal number of worker processes based on live system resources, preventing memory exhaustion and maximizing performance.

## Phase 5.6: LLM & TTS Refinements (September 2025)

*   [x] **Refactor LLM Summarization:** Replaced the two-step summarization process with a single system prompt instructing the LLM to be concise, improving efficiency and resolving truncation issues.
*   [x] **Optimize Resource Usage:** Removed the loading of the unused 'coding' LLM to reduce memory footprint.
*   [x] **Integrate Marvis TTS:** Added Marvis TTS as a third, selectable TTS engine, introducing voice cloning capabilities.
*   [x] **Update Project Documentation:** Updated PRD, architecture, API, and task list documents to reflect all recent changes.



## Phase 6: Future Enhancements (Planned)

*   [ ] **Evaluate and Integrate New TTS Options:**
    *   [x] Evaluate SparkTTS and F5-TTS for suitability on constrained devices.
    *   [x] Integrate and test KittenTTS for performance and quality on target hardware.
    *   [ ] Explore SparkTTS if KittenTTS does not meet naturalness expectations, with careful consideration of resource demands.
*   [ ] **Improve Model Accuracy:**
    *   [ ] Explore context management for longer, more coherent conversations.
    *   [ ] Evaluate different STT and local LLM models for better performance/accuracy trade-offs.
*   [ ] **Voice Cloning:**
    *   [ ] Research and implement voice cloning capabilities for personalization, particularly for the elderly care companion app use case.
*   [x] **Deployment & Packaging:**
    *   [x] Containerize the application (e.g., using Docker) - now functional via Docker Compose.
    *   [ ] Provide simplified deployment instructions.

## Phase 7: Lisa-Riley Integration

*   [ ] **Define Riley's API/Interface:**
    *   [ ] Finalize data structures for communication between Lisa and Riley (input to Riley, output from Riley).
*   [ ] **Implement Conditional Call to Riley in Lisa:**
    *   [ ] Identify interception point in `main.py` after STT and before Lisa's LLM.
    *   [ ] Implement "certain rules" logic to decide when to call Riley.
    *   [ ] Implement client-side logic in Lisa to send data to Riley's API.
*   [ ] **Develop Riley's Core Functionality:**
    *   [ ] Set up Riley's project structure and basic FastAPI application.
    *   [ ] Choose and integrate a database for memory storage (short-term, long-term).
    *   [ ] Implement memory storage and retrieval mechanisms.
    *   [ ] Implement intelligence/response generation logic within Riley.
    *   [ ] Implement Riley's API endpoint to receive data from Lisa.
*   [ ] **Integrate Riley's Response into Lisa's TTS:**
    *   [ ] Modify Lisa's `main.py` to use Riley's generated response for TTS when Riley is consulted.
*   [ ] **Testing and Validation:**
    *   [ ] Develop unit tests for Lisa-Riley communication.
    *   [ ] Develop end-to-end integration tests for memory-aware conversations.
    *   [ ] Conduct performance testing for the integrated system.