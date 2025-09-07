## Session Update: Evaluation of SparkTTS and F5-TTS

**Date:** August 27, 2025

**Summary:** This session involved evaluating SparkTTS and F5-TTS as potential alternative Text-to-Speech (TTS) solutions, particularly for their suitability on constrained devices and advanced features like voice cloning and naturalness.

**Key Findings:**

*   **SparkTTS:**
    *   **Pros:** Leverages LLMs for synthesis, aims for high naturalness (95% human-like), supports zero-shot voice cloning, multilingual, open-source with Python accessibility.
    *   **Cons:** Actual footprint and computational demands on *extremely* constrained devices (e.g., Raspberry Pi) are unclear. Its LLM reliance might make it heavier than Kokoro or KittenTTS.
    *   **Suitability:** Promising for local integration if its resource demands can be met by the target device.

*   **F5-TTS:**
    *   **Pros:** Uses advanced AI algorithms (Flow Matching, Diffusion Transformer) for highly realistic and expressive voices, designed for real-time processing, supports zero-shot voice cloning, multilingual.
    *   **Cons:** Appears to be more of a research project with an online tool/API, and less clear about a direct Python library for local, on-device inference. Its advanced architecture might be computationally intensive for constrained devices.
    *   **Suitability:** Less suitable for direct local, on-device integration due to unclear local Python API and potential resource demands.

**Recommendation:**

*   Both SparkTTS and F5-TTS offer advanced features and potentially higher naturalness than Kokoro or KittenTTS. However, their suitability for *extremely* constrained devices remains a significant question mark.
*   **KittenTTS** (which was recently integrated) is specifically designed for constrained devices (under 25MB model size, CPU-optimized) and should be **thoroughly tested first** on the target hardware to evaluate its performance and voice quality.
*   If KittenTTS does not meet expectations for naturalness *and* there's a willingness to potentially sacrifice some performance or increase device capabilities, then **SparkTTS could be explored as a next step**, with the understanding that its suitability for *extremely* constrained devices needs to be verified. F5-TTS is likely not a good fit for local, on-device use.

## Session Update: Docker Build Debugging

**Date:** August 29, 2025

**Summary:** This session focused on debugging a series of `ModuleNotFoundError` issues that occurred when running the application inside a Docker container. The build process was completing successfully, but the application would crash at runtime.

**Key Findings & Resolutions:**

*   **`ModuleNotFoundError: No module named 'num2words'`:**
    *   **Diagnosis:** The `num2words` package was used in the application code but was missing from the `requirements.txt` file. A formatting error also occurred when attempting to add it, causing an invalid requirement string (`kittentts"num2words"`).
    *   **Resolution:** The `requirements.txt` file was corrected to include `num2words` on its own line.

*   **`ModuleNotFoundError: No module named 'kokoro'`:**
    *   **Diagnosis:** This was a more complex issue. Despite `kokoro-tts` being correctly listed in `requirements.txt` and the `pip install` command finishing without error during the Docker build, the module was not available at runtime. Further investigation using a verification step (`RUN python -c "import kokoro"`) proved that `pip` was silently failing to install the package correctly from within the `requirements.txt` context.
    *   **Resolution:** The `Dockerfile` was restructured to isolate the problematic package. `kokoro-tts` was removed from `requirements.txt` and installed using a separate, dedicated `RUN pip install kokoro-tts` command. An explicit verification step was added immediately after this command to ensure the build would fail if the import was not successful, preventing runtime errors.

## Session Update: Final Docker Diagnosis & Project State

**Date:** August 29, 2025

**Summary:** After multiple attempts to fix the `kokoro-tts` installation failure within the Docker environment (including using a full Python base image), it has been concluded that the `kokoro-tts` package is fundamentally incompatible with the `python:3.12` Docker environment. The package fails to install correctly in a way that `pip` can detect, leading to a `ModuleNotFoundError` at runtime.

**Decision:**

*   Per the project owner's request, `kokoro-tts` will remain the primary TTS engine in the codebase. No workaround to switch to a different default TTS (like `KittenTTS`) will be implemented.
*   The project's `Dockerfile` will be left in a state that attempts to install `kokoro-tts`.
*   **Result:** The Docker build for this project is currently **not functional** and will fail until a new version of `kokoro-tts` is released that resolves this installation issue.

**Recommendation:**

*   A note should be added to the project's main `README.md` to warn users that the current Docker configuration is not buildable.
*   The local changes to `Dockerfile`, `requirements.txt`, and `doc/session_summary.md` should be pushed to the GitHub repository to record this final state.

## Session Update: Resolution of Kokoro TTS Docker Issue

**Date:** August 29, 2025

**Summary:** The persistent `kokoro-tts` installation issues within the Docker environment have been resolved by adopting an external, pre-built Dockerized solution for Kokoro TTS. Instead of installing `kokoro-tts` directly within Lisa's container, Lisa now communicates with a separate `Kokoro FastAPI Wrapper` service running in its own Docker container.

**Resolution:**

*   The `kokoro-tts` dependency has been removed from Lisa's `requirements.txt`.
*   Lisa's `main.py` has been modified to make HTTP API calls to the `Kokoro FastAPI Wrapper` service for TTS synthesis.
*   A `docker-compose.yml` file has been introduced to orchestrate both Lisa's container and the `Kokoro FastAPI Wrapper` container.
*   The `Dockerfile` for Lisa has been cleaned up to remove the problematic `kokoro` import verification step.

**Result:** The Lisa project's Docker build is now functional, and the TTS functionality is provided by the external Kokoro FastAPI service, allowing for a robust and maintainable setup.

## Session Update: Voice Selection and Stability Enhancements

**Date:** September 6, 2025

**Summary:** This session focused on enhancing voice selection capabilities for both Kitten TTS and Kokoro TTS, as well as addressing critical stability issues related to LLM conversation history and client-side audio playback.

**Key Enhancements & Resolutions:**

*   **Kitten TTS Voice Selection:**
    *   Implemented functionality to allow users to select specific Kitten TTS voices (e.g., `expr-voice-4m`) via the `KITTEN_TTS_VOICE` environment variable in `.env`.
    *   Corrected hardcoded voice selection in `main.py` to utilize the environment variable.

*   **Kokoro TTS Voice Selection:**
    *   Implemented functionality to allow users to select specific Kokoro TTS voices (e.g., `af_bella`) via the `KOKORO_TTS_VOICE` environment variable in `.env`.
    *   Identified and corrected incorrect voice filenames (e.g., "Sky" vs. "af_sky") by consulting `VOICES.md` in the Kokoro TTS repository.
    *   Corrected hardcoded voice selection in `main.py` to utilize the environment variable.

*   **LLM Conversation History Management:**
    *   Implemented a conversation history mechanism for the LLM in `main.py` to prevent the LLM from repeating previous responses.
    *   The LLM now receives the full conversation context, leading to more coherent and non-repetitive interactions.

*   **Client-Side Audio Playback Stability:**
    *   Resolved the `QuotaExceededError` in `script.js` by ensuring proper cleanup and re-initialization of `MediaSource` and `SourceBuffer` objects in the `AudioPlayer` class.
    *   Added explicit `audioPlayer.stop()` calls when new transcriptions are received to ensure a clean state for each new audio stream.

*   **Filter LLM Prompt Tags from TTS Output:**
    *   Removed unwanted LLM prompt tags (e.g., `<|user|>`, `<|end|>`, `<|assistant|>`) from the text sent to the TTS engine in `main.py` to ensure clean speech responses.

## Session Update: TTS Multi-core Optimization & Serialization Fixes

**Date:** September 6, 2025

**Summary:** This session focused on implementing multi-core processing for Text-to-Speech (TTS) synthesis to improve playback smoothness and reduce latency, leveraging `ProcessPoolExecutor`. Significant challenges related to PyTorch tensor serialization across process boundaries were encountered and resolved.

**Key Enhancements & Resolutions:**

*   **TTS Multi-core Implementation:**
    *   Refactored TTS synthesis to utilize `ProcessPoolExecutor` for parallel processing of sentences, aiming to leverage multiple CPU cores.
    *   Confirmed that the system is now capable of using multiple cores for TTS synthesis.
*   **PyTorch Tensor Serialization Fixes:**
    *   Addressed `Cowardly refusing to serialize non-leaf tensor which requires_grad` errors.
    *   Implemented `.detach()` calls on `torch.Tensor` outputs from both Kokoro and KittenTTS models before passing them across process boundaries.
    *   Refactored `_perform_tts_synthesis_sync` to load TTS models within each worker process, preventing serialization issues with the model instances themselves.
*   **Dependency Resolution:**
    *   Resolved `setuptools` version conflict by explicitly adding `setuptools<80` to `requirements.txt`.

**Outcome:** The TTS pipeline is now optimized for multi-core CPUs, and the application runs without the previously encountered serialization errors, leading to potentially smoother and faster speech output.

## Session Update: LLM Summarization Logic Refinement

**Date:** September 7, 2025

**Summary:** The LLM summarization logic in `main.py` was refined to address truncation issues and improve the conversational quality of responses. The previous strict 50-word limit for summarization has been removed.

**Key Changes:**

*   The `summarization_prompt` in the `summarize_text` function in `main.py` was modified.
*   The new prompt aims for concise summarization with enough information for an authentic and close conversational style, without a strict word count.

**Outcome:** This change is expected to resolve truncation problems for larger LLM responses and enhance the natural flow of conversation.