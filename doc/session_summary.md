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

## Session Update: Iterative Refinement of LLM Summarization for Conciseness and Completeness

**Date:** September 7, 2025

**Summary:** This session involved multiple iterations to refine the LLM summarization process to achieve clear, concise, and untruncated responses with complete sentences, addressing user feedback on audio truncation.

**Key Enhancements & Resolutions:**

*   **Initial Summarization Prompt Refinement:**
    *   The `summarization_prompt` in `main.py` was initially modified from a strict 50-word limit to aim for a more conversational style.
*   **Aggressive `max_tokens` Reduction for Summarization LLM:**
    *   The `max_tokens` parameter for the summarization LLM in `summarize_text` was progressively reduced from 500 to 100, and then to 50, in an attempt to force shorter summaries.
*   **Implementation of Hard Character Limit:**
    *   A hard character limit of 150 was introduced in `get_llm_response` to ensure the text sent to the TTS engine is always within a manageable length. This limit includes logic to attempt breaking at natural sentence or word boundaries to prevent mid-sentence truncation.
*   **Final Summarization Prompt Adjustment for Completeness:**
    *   The `summarization_prompt` was further refined to explicitly instruct the LLM to summarize in "a few complete sentences," emphasizing grammatical correctness and avoiding introductory phrases, to ensure the output is short, sweet, and untruncated.

**Outcome:** The combination of a refined summarization prompt, reduced `max_tokens` for the summarization LLM, and a hard character limit with intelligent breaking logic has successfully resulted in concise, complete, and untruncated audio responses, meeting the user's requirements.

## Session Update: LLM Simplification and Marvis TTS Integration

**Date:** September 8, 2025

**Summary:** This session focused on resolving a persistent summarization bug, optimizing resource usage, and integrating a new TTS engine.

**Key Enhancements & Resolutions:**

*   **LLM Summarization Refactor:** The previous two-step summarization process, which was causing truncated output, was entirely removed. It was replaced with a more efficient single-prompt approach where a system prompt instructs the LLM to be concise from the outset.
*   **Resource Optimization:** The application no longer loads the secondary 'coding' LLM at startup, reducing the application's memory footprint.
*   **Marvis TTS Integration:** The powerful Marvis TTS engine was successfully integrated as a third, selectable engine option (`TTS_ENGINE=marvis`). This introduces advanced voice cloning capabilities to the project.
*   **Documentation Update:** All relevant project documents (`PRD.md`, `architecture.md`, `api_documentation.md`, `Tasklist.md`) were updated to reflect these significant changes.

## Session Update: Removal of Unstable TTS Engines

**Date:** September 11, 2025

**Summary:** This session focused on removing the `pyttsx3` and `piper-tts` engines to improve project stability.

**Key Enhancements & Resolutions:**

*   **`pyttsx3` Removal:** The `pyttsx3` engine was identified as a non-performant bottleneck and was removed from the project.
*   **`piper-tts` Removal:** After a lengthy and unsuccessful debugging process, the `piper-tts` library was found to be unstable and its API unpredictable in the current environment. It has been completely removed from the project to ensure stability.
*   **Code and Documentation Cleanup:** All code and documentation related to `pyttsx3` and `piper-tts` has been removed from the project.

**Outcome:** The project is now in a more stable state, with only the reliable TTS engines (`Kokoro`, `KittenTTS`, `Marvis`) remaining. This simplifies the configuration and reduces maintenance overhead.

## Session Update: LLM Response Handling

**Date:** September 11, 2025

**Summary:** This session focused on improving the robustness of the LLM response handling.

**Key Enhancements & Resolutions:**

*   **LLM Output Truncation:** Implemented logic to strictly parse the LLM's output and truncate it at the first occurrence of a stop token (e.g., `<|user|>`, `<|end|>`). This prevents the model from generating and speaking hallucinated conversational turns, which was identified as a critical bug.

**Outcome:** The application is now more robust against common LLM generation errors and will produce cleaner, more predictable audio responses across all supported models.

## Session Update: LLM and TTS Configuration

**Date:** September 11, 2025

**Summary:** This session focused on adding new configuration options for the LLM and TTS.

**Key Enhancements & Resolutions:**

*   **LLM Detail Level:** Implemented a new `LLM_DETAIL_LEVEL` environment variable to control the verbosity and detail of the LLM's responses. This dynamically adjusts the system prompt and `max_tokens` for the LLM.
*   **TTS Tempo Control:** Implemented a new `TTS_TEMPO` environment variable to control the speaking rate of the TTS voice. This feature is currently only supported by the `Kokoro` TTS engine.

**Outcome:** The application now offers more fine-grained control over the LLM's response style and the TTS speaking tempo, enhancing user customization.

## Session Update: Fix for f-string SyntaxError

**Date:** September 11, 2025

**Summary:** This session addressed a recurring `SyntaxError` in `main.py` related to f-string formatting.

**Key Enhancements & Resolutions:**

*   **f-string Formatting:** The `get_llm_response` function was refactored to use triple-quoted f-strings for all multi-line string literals. This ensures correct parsing by the Python interpreter and resolves the `SyntaxError: unterminated f-string literal`.

**Outcome:** The `main.py` file is now syntactically correct, eliminating the f-string related errors.