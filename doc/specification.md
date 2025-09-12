My name is Lisa, a web app AI voice assistant for continuous two-way voice interaction with you. The app feature client-side Voice Activity Detection (VAD) and support for multiple Large Language Model (LLM) providers.
The app should be developed using the latest web tech stack (HTML, CSS, JavaScript) that is modern, professional, minimalist and responsive when running on the computer and mobile devices.

**Speech-to-Text (STT):** Faster Whisper will be used for local, CPU-optimized STT inference. Client-side VAD ensures that only active speech segments are sent for transcription.

**Text-to-Speech (TTS):** The application supports multiple, selectable, local TTS engines. The default is `Kokoro`, but it can be configured to use `KittenTTS` or `Marvis`. TTS audio will be generated as a continuous stream in Opus format, allowing for both low initial latency and gapless playback.

**Language Model (LLM):** The application currently focuses on **local, CPU-optimized LLMs** (e.g., TinyLlama or LiquidAI LFM). While multiple LLM providers (Google Gemini API, OpenAI, DeepSeek, Grok) can be integrated, they are currently disabled/commented out. LLM responses are streamed for low-latency TTS.

## Performance and Portability Assessment

### Performance Scalability

The current application's performance, particularly the Speech-to-Text (STT) component using Faster Whisper on CPU, is directly tied to the underlying hardware capabilities. Client-side VAD reduces network traffic, and Opus encoding for TTS further optimizes data transfer, contributing to lower perceived latency. While functional on constrained machines, significant performance improvements can be achieved on newer hardware:

*   **CPU Upgrade:** A newer machine with a more powerful CPU (more cores, higher clock speeds, and modern instruction set support) would drastically reduce STT transcription latency, leading to a more real-time and responsive user experience.
*   **GPU Acceleration:** If deployed on a machine equipped with a compatible NVIDIA GPU (and properly configured with CUDA and cuDNN), Faster Whisper can leverage GPU acceleration for orders of magnitude faster performance, potentially achieving near-instantaneous transcription.

### Portability and Redeployment

The current technology stack is highly portable and designed for flexible deployment:

*   **Cross-Platform Compatibility:** The backend (Python and FastAPI) is cross-platform and can run on various operating systems (Windows, Linux, macOS).
*   **Self-Contained STT:** Faster Whisper is a Python library that manages its own model caching, simplifying deployment as it's not reliant on external executables once set up.
*   **Local LLM/TTS:** The current focus on local LLMs and a variety of local TTS engines means these components are independent of external cloud services, requiring only local resources.
*   **Standard Frontend:** The HTML/CSS/JavaScript frontend runs in any modern web browser.
*   **Containerization Readiness:** The architecture is well-suited for containerization (e.g., using Docker), which would further simplify deployment by bundling the application and all its dependencies into a single, portable unit.

## Future Vision

The long-term goal for Lisa is to evolve into a truly empathetic and natural-sounding voice assistant, with continuous improvements in performance and user experience.

### Speech Emotion Recognition (SER)

Lisa will be able to understand not just what the user is saying, but also how they are feeling. By analyzing the user's tone of voice, Lisa will be able to detect emotions such as happiness, sadness, anxiety, and excitement. This will be achieved using advanced speech recognition models like Wav2Vec2.0.

### Empathetic and Emotionally-Aware Responses

Based on the detected emotion, Lisa will be able to tailor its responses to be more empathetic and appropriate to the user's emotional state. This will create a more engaging and supportive conversational experience.

### Audio Playback Quality/Gaps

Further investigation and fine-tuning of audio playback to eliminate perceived gaps and improve overall quality.

### VAD Sensitivity Tuning

Adjusting client-side VAD parameters for optimal speech detection in various environments. The application now features dynamic VAD thresholding, which automatically adjusts the energy threshold based on the ambient noise level of the user's environment. This results in more reliable speech detection in both quiet and noisy settings.

### Error Handling & Robustness

Add more comprehensive error handling and user feedback mechanisms across all components.

### Performance Optimization

Continue exploring ways to reduce latency, potentially revisiting WebRTC or exploring alternative audio codecs if necessary.

### Voice Interruption

Implement a wake word or specific phrase to interrupt Lisa's response without needing a button press.

### STT Accuracy

Further improve STT accuracy (e.g., by trying larger Faster Whisper models if performance allows).

### LLM Accuracy

Improve LLM response accuracy through prompt engineering or further LLM tuning.

### Customization

Provide options for customizing the voice, language, and other parameters.

### Advanced Conversation Management

Implement features for managing conversation history, context, and user profiles.
