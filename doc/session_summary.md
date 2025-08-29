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
