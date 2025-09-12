# API Documentation

This document provides a detailed description of the API endpoints for the Lisa AI Voice Assistant.

## Overview

The primary interaction with Lisa is now exclusively through a WebSocket connection (`/ws`) for real-time, bidirectional audio and text streaming. Lisa handles Speech-to-Text (STT) and Text-to-Speech (TTS) operations. For intelligent responses, Lisa communicates with the central Orchestration Service (Mickey), which in turn interacts with the Intelligence & Memory Module (Riley).

## WebSocket Endpoint: `/ws`

*   **Method:** `WebSocket`
*   **Description:** Establishes a real-time, full-duplex connection for streaming audio from the client for Speech-to-Text (STT) and receiving streaming audio data from Text-to-Speech (TTS). After STT, the transcribed text is sent to the Orchestration Service (Mickey) for processing, and the intelligent response from Mickey is then synthesized into audio by Lisa's TTS engine.

### Incoming Messages (from Client to Server):

*   **Sample Rate:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "samplerate", "data": <number>}`
    *   **Description:** The first message sent by the client to inform the server of the client's audio sample rate.
*   **Audio Data:**
    *   **Type:** Binary data (bytes).
    *   **Content:** Raw PCM audio segments from the user's microphone.
    *   **Format:** Float32Array (raw PCM), at the sample rate specified in the initial `samplerate` message, mono channel.
    *   **Purpose:** Used by the server for VAD and STT processing.

### Outgoing Messages (from Server to Client):

*   **`calibration_start`:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "calibration_start"}`
    *   **Description:** Sent when the WebSocket connection is established to inform the client that the VAD is calibrating.
*   **`calibration_complete`:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "calibration_complete"}`
    *   **Description:** Sent when the VAD calibration is complete.
*   **`model_update`:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "model_update", "model": "string"}`
    *   **Description:** Sent before a response is streamed to inform the client which LLM is being used.

*   **`transcription`:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "transcription", "text": "string"}`
    *   **Description:** The final, combined transcription of a user's complete utterance, sent after speech detection.
*   **Audio Data:**
    *   **Type:** Binary data (bytes).
    *   **Content:** Opus-encoded audio segments of Lisa's synthesized voice response from the TTS engine. These are streamed as they become available.
    *   **Format:** Opus compressed audio, 24kHz sample rate, 16-bit signed integers (mono channel).
*   **`EOS`:**
    *   **Type:** `Text`
    *   **Content:** `"EOS"`
    *   **Description:** Sent by the server to indicate the End-Of-Stream for a complete audio response. This signals the client that no more audio segments are expected for the current response.
*   **`error`:**
    *   **Type:** `JSON`
    *   **Content:** `{"type": "error", "message": "string"}`
    *   **Description:** An error message from the server.
