# Architecture Diagrams

This document contains diagrams that visualize the current and future architecture of the Lisa application.

## Current Architecture

This diagram shows the current setup of the application, now supporting real-time, two-way voice interaction with server-side Voice Activity Detection (VAD), streaming LLM responses, and focusing on a local, CPU-optimized LLM for continued operation on constrained machines. It uses a client-server model with WebSocket for efficient audio and text streaming, configured with `ws_ping_interval` and `ws_ping_timeout` for connection keep-alive. The client sends raw audio (float32, converted to int16 for server-side VAD), and the server streams back audio encoded as Opus within a WebM container.

```mermaid
graph TD
    subgraph Browser
        A[index.html] --> B(script.js);
        B --> C{Microphone Input};
        C -- "1. Raw Audio (AudioWorklet)" --> D[AudioWorklet Processor];
        D -- "2. Raw Audio Chunks (WebSocket)" --> E;
        B --> H[Audio Output];
    end

    subgraph Local Server (main.py)
        E[FastAPI Server (WebSocket)] --> I[Accumulate Raw Audio];
        I -- "3. VAD (webrtcvad)" --> J[Speech Segments];
        J -- "4. Transcribe (Faster Whisper STT Engine)" --> K[LLM Orchestrator];
        K -- "5. Prompt" --> L[Local LLM (e.g., LiquidAI LFM)];
        L -- "6. Streaming Response" --> M[Sentence Tokenization];
        M -- "7. TTS Raw Audio Chunks (TTS Engine)" --> N[Streaming Opus Encoder (FFmpeg)];
        N -- "8. Streamable WebM/Opus Packets (WebSocket)" --> B;
        N -- "9. EOS Signal (WebSocket)" --> B;
    end
```

## Future Architecture (with Emotion Recognition and Enhanced Modularity)

This diagram illustrates the long-term vision for Lisa, incorporating advanced features like emotion detection and empathetic responses, alongside a more modular and scalable LLM integration.

```mermaid
graph TD
    subgraph Browser
        A[User's Voice] --> B(Microphone Input);
        B -- "1. Speech Detected (VAD)" --> C[Client-side Audio Processing];
        C -- "2. Encoded Audio Stream" --> D[WebSocket Connection];
        D --> E[Audio Output];
    end

    subgraph Local Server
        subgraph AI Core
            F[STT Engine] -- "3. Text" --> G[LLM Orchestrator];
            H[Emotion Recognition (Optional)] -- "4. Emotion Data" --> G;
            G -- "5. Contextualized Prompt" --> I[Multiple LLM Providers];
            I -- "6. Streaming Response" --> J[TTS Engine];
            J -- "7. Encoded Audio Stream" --> D;
        end
    end
```

## Roadmap to a Reusable Agent

This section outlines the high-level steps to evolve the Lisa application from a simple prototype into a robust, reusable, and scalable AI voice agent.

### Step 1: Prove the Core Functionality (Achieved & Refined)

The immediate goal of getting the core application working, including real-time two-way voice interaction, server-side VAD, streaming LLM responses, and multi-LLM provider support, has been largely achieved. We have confirmed the server-side audio pipeline is fully functional, and the audio playback issues have been resolved. The current focus is on optimizing performance.

### Step 2: Refine and Harden the Agent

Once the core functionality is proven, the next step is to make the agent more robust and flexible. This includes:
*   **Refining the API:** Designing a clean and well-documented API that is easy for other applications to use.
*   **Adding Comprehensive Error Handling:** Implementing comprehensive error handling across all components (client, server, LLM integrations, TTS) to make the agent more resilient and provide better user feedback.
*   **Configuration Management:** Using configuration files to manage settings like model names, server ports, API keys, and LLM provider weights, making the agent easily adaptable.
*   **Audio Playback Optimization:** Addressing remaining issues with audio playback quality, timing, and gaps.

### Step 3: Containerize the Agent with Docker

This is the key step to making the agent "plug and play." We will:
*   **Create a `Dockerfile`:** This file will contain all the instructions to package our Python server, AI models, and all dependencies into a single, self-contained Docker container.
*   **Build a Docker Image:** This will create a portable image of our agent that can be run on any machine with Docker installed.

### Step 4: Deploy and Scale

With the agent containerized, we can then focus on deployment and scalability:
*   **Deployment:** The agent can be easily deployed to a local machine, a local server, or a cloud provider.
*   **Scalability:** For high-demand applications, we can explore advanced topics like:
    *   **Horizontal Scaling:** Running multiple instances of the agent behind a load balancer.
    *   **Asynchronous Processing:** Using task queues to handle long-running model inference tasks without blocking the server.