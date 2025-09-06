# Lisa - AI Voice Assistant

This document is the main playbook for the Lisa AI Voice Assistant project. It provides a comprehensive guide for setting up and running the application.

## Project Description

Lisa is a web-based AI voice assistant that you can have natural conversations with. It uses a hybrid approach: **Speech-to-Text (STT) and the Large Language Model (LLM) run locally on your machine (or within a local Docker container)**, while **Text-to-Speech (TTS) is provided by a separate, Dockerized external service**, ensuring privacy and efficient operation on various hardware.



## Features

*   **Conversational AI:** Engage in natural voice conversations with Lisa, powered by a local, CPU-optimized Large Language Model.
*   **Real-time Speech-to-Text:** Transcribes your speech locally using highly optimized Faster Whisper, providing quick feedback.
*   **Continuous Listening & Sentence Combining:** Intelligently combines your spoken phrases into full sentences for better context understanding by the AI.
*   **Interruptible Responses:** You can interrupt Lisa's audio responses at any time using a dedicated button.
*   **Clear Voice Responses:** Lisa responds with a clear, audible voice, powered by the external Kokoro FastAPI service.
*   **Modern Web Interface:** A clean, minimalist, and responsive user interface.

## Technology Stack

*   **Frontend (User Interface):**
    *   HTML
    *   CSS
    *   JavaScript (with Web Audio API for microphone input and resampling)
*   **Backend (The "Brain"):**
    *   Python
    *   FastAPI (for the web server and WebSocket communication)
    *   Docker Compose (for orchestrating services)
*   **AI Models and Libraries:**
    *   `faster-whisper`: For highly optimized local Speech-to-Text (STT) inference on CPU.
    *   `Kokoro FastAPI Wrapper`: External Dockerized service for advanced Text-to-Speech (TTS) capabilities.
    *   `numpy`: For numerical operations, especially with audio data.
    *   `soundfile`: For handling audio files.
    *   `python-dotenv`: For managing API keys and environment variables.

## Prerequisites

To run Lisa using Docker Compose, you need to have the following software installed on your computer:

1.  **Docker Desktop:** This includes Docker Engine and Docker Compose. Download and install it from the official Docker website: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
    *   Ensure Docker Desktop is running before proceeding.

## Installation and Setup

Follow these steps to set up and run the project using Docker Compose:

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd lisa
    ```

2.  **Build and Run with Docker Compose:**
    *   Navigate to the `lisa` directory:
        ```bash
        cd C:\DevProjects\lisa
        ```
    *   Build the Docker images (this might take some time on the first run):
        ```bash
        docker-compose build
        ```
    *   Start the services:
        ```bash
        docker-compose up
        ```
    *   The Lisa web interface will be accessible at `http://localhost:8001`.

3.  **Obtain API Keys (Optional):**
    *   If you plan to use cloud-based LLMs (e.g., Google Gemini), obtain your API key from the respective platform.
    *   **Create a `.env` file:** In your project's root directory (`C:\DevProjects\lisa\`), create a file named `.env` and add your API keys to it (e.g., `GEMINI_API_KEY=YOUR_GEMINI_API_KEY`). These environment variables will be automatically picked up by Docker Compose.

## Running the Application

The application is now running via Docker Compose.

1.  **Access the Frontend Web App:**
    *   Open your web browser and navigate to `http://localhost:8001`.
    *   Allow microphone access when prompted.
    *   Start speaking to Lisa!

## Project Structure

*   `index.html`: The main web page for the application.
*   `style.css`: The stylesheet for the web page.
*   `script.js`: The JavaScript code that handles the user interface and communication with the backend.
*   `main.py`: The Python script for the backend server, which contains the AI logic.
*   `docker-compose.yml`: Defines and orchestrates the multi-container Docker application.
*   `Doc/`: A folder containing all the project documentation:
    *   `specification.md`: The initial project specification.
    *   `PRD.md`: The Product Requirements Document.
    *   `Tasklist.md`: The list of tasks for the project.
    *   `architecture.md`: The architecture diagrams and roadmap.
    *   `api_documentation.md`: The documentation for the API endpoints.
    *   `session_summary.md`: A detailed log of development progress and decisions.
