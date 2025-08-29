# Lisa - AI Voice Assistant

This document is the main playbook for the Lisa AI Voice Assistant project. It provides a comprehensive guide for setting up and running the application.

## Project Description

Lisa is a web-based AI voice assistant that you can have natural conversations with. It uses a hybrid approach: **Speech-to-Text (STT) runs locally on your machine**, and the **Large Language Model (LLM) and Text-to-Speech (TTS) are also primarily local and CPU-optimized**, ensuring privacy and efficient operation on various hardware.

## Features

*   **Conversational AI:** Engage in natural voice conversations with Lisa, powered by a local, CPU-optimized Large Language Model.
*   **Real-time Speech-to-Text:** Transcribes your speech locally using highly optimized Faster Whisper, providing quick feedback.
*   **Continuous Listening & Sentence Combining:** Intelligently combines your spoken phrases into full sentences for better context understanding by the AI.
*   **Interruptible Responses:** You can interrupt Lisa's audio responses at any time using a dedicated button.
*   **Clear Voice Responses:** Lisa responds with a clear, audible voice, powered by Kokoro-TTS.
*   **Modern Web Interface:** A clean, minimalist, and responsive user interface.

## Technology Stack

*   **Frontend (User Interface):**
    *   HTML
    *   CSS
    *   JavaScript (with Web Audio API for microphone input and resampling)
*   **Backend (The "Brain"):**
    *   Python
    *   FastAPI (for the web server and WebSocket communication)
*   **AI Models and Libraries:**
    *   `faster-whisper`: For highly optimized local Speech-to-Text (STT) inference on CPU.
    *   `kokoro`: For advanced Text-to-Speech (TTS) capabilities.
    *   `numpy`: For numerical operations, especially with audio data.
    *   `soundfile`: For handling audio files (used by Kokoro-TTS).
    *   `python-dotenv`: For managing API keys and environment variables.

## Prerequisites

Before you begin, you need to have the following software installed on your computer:

1.  **Python 3.12:** It is crucial to install Python version 3.12, as newer versions (like 3.13) may have compatibility issues with some of the required libraries. You can download it from the official Python website: [https://www.python.org/downloads/release/python-3120/](https://www.python.org/downloads/release/python-3120/)
    *   During installation, make sure to check the box that says **"Add Python to PATH"** (or similar wording).
2.  **Visual Studio Build Tools:** These are required for some of the Python libraries (like `faster-whisper`'s dependencies) to compile correctly. You can download them from here: [https://visualstudio.microsoft.com/downloads/](https://visualstudio.microsoft.com/downloads/) (Scroll down to "Tools for Visual Studio" and download the "Build Tools for Visual Studio"). When installing, make sure to select the **"Desktop development with C++"** workload.

## Installation and Setup

Follow these steps to set up the project:

1.  **Install Prerequisites:** Make sure you have installed all the software listed in the "Prerequisites" section.
2.  **Open the Developer Command Prompt:** After installing Python 3.12, close any existing command prompts and open a **NEW** "Developer Command Prompt for VS" as an administrator.
3.  **Navigate to the project directory:**
    ```bash
    cd C:\DevProjects\lisa
    ```
4.  **Install Python Libraries:** Run the following command to install all the necessary Python libraries. This command explicitly uses Python 3.12:
    ```bash
    py -3.12 -m pip install fastapi uvicorn python-dotenv kokoro numpy soundfile faster-whisper
    ```
5.  **Obtain API Keys (Optional):**
    *   If you plan to use cloud-based LLMs (e.g., Google Gemini), obtain your API key from the respective platform.
    *   **Create a `.env` file:** In your project's root directory (`C:\DevProjects\lisa\`), create a file named `.env` and add your API keys to it (e.g., `GEMINI_API_KEY=YOUR_GEMINI_API_KEY`).

## Running the Application

The application consists of a backend server and a frontend web app.

1.  **Start the Backend Server:**
    *   In the Developer Command Prompt (where you installed libraries), navigate to the project directory:
        ```
        cd C:\DevProjects\lisa
        ```
    *   Start the server with this command:
        ```
        py -3.12 main.py
        ```
    *   **Important Notes for Server Startup:**
        *   **Model Download:** The first time you run the server, Faster Whisper will automatically download the `small.en` model. This might take a few minutes depending on your internet connection. Subsequent runs will use the cached model.
        *   **Server Ready:** The server is ready when you see messages like `Application startup complete.` and `Uvicorn running on http://0.0.0.0:8000`.

2.  **Access the Frontend Web App:**
    *   Simply open the `index.html` file in your web browser.
    *   Allow microphone access when prompted.
    *   Start speaking to Lisa!

## Project Structure

*   `index.html`: The main web page for the application.
*   `style.css`: The stylesheet for the web page.
*   `script.js`: The JavaScript code that handles the user interface and communication with the backend.
*   `main.py`: The Python script for the backend server, which contains the AI logic.
*   `Doc/`: A folder containing all the project documentation:
    *   `specification.md`: The initial project specification.
    *   `PRD.md`: The Product Requirements Document.
    *   `Tasklist.md`: The list of tasks for the project.
    *   `architecture.md`: The architecture diagrams and roadmap.
    *   `api_documentation.md`: The documentation for the API endpoints.
    *   `session_summary.md`: A detailed log of development progress and decisions.