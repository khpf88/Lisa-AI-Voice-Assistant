console.log('script.js parsing started. (New Log)');

class AudioPlayer {
    constructor() {
        this.audio = new Audio();
        this.queue = [];
        this.sourceBuffer = null;
        this.mediaSource = null;
        this.isStreamEnding = false;
        this.initialBuffer = [];
        this.isInitialBufferSent = false;

        // Bind methods to ensure 'this' context is correct
        this._onSourceOpen = this._onSourceOpen.bind(this);
        this._onUpdateEnd = this._onUpdateEnd.bind(this);
        this._onAudioEnded = this._onAudioEnded.bind(this);
        this._onAudioError = this._onAudioError.bind(this);

        this.audio.addEventListener('ended', this._onAudioEnded);
        this.audio.addEventListener('error', this._onAudioError);
        
        this._setupMediaSource();
    }

    _setupMediaSource() {
        if (this.mediaSource) {
            // Clean up previous MediaSource if it exists
            this.mediaSource.removeEventListener('sourceopen', this._onSourceOpen);
        }
        if (this.audio.src) {
            URL.revokeObjectURL(this.audio.src);
        }

        this.mediaSource = new MediaSource();
        this.mediaSource.addEventListener('sourceopen', this._onSourceOpen);
        this.audio.src = URL.createObjectURL(this.mediaSource);
    }

    _onSourceOpen() {
        const mimeType = 'audio/webm; codecs=opus';
        if (MediaSource.isTypeSupported(mimeType)) {
            this.sourceBuffer = this.mediaSource.addSourceBuffer(mimeType);
            this.sourceBuffer.addEventListener('updateend', this._onUpdateEnd);
            this._processQueue(); 
        } else {
            console.error(`Unsupported MIME type: ${mimeType}`);
        }
    }

    _onUpdateEnd() {
        this._processQueue();
    }

    _onAudioEnded() {
        this.updateUIForListening(); 
    }

    _onAudioError(e) {
        console.error("Audio element error:", e);
        // Don't call stop() here to avoid potential infinite loops on error
    }

    _processQueue() {
        if (this.queue.length > 0 && this.sourceBuffer && !this.sourceBuffer.updating) {
            try {
                const chunk = this.queue.shift();
                this.sourceBuffer.appendBuffer(chunk);
                if (this.audio.paused) {
                    this.audio.play().catch(this._onAudioError);
                }
            } catch (e) {
                console.error('Error appending buffer:', e);
            }
        } else if (this.isStreamEnding && this.queue.length === 0 && this.sourceBuffer && !this.sourceBuffer.updating) {
            if (this.mediaSource && this.mediaSource.readyState === 'open') {
                try {
                    this.mediaSource.endOfStream();
                } catch (e) {
                    // Ignore errors as we are cleaning up anyway
                }
            }
        }
    }

    appendBuffer(chunk) {
        if (!this.isInitialBufferSent) {
            this.initialBuffer.push(chunk);
            const totalSize = this.initialBuffer.reduce((acc, val) => acc + val.length, 0);
            if (totalSize > 16384) {
                this.isInitialBufferSent = true;
                const concatenated = new Uint8Array(totalSize);
                let offset = 0;
                for (const buffer of this.initialBuffer) {
                    concatenated.set(buffer, offset);
                    offset += buffer.length;
                }
                this.queue.push(concatenated);
                this.initialBuffer = [];
                this._processQueue();
            }
        } else {
            this.queue.push(chunk);
            this._processQueue();
        }
    }

    stop() {
        console.log("Stopping and cleaning up AudioPlayer.");
        this.queue = [];
        this.isStreamEnding = false;
        this.initialBuffer = [];
        this.isInitialBufferSent = false;

        if (this.sourceBuffer) {
            if (this.sourceBuffer.updating) {
                try {
                    this.sourceBuffer.abort();
                } catch (e) {
                    console.error("Error aborting SourceBuffer:", e);
                }
            }
            this.sourceBuffer.removeEventListener('updateend', this._onUpdateEnd);
            // Detach sourceBuffer from mediaSource
            if (this.mediaSource && this.mediaSource.readyState === 'open') {
                try {
                    this.mediaSource.removeSourceBuffer(this.sourceBuffer);
                } catch (e) {
                    console.error("Error removing SourceBuffer:", e);
                }
            }
        }

        if (this.mediaSource) {
            if (this.mediaSource.readyState === 'open') {
                try {
                    this.mediaSource.endOfStream();
                } catch (e) {
                    // Ignore errors as we are cleaning up anyway
                }
            }
            this.mediaSource.removeEventListener('sourceopen', this._onSourceOpen);
        }

        this.audio.removeEventListener('ended', this._onAudioEnded);
        this.audio.removeEventListener('error', this._onAudioError);
        this.audio.pause();
        this.audio.removeAttribute('src'); // Detach MediaSource from audio element
        this.audio.load(); // Reset audio element

        this.sourceBuffer = null;
        this.mediaSource = null;

        // Re-setup MediaSource for the next stream
        this._setupMediaSource(); // <--- Add this line

        this.updateUIForListening();
    }

    endStream() {
        this.isStreamEnding = true;
        this._processQueue();
    }

    updateUIForListening() {
        const statusP = document.getElementById('status');
        const interruptButton = document.getElementById('interruptButton');
        const waveformVisualizer = document.getElementById('waveform-visualizer');

        if(interruptButton) interruptButton.style.display = 'none';
        if(statusP) statusP.textContent = 'Listening for your voice...';
        if (waveformVisualizer) {
            waveformVisualizer.classList.remove('speaking');
            waveformVisualizer.classList.add('listening');
        }
    }
}

const statusP = document.getElementById('status');
const interruptButton = document.getElementById('interruptButton');
const waveformVisualizer = document.getElementById('waveform-visualizer');

let audioContext;
let ws;
let audioPlayer;

function stopAudioPlayback() {
    console.log('stopAudioPlayback called');
    if (audioPlayer) {
        audioPlayer.stop();
    }
}

async function setupRealtimeCommunication() {
    console.log('setupRealtimeCommunication called');

    // 1. Get microphone access and create an AudioContext
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);

    // 2. Create a ScriptProcessorNode for continuous audio processing.
    // Load the AudioWorklet processor
    await audioContext.audioWorklet.addModule('/static/audio-processor.js');
    const audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');
    
    // 3. Setup WebSocket
    const WS_URL = 'ws://127.0.0.1:8002/ws';
    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.onopen = () => {
        if(statusP) statusP.textContent = 'Listening for your voice...';
        console.log('WebSocket connected');

        if (waveformVisualizer) {
            waveformVisualizer.innerHTML = '<div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div><div class="bar"></div>';
            waveformVisualizer.style.visibility = 'visible';
            waveformVisualizer.classList.add('listening');
        }
        // Connect the audio processing graph
        source.connect(audioWorkletNode);
        audioWorkletNode.connect(audioContext.destination); // Connect to destination to keep the node alive
    };

    // 4. Handle incoming messages (TTS audio and transcriptions)
    ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
            if(statusP) statusP.textContent = `Lisa is speaking...`;
            if (waveformVisualizer) {
                waveformVisualizer.classList.remove('listening');
                waveformVisualizer.classList.add('speaking');
            }
            if(interruptButton) interruptButton.style.display = 'block';
            audioPlayer.appendBuffer(new Uint8Array(event.data));
        } else if (typeof event.data === 'string') {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'transcription') {
                    console.log("Transcription received:", message.text);
                    if(statusP) statusP.textContent = 'Lisa is thinking...';
                    if (audioPlayer) {
                        audioPlayer.stop(); // <--- Add this line to ensure clean state
                        audioPlayer.queue = [];
                        audioPlayer.isStreamEnding = false;
                        audioPlayer.initialBuffer = [];
                        audioPlayer.isInitialBufferSent = false;
                    }
                } else if (message.type === 'session_info') {
                    const { system_specs, model_info, accumulated_tokens } = message.data;
                    const specsInfo = document.getElementById('specs-info');
                    const modelInfoDiv = document.getElementById('model-info');
                    const tokensInfo = document.getElementById('tokens-info');

                    if (specsInfo) {
                        specsInfo.textContent = `CPU: ${system_specs.cpu_info} | GPU: ${system_specs.gpu_info} | RAM: ${system_specs.ram_gb}`;
                    }
                    if (modelInfoDiv) {
                        modelInfoDiv.innerHTML = `<span>STT: ${model_info.stt}</span> | <span>LLM: ${model_info.llm}</span> | <span>TTS: ${model_info.tts}</span>`;
                    }
                    if (tokensInfo) {
                        tokensInfo.textContent = `Session Tokens: ${accumulated_tokens}`;
                    }
                } else if (message.type === 'token_update') {
                    const { accumulated_tokens } = message.data;
                    const tokensInfo = document.getElementById('tokens-info');
                    if (tokensInfo) {
                        tokensInfo.textContent = `Session Tokens: ${accumulated_tokens}`;
                    }
                } else if (message.text === 'EOS') {
                    console.log('End of audio stream received.');
                    if (audioPlayer) {
                        audioPlayer.endStream();
                    }
                }
            } catch (e) {
                // It might be the EOS string, which is not JSON
                if (event.data === 'EOS') {
                    console.log('End of audio stream received.');
                    if (audioPlayer) {
                        audioPlayer.endStream();
                    }
                } else {
                    console.log("Received non-JSON string message:", event.data);
                }
            }
        }
    };

    // 5. Send audio data to the server
    audioWorkletNode.port.onmessage = (event) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            
            const inputData = new Float32Array(event.data);
            // The server will handle resampling and conversion, so we send float32
            ws.send(inputData.buffer);
        }
    };

    ws.onclose = () => {
        if(statusP) statusP.textContent = 'Disconnected. Please refresh.';
        console.log('WebSocket disconnected');
        // Stop audio processing
        audioWorkletNode.disconnect();
        source.disconnect();
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if(statusP) statusP.textContent = 'WebSocket error. Check console.';
    };
}



function startApp() {
    console.log('startApp function called by click.');
    document.removeEventListener('click', startApp);
    
    async function initialize() {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            await audioContext.resume();
            
            audioPlayer = new AudioPlayer(); // Re-introduce initial creation

            await setupRealtimeCommunication();
        } catch (error) {
            console.error('Error during initialization:', error);
            if(statusP) statusP.textContent = 'Initialization failed. Check console.';
        }
    }
    initialize();
}

window.onload = () => {
    console.log('window.onload fired. (New Log)');
    if (statusP) {
        statusP.textContent = 'Click anywhere to start';
        document.addEventListener('click', startApp);
        console.log('Click event listener attached.');
        if(interruptButton) interruptButton.addEventListener('click', stopAudioPlayback);
    } else {
        console.error('#status element not found!');
    }
};