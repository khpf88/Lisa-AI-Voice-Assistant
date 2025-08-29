class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.port.onmessage = (event) => {
            // Handle messages from the main thread if needed
        };
        this.resampleBuffer = null;
    }

    // Simple linear interpolation resampling function
    _resample(inputBuffer, fromSampleRate, toSampleRate) {
        const inputLength = inputBuffer.length;
        const outputLength = Math.round(inputLength * toSampleRate / fromSampleRate);
        const outputBuffer = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const theoreticalIndex = i * (inputLength - 1) / (outputLength - 1);
            const lowerIndex = Math.floor(theoreticalIndex);
            const upperIndex = Math.ceil(theoreticalIndex);
            const interpolationFactor = theoreticalIndex - lowerIndex;

            if (lowerIndex === upperIndex) {
                outputBuffer[i] = inputBuffer[lowerIndex];
            } else {
                outputBuffer[i] = (1 - interpolationFactor) * inputBuffer[lowerIndex] + interpolationFactor * inputBuffer[upperIndex];
            }
        }
        return outputBuffer;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const audioData = input[0]; // Get the first channel's data
            const targetSampleRate = 16000;

            // Resample the audio data
            const resampledData = this._resample(audioData, sampleRate, targetSampleRate);

            // Send the resampled audio data back to the main thread
            this.port.postMessage(resampledData.buffer, [resampledData.buffer]);
        }
        return true; // Keep the processor alive
    }
}

registerProcessor('audio-processor', AudioProcessor);
