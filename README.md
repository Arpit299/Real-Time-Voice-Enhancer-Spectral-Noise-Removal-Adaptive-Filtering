# Real-Time Voice Enhancer — Spectral Noise Removal & Adaptive Filtering

Real-Time Voice Enhancer is a Python-based audio processing system designed to reduce background noise and improve speech clarity using spectral analysis, adaptive filtering, and signal-processing techniques.

The system supports real-time microphone processing as well as offline WAV enhancement.

## Features

- Real-time microphone noise removal
- Spectral noise suppression using FFT
- Adaptive NLMS filtering
- High-pass filtering for low-frequency noise
- Automatic gain control
- Background-noise calibration
- WAV input and output processing
- Real-time audio streaming
- Configurable processing parameters
- Built-in self-test
- Numerical stability checks
- Lightweight single-file architecture

## Tech Stack

**Python | NumPy | SoundDevice | FFT | Digital Signal Processing | Adaptive Filtering**

## Signal Processing Pipeline

```text
Microphone Input
       ↓
Audio Buffering
       ↓
High-Pass Filtering
       ↓
FFT Spectral Analysis
       ↓
Noise Spectrum Estimation
       ↓
Spectral Noise Suppression
       ↓
Adaptive NLMS Filtering
       ↓
Automatic Gain Control
       ↓
Enhanced Voice Output
```

## DSA Used

The project combines signal-processing algorithms with practical data structures:

- `deque` for bounded adaptive signal history
- Circular/buffered audio processing
- Sliding-window spectral analysis
- Vectorized numerical operations
- Adaptive coefficient arrays
- Frame-based streaming architecture

## Core Algorithms

### FFT Spectral Analysis

Fast Fourier Transform converts the time-domain audio signal into frequency-domain components, allowing the system to estimate and suppress unwanted frequency energy.

### Spectral Noise Suppression

The system maintains a noise spectrum estimate and calculates a frequency-dependent suppression mask.

Conceptually:

```text
Enhanced Spectrum = Input Spectrum × Suppression Mask
```

### NLMS Adaptive Filtering

Normalized Least Mean Squares continuously adapts filter coefficients according to the incoming signal.

The normalized update helps keep the adaptation stable across changes in signal amplitude.

### High-Pass Filtering

Low-frequency components that commonly contain rumble, vibration, and environmental noise are attenuated before further processing.

### Automatic Gain Control

The final signal level is normalized while maintaining a safe output peak to prevent excessive amplification and clipping.

## Installation

Clone the repository:

```bash
git clone https://github.com/Arpit299/realtime-voice-enhancer.git
cd realtime-voice-enhancer
```

Install dependencies:

```bash
pip install numpy sounddevice
```

## Run Self-Test

```bash
python realtime_voice_enhancer.py --self-test
```

Example:

```text
SELF-TEST: PASS
Input SNR: 2.88 dB
Enhanced SNR: 4.29 dB
Output Peak: 0.8376
```

## Run in Real Time

```bash
python realtime_voice_enhancer.py --realtime
```

The system performs a short background-noise calibration before beginning continuous microphone processing.

For reliable real-time operation, headphones are recommended to prevent acoustic feedback.

## Process a WAV File

```bash
python realtime_voice_enhancer.py --input noisy.wav --output enhanced.wav
```

The system reads the input WAV file, processes the audio frame by frame, and writes the enhanced result to a new WAV file.

## Example Workflow

```text
Noisy Microphone Signal
          ↓
Noise Calibration
          ↓
Frame Extraction
          ↓
High-Pass Filtering
          ↓
FFT
          ↓
Noise Spectrum Estimation
          ↓
Spectral Suppression
          ↓
NLMS Adaptive Filtering
          ↓
Gain Control
          ↓
Cleaned Voice
```

## Project Structure

```text
realtime-voice-enhancer/
│
├── realtime_voice_enhancer.py
└── README.md
```

## Performance Validation

The built-in synthetic test evaluates the enhancement pipeline using a controlled noisy signal.

Validated checks include:

- Successful signal processing
- SNR calculation
- Enhancement output generation
- Peak-level safety
- WAV encoding and decoding
- Numerical stability
- CLI execution
- Python syntax compilation

## Applications

- Real-time voice cleanup
- Online meetings
- Voice recording
- Podcast preprocessing
- Speech enhancement research
- Audio DSP experimentation
- Voice-controlled applications
- Microphone quality improvement
- Background-noise reduction experiments

## Limitations

This project is designed as a lightweight educational and engineering implementation rather than a studio-grade neural speech-enhancement system.

Performance depends on:

- Microphone quality
- Background-noise characteristics
- Room acoustics
- Calibration quality
- Processing parameters
- CPU performance

Highly non-stationary noise or overlapping speech may require more advanced machine-learning-based enhancement methods.

## Future Improvements

Potential extensions include:

- Deep-learning speech enhancement
- Voice activity detection
- Multi-band adaptive filtering
- Automatic noise-type classification
- GPU acceleration
- WebRTC integration
- Spectrogram visualization
- Multi-microphone processing
- Voice-preserving dereverberation
- Real-time quality metrics

