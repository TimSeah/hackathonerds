# Speech-to-Text (STT)

Real-time speech-to-text transcription that types directly into your active window.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Basic usage with default settings:
```bash
python stt.py
```

With custom model:
```bash
python stt.py -m large-v2
```

With custom language:
```bash
python stt.py -l es
```

## Arguments

- `-m, --model`: Model size (tiny, base, small, medium, large-v2, etc.) Default: small.en
- `-r, --rt-model`: Real-time model size. Default: tiny.en
- `-l, --lang`: Language code (en, es, fr, etc.) Default: en
- `-d, --root`: Root directory for model downloads

## How It Works

1. Start speaking after running the script
2. Text is transcribed in real-time
3. Completed sentences are automatically typed into your active window
4. Press Ctrl+C to stop

## Configuration

Edit `WRITE_TO_KEYBOARD_INTERVAL` in stt.py to adjust typing speed (0 to disable, 0.002 is fast, 0.05 is slower).