# VoiceGuard - AI Voice Detection API

This API detects whether an audio file (MP3) is **AI-Generated** or **Human-Spoken**.
It supports 5 languages: Tamil, English, Hindi, Malayalam, Telugu.

## 🚀 Setup Instructions

### 1. Prerequisites
- **Python 3.8+** installed.
- **FFmpeg** installed and added to system PATH.
    - [Download FFmpeg here](https://ffmpeg.org/download.html).
    - Or place `ffmpeg.exe` and `ffprobe.exe` in a folder named `ffmpeg` inside this directory.

### 2. Install Dependencies
Open a terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### 3. Run the Server
You can run the server directly:
```bash
python main.py
```
Or using uvicorn:
```bash
uvicorn main:app --reload
```
The server will start at `http://127.0.0.1:8000`.

## 🧪 How to Test

### Swagger UI (Easy)
1. Open `http://localhost:8000/docs` in your browser.
2. Navigate to `/api/voice-detection`.
3. Click **Try it out**.
4. Set the header `x-api-key` to: `voiceguard-secret-key`
5. Paste your Base64 audio string and language.
6. Execute.

### Command Line
Use the included test client:
```bash
python test_client.py "path/to/audio/file.mp3" Tamil
```

## 🔒 Configuration
- Default API Key: `voiceguard-secret-key`
- Change settings in `config.py` if needed.
