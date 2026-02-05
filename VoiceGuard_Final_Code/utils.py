import base64
import io
import os
import sys
import numpy as np
from pydub import AudioSegment

# Setup FFmpeg path if not in system PATH
# Check if ffmpeg is in the parent directory (based on previous context)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG_DIR = os.path.join(PARENT_DIR, "ffmpeg-master-latest-win64-gpl", "bin")

if os.path.exists(FFMPEG_DIR):
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
    AudioSegment.converter = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
    AudioSegment.ffprobe = os.path.join(FFMPEG_DIR, "ffprobe.exe")

def decode_audio(base64_string: str) -> np.ndarray:
    """
    Decodes a base64 MP3 string into a normalized floating point numpy array.
    """
    try:
        # Decode base64
        audio_bytes = base64.b64decode(base64_string)
        
        # Load into pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
        
        # Resample to 22050 Hz (standard for librosa analysis)
        audio_segment = audio_segment.set_frame_rate(22050).set_channels(1)
        
        # Convert to numpy array
        # pydub audio is int16, we want float32 centered at 0
        samples = np.array(audio_segment.get_array_of_samples())
        
        if audio_segment.sample_width == 2:
            data = samples.astype(np.float32) / 32768.0
        elif audio_segment.sample_width == 4:
            data = samples.astype(np.float32) / 2147483648.0
        else:
            # Fallback for 8-bit or other
            data = samples.astype(np.float32) / float(1 << (8 * audio_segment.sample_width - 1))
            
        # Robustness: Check for minimum length (e.g., 0.5 seconds at 22050Hz)
        if len(data) < 22050 * 0.5:
            raise ValueError("Audio processing failed: Audio too short (min 0.5s required)")

        return data, 22050
        
    except Exception as e:
        raise ValueError(f"Failed to process audio: {str(e)}")
