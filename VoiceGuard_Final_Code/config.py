import os

# Security
API_KEY_NAME = "x-api-key"
# In a real scenario, this should be loaded from env vars
# For hackathon ease, we set a default but allow env override
API_KEY = os.getenv("VOICE_GUARD_API_KEY", "voiceguard-secret-key")

# Supported Languages
SUPPORTED_LANGUAGES = {
    "Tamil", "English", "Hindi", "Malayalam", "Telugu"
}

# Supported Formats
SUPPORTED_FORMATS = {"mp3"}
