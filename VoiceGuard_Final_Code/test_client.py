import requests
import base64
import sys
import os
import json

# Configuration
API_URL = "http://localhost:8000/api/voice-detection"
API_KEY = "voiceguard-secret-key"

def test_file(file_path, language="English"):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Reading and encoding {file_path}...")
    try:
        with open(file_path, "rb") as f:
            audio_content = f.read()
            audio_base64 = base64.b64encode(audio_content).decode('utf-8')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    payload = {
        "language": language,
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print("\n--- Response ---")
        print(f"Status Code: {response.status_code}")
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_client.py <path_to_mp3_file> [language]")
        print("Example: python test_client.py sample.mp3 Tamil")
    else:
        path = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "English"
        test_file(path, lang)
